
from parameter_definitions import *
from implementation import Implementation as imp, PipelineConfig

from dataflow import *
from sympy import Lambda, Symbol

import unittest

class Pipeline:

    def __init__(self, tp):

        # Set telescope parameters
        self.tp = tp

        self.Nisland = tp.minimum_channels
        self.Mdbl = 8
        self.Mcpx = self.Mdbl * 2

        self._create_domains()
        self._create_dataflow()

    def _create_domains(self):
        tp = self.tp

        # Make baseline domain
        self.baseline = Domain('Baseline')
        usedBls = tp.nbaselines
        self.allBaselines = self.baseline.regions(usedBls)
        if isinstance(tp.Bmax_bins, Symbol):
            b = Symbol("b")
            self.binBaselines = self.allBaselines.split(tp.nbaselines, sym_props={
                'bmax': Lambda(b, Symbol("B_max")(b)),
                'size': 1
            })
        else:
            self.binBaselines = self.allBaselines.split(len(tp.Bmax_bins), props={
                'bmax': lambda i: tp.Bmax_bins[i],
                'size': lambda i: tp.nbaselines_full * tp.frac_bins[i]
            })

        # Make time domain
        self.time = Domain('Time', 's')
        self.obsTime = self.time.regions(tp.Tobs)
        self.dumpTime = self.obsTime.split(tp.Tobs / tp.Tdump_ref)
        self.snapTime = self.obsTime.split(tp.Tobs / tp.Tsnap)
        self.kernelPredTime = self.obsTime.split(
            lambda rbox: tp.Tobs / tp.Tkernel_predict(rbox(self.baseline,'bmax')))
        self.kernelBackTime = self.obsTime.split(
            lambda rbox: tp.Tobs / tp.Tkernel_backward(rbox(self.baseline,'bmax')))

        # Make frequency domain
        self.frequency = Domain('Frequency', 'ch')
        self.allFreqs = self.frequency.regions(tp.Nf_max)
        self.eachFreq = self.allFreqs.split(tp.Nf_max)
        self.visFreq = self.allFreqs.split(tp.Nf_vis)
        self.outFreqs = self.allFreqs.split(tp.Nf_out)
        self.islandFreqs = self.allFreqs.split(self.Nisland)
        self.predFreqs = self.allFreqs.split(
            lambda rbox: tp.Nf_vis_predict(rbox(self.baseline,'bmax')))
        self.backFreqs = self.allFreqs.split(
            lambda rbox: tp.Nf_vis_backward(rbox(self.baseline,'bmax')))
        self.gcfPredFreqs = self.allFreqs.split(
            lambda rbox: tp.Nf_gcf_predict(rbox(self.baseline,'bmax')))
        self.gcfBackFreqs = self.allFreqs.split(
            lambda rbox: tp.Nf_gcf_backward(rbox(self.baseline,'bmax')))
        self.fftPredFreqs = self.allFreqs.split(tp.Nf_FFT_predict)
        self.fftBackFreqs = self.allFreqs.split(tp.Nf_FFT_backward)
        self.projFreqs = self.allFreqs.split(tp.Nf_proj)

        # Make beam domain
        self.beam = Domain('Beam')
        self.allBeams = self.beam.regions(tp.Nbeam)
        self.eachBeam = self.allBeams.split(tp.Nbeam)

        # Make polarisation domain
        self.polar = Domain('Polarisation')
        self.iquvPolars = self.polar.regions(tp.Npp)
        self.iquvPolar = self.iquvPolars.split(tp.Npp)
        self.xyPolars = self.polar.regions(tp.Npp)
        self.xyPolar = self.xyPolars.split(tp.Npp)

        # Make (major) loop domain
        self.loop = Domain('Major Loop')
        self.allLoops = self.loop.regions(tp.Nmajortotal)
        self.eachSelfCal = self.allLoops.split(tp.Nselfcal + 1)
        self.eachLoop = self.allLoops.split(tp.Nmajortotal)

        # Make facet domain
        self.facet = Domain('Facet')
        self.allFacets = self.facet.regions(tp.Nfacet**2)
        self.eachFacet = self.allFacets.split(tp.Nfacet**2)

        # Make taylor term domain
        self.taylor = Domain('Taylor')
        self.allTaylor = self.taylor.regions(tp.number_taylor_terms)
        self.eachTaylor = self.allTaylor.split(tp.number_taylor_terms)
        self.predTaylor = self.allTaylor.split(tp.Ntaylor_predict)
        self.backTaylor = self.allTaylor.split(tp.Ntaylor_backward)

    def _transfer_cost_vis(self, Tdump):
        """Utility transfer cost function for visibility data. Multiplies out
        frequency, baselines, polarisations and time given a dump time"""

        # Dump time might depend on baseline...
        if isinstance(Tdump, Lambda):
            Tdump_ = lambda rbox: Tdump(rbox(self.baseline, 'bmax'))
        else:
            Tdump_ = lambda rbox: Tdump
        return (lambda rbox:
          self.tp.Mvis * rbox(self.frequency, 'size')
                       * rbox(self.baseline, 'size')
                       * rbox(self.polar, 'size')
                       * rbox(self.time, 'size')
                       / Tdump_(rbox))

    def _create_dataflow(self):
        """ Creates common data flow nodes """

        self.tm = Flow('Telescope Management', cluster = 'interface')

        self.lsm = Flow('Local Sky Model', attrs = {'pos':'0 0'},
                        deps = [self.tm], cluster = 'interface')

        self.uvw = Flow(
            'UVW generation',
            [self.eachBeam, self.eachFacet, self.snapTime, self.islandFreqs,
             self.allBaselines],
            deps = [self.tm],
            costs = {
                'transfer': lambda rbox:
                  3 * 8 * rbox(self.frequency, 'size')
                        * rbox(self.baseline, 'size')
                        * rbox(self.time, 'size')
                        / self.tp.Tdump_scaled
            })

        self.corr = Flow(
            'Correlator',
            [self.allBeams, self.dumpTime,
             self.eachFreq, self.xyPolars, self.allBaselines],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)})

        # TODO - buffer contains averaged visibilities...
        self.buf = Flow(
            'Buffer',
            [self.islandFreqs, self.allBaselines, self.snapTime, self.xyPolar],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)}
        )

    def _cost_from_product(self, rbox, product, cost):

        # Get time field
        product_costs = self.tp.products.get(product, {})
        t = product_costs.get("T", 0)

        # Baseline-dependent task?
        if product_costs.has_key(cost + "_task"):
            task_cost = product_costs[cost + "_task"]
            # Pass baseline length, multiply by number of baselines
            return rbox(self.baseline,'size') * \
                t(rbox(self.baseline,'bmax')) * \
                task_cost(rbox(self.baseline,'bmax'))
        else:
            # Otherwise simply return the value as-is
            return t * product_costs.get(cost, 0) / product_costs.get("N", 1)

    def _costs_from_product(self, product):
        return {
            'compute': lambda rbox: self._cost_from_product(rbox, product, 'Rflop'),
            'transfer': lambda rbox: self._cost_from_product(rbox, product, 'Rout'),
        }

    def create_ingest(self):

        ingest = Flow(
            'Receive',
            [self.eachBeam, self.snapTime,
             self.islandFreqs, self.xyPolar, self.allBaselines],
            deps = [self.tm, self.corr],
            cluster='ingest',
            costs = self._costs_from_product(Products.Receive))

        demix = Flow(
            'Demix',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar],
            deps = [ingest],
            cluster='ingest',
            costs = self._costs_from_product(Products.Demix))

        average = Flow(
            'Average',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar, self.allBaselines],
            deps = [demix],
            cluster='ingest',
            costs = self._costs_from_product(Products.Average))

        rfi = Flow(
            'Flagging',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar, self.allBaselines],
            deps = [average],
            cluster='ingest',
            costs = self._costs_from_product(Products.Flag))

        buf = Flow('Buffer', [self.islandFreqs], deps=[rfi])

        return (ingest, demix, average, rfi, buf)

    def create_flagging(self,vis):

        return Flow(
            'Flagging',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [vis],
            costs = self._costs_from_product(Products.Flag))

    def create_calibrate(self,vis,model):

        # No calibration?
        if Products.Solve in self.tp.products:

            # Source finding
            if self.tp.pipeline == Pipelines.ICAL:
                source_find = Flow(
                    'Source Finding',
                    [self.eachSelfCal],
                    costs = self._costs_from_product(Products.Source_Find),
                    deps = [model], cluster='calibrate',
                )
                sources = [source_find]
            else:
                sources = []
                source_find = None

            # DFT Predict
            dft = Flow(
                'DFT',
                [self.eachBeam, self.eachSelfCal, self.xyPolar, self.snapTime, self.visFreq],
                costs = self._costs_from_product(Products.DFT),
                deps = sources, cluster='calibrate',
            )

            # Solve
            solve = Flow(
                'Calibration',
                [self.eachBeam, self.eachSelfCal, self.snapTime, self.allFreqs],
                costs = self._costs_from_product(Products.Solve),
                deps = [dft, vis], cluster='calibrate',
            )
            calibration = solve

        else:

            source_find = None
            dft = None
            solve = None

            # Load calibration
            calibration = Flow(
                'Calibration Buffer',
                [self.eachBeam, self.eachSelfCal, self.snapTime, self.allFreqs],
                cluster='calibrate',
            )

        # Apply the calibration
        app = Flow(
            'Correct',
            [self.snapTime, self.islandFreqs, self.eachBeam,
             self.allLoops, self.xyPolar],
            deps = [vis, calibration], cluster='calibrate',
            costs = self._costs_from_product(Products.Correct)
        )

        return (source_find, dft, solve, app)

    def create_predict(self,vis,model):
        """ Predicts visibilities from a model """

        if self.tp.scale_predict_by_facet:
            predictFacets = self.eachFacet
        else:
            predictFacets = self.allFacets

        # Predict
        fft = Flow(
            'IFFT',
            # FIXME: eachLoop and xyPolar might be wrong
            [self.eachBeam, self.eachLoop, predictFacets, self.xyPolar,
             self.snapTime, self.fftPredFreqs],
            costs = self._costs_from_product(Products.IFFT),
            deps = [model], cluster='predict',
        )

        gcf = Flow(
            'Degrid w-kernels',
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.kernelPredTime, self.gcfPredFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Degridding_Kernel_Update),
            deps = [vis], cluster='predict'
        )

        degrid = Flow(
            "Degrid",
            [self.eachBeam, self.eachLoop, predictFacets, self.xyPolar,
             self.snapTime, self.predFreqs, self.binBaselines,
             self.predTaylor],
            costs = self._costs_from_product(Products.Degrid),
            deps = [fft, gcf], cluster='predict'
        )

        return (fft, gcf, degrid)

    def create_rotate(self,vis,backward):
        """ Rotates the phase center of visibilities for facetting """

        # Select the right cost function
        product = Products.PhaseRotation
        facets = self.eachFacet
        cluster = 'backward'
        if not backward:
            product = Products.PhaseRotationPredict
            assert(self.tp.scale_predict_by_facet)
            facets = self.allFacets
            cluster = 'predict'

        # TODO: This product is for backward *and* forward!
        rotate = Flow(
            "Rotate Phase",
            [self.eachBeam, self.eachLoop, facets, self.xyPolar,
             self.snapTime, self.islandFreqs, self.binBaselines],
            costs = self._costs_from_product(product),
            deps = [vis], cluster = cluster
        )

        return rotate

    def create_project(self, facets, regLoop):
        """ Reprojects facets so they can be seen as part of the same image plane """

        # No reprojection?
        if not Products.Reprojection in self.tp.products:
            return facets

        return Flow(
            'Project',
            [self.eachBeam, regLoop, self.xyPolar,
             self.snapTime, self.projFreqs, self.eachFacet],
            costs = self._costs_from_product(Products.Reprojection),
            deps = [facets],
            cluster = 'backend'
        )

    def create_subtract(self, vis, model_vis):
        """ Subtracts model visibilities from measured visibilities """

        return Flow(
            "Subtract",
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.islandFreqs, self.allBaselines],
            costs = self._costs_from_product(Products.Subtract_Visibility),
            deps = [vis, model_vis],
        )

    def create_backward(self, vis, uvw):
        """ Creates dirty image from visibilities """

        regLoop = self.eachLoop
        fftFreqs = self.fftBackFreqs

        tp = self.tp
        gcf = Flow(
            'Grid w-kernels',
            [self.eachBeam, regLoop, self.xyPolar,
             self.kernelBackTime, self.gcfBackFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Gridding_Kernel_Update),
            deps = [uvw], cluster = 'backward',
        )

        grid = Flow(
            'Grid',
            [self.eachBeam, regLoop, self.eachFacet, self.xyPolar,
             self.snapTime, self.backFreqs, self.binBaselines,
             self.backTaylor],
            costs = self._costs_from_product(Products.Grid),
            deps = [vis, gcf], cluster = 'backward',
        )

        fft = Flow(
            'FFT',
            # FIXME: eachLoop and xyPolar might be wrong?
            [self.eachBeam, self.eachLoop, self.eachFacet,
             self.xyPolar, self.snapTime, self.fftBackFreqs],
            costs = self._costs_from_product(Products.FFT),
            deps = [grid], cluster = 'backward',
        )

        return (fft, gcf, grid)

    def create_clean(self, dirty):

        # Skip for fast imaging
        if not Products.Subtract_Image_Component in self.tp.products:
            return (None, None, dirty)

        if self.tp.pipeline != Pipelines.DPrepA_Image:
            spectral_fit = dirty
        else:
            spectral_fit = Flow(
                'Spectral Fitting',
                [self.eachBeam, self.eachLoop,
                 self.xyPolar, self.obsTime, self.eachTaylor],
                costs = self._costs_from_product(Products.Image_Spectral_Fitting),
                deps = [dirty], cluster = 'backend',
            )

        identify =  Flow(
            'Identify',
            [self.eachBeam, self.eachLoop,
             self.obsTime, self.allFreqs],
            costs = self._costs_from_product(Products.Identify_Component),
            deps = [spectral_fit],
            cluster = 'backend'
        )

        subtract =  Flow(
            'Subtract Component',
            [self.eachBeam, self.eachLoop,
             self.obsTime, self.allFreqs],
            costs = self._costs_from_product(Products.Subtract_Image_Component),
            deps = [dirty,identify],
            cluster = 'backend'
        )

        return (spectral_fit, identify, subtract)

    def create_update(self, lsm):

        return Flow(
            'Update LSM',
            [self.eachLoop],
            deps = [lsm],
        )

    def create_imaging(self):

        # LSM might get updated per major loop
        lsm = self.lsm
        has_clean = Products.Subtract_Image_Component in self.tp.products
        if has_clean:
            lsm = self.create_update(lsm)

        # UVWs are supposed to come from TM
        uvws = self.uvw
        # Visibilities from buffer, flagged
        vis = self.create_flagging(self.buf)

        # Calibrate
        vis_calibrated = self.create_calibrate(vis, lsm)[-1]

        # Predict
        degrid = self.create_predict(uvws, lsm)[-1]
        if self.tp.scale_predict_by_facet:
            predVis = self.create_rotate(degrid, False)
        else:
            predVis = degrid

        # Subtract
        subtract = self.create_subtract(vis_calibrated, predVis)

        # Phase rotate
        rotate = self.create_rotate(subtract, True)

        # Intermediate loops
        fftBack = self.create_backward(rotate, uvws)[0]
        project = self.create_project(fftBack, self.eachLoop)
        if has_clean:
            (_, _, clean) = self.create_clean(project)
            lsm.depend(clean, 0)
            out = clean
        else:
            out = project

        return out.recursiveDeps()

    def create_pipeline(self):

        if self.tp.pipeline in Pipelines.imaging:
            return self.create_imaging()

        if self.tp.pipeline == Pipelines.Ingest:
            return self.create_ingest()

class PipelineTestsBase(unittest.TestCase):
    """ Common helpers for pipeline test cases """

    def _loadTelParams(self, pipeline):

        cfg = PipelineConfig(telescope=Telescopes.SKA1_Mid,
                             band=Bands.Mid1,
                             pipeline=pipeline,
                             max_baseline=150000,
                             Nf_max='default',
                             blcoal=True,
                             on_the_fly=True,
                             scale_predict_by_facet=True)
        adjusts = {
            'Nfacet': 8,
            'Tsnap': 40
        }

        tp = imp.calc_tel_params(cfg, adjusts=adjusts)
        self.df = Pipeline(tp)

    def _assertEqualProduct(self, flow, product):
        """Checks whether the given flow has the same compute and transfer
        rate as the product in our telescope parameters. If the
        product is undefined, the flow should be None.
        """

        if product not in self.df.tp.products:
            # Product not defined? Make sure there's no node for it either
            fname = "" if flow is None else flow.name # ...
            self.assertIsNone(flow, msg="Product %s does not exist, but Flow %s does!" % (product, fname))
        else:
            self.assertIsNotNone(flow, msg="Product %s exists, but no associated Flow defined!" % product)
            # Otherwise match compute & transfer cost
            self.assertAlmostEqual(
                float(flow.cost('compute')/self.df.tp.Tobs),
                float(self.df.tp.products[product]['Rflop']),
                delta = self.df.tp.products[product]['Rflop'] / 1e10)
            self.assertAlmostEqual(
                float(flow.cost('transfer')/self.df.tp.Tobs),
                float(self.df.tp.products[product]['Rout']),
                delta = self.df.tp.products[product]['Rout'] / 1e10)

    def _assertPipelineComplete(self, flows):
        """Checks whether the given flow has the same compute and transfer
        rate as the product in our telescope parameters. If the
        product is undefined, the flow should be None.
        """

        # Sort costs, match descending
        def flowCost(f): return -f.cost('compute')
        flowsSorted = sorted(flows, key=flowCost)
        def productsCost((n,c)): return -c['Rflop']
        productsSorted = sorted(self.df.tp.products.items(), key=productsCost)

        # Zip, match and sum
        costSum = 0
        for flow, (pname, pcosts) in zip(flowsSorted, productsSorted):
            fcost = float(flow.cost('compute')/self.df.tp.Tobs)
            pcost = float(pcosts['Rflop'])
            costSum += pcost
            self.assertAlmostEqual(
                fcost, pcost, delta=pcost/1e10,
                msg="Flow %s cost does not match product %s (%f != %f, factor %f)!\n\n"
                    "This often means that some products don't "
                    "have a matching flow (or vice-versa)\n"
                    "Flow list: %s\nProduct list: %s" % (
                        flow.name, pname, fcost, pcost, pcost/max(fcost, 0.001),
                        map(lambda f: f.name, flowsSorted),
                        map(lambda (n,c): n, productsSorted))
            )

        # Finally check sum
        self.assertAlmostEqual(
            float(costSum),
            float(self.df.tp.Rflop),
            delta = self.df.tp.Rflop/1e10)

class PipelineTestsImaging(PipelineTestsBase):
    """Tests the data flows constructed from the parametric model for
    consistency. This means we both sanity-check the construction as
    well as whether our predictions match the ones on the parametric
    model where there's overlap. This should allow us to ensure that
    we inferred the right architecture.
    """

    def setUp(self):
        self._loadTelParams(Pipelines.ICAL)

    def test_baseline_domain(self):

        # Check that size of baseline region sets sums up to all
        # baselines. This is a fairly important requirement for the
        # math to work out...
        def blSize(rb): return rb(self.df.baseline, 'size')
        self.assertEqual(self.df.allBaselines.sum(blSize),
                         self.df.allBaselines.size)
        self.assertAlmostEqual(self.df.binBaselines.sum(blSize),
                               self.df.allBaselines.size)

    def test_time_domain(self):

        def timeSize(rb): return rb(self.df.time, 'size')
        def ntimeSize(rb): return -timeSize(rb)
        tp = self.df.tp

        # Dump time and snap time
        self.assertAlmostEqual(self.df.dumpTime.max(timeSize),
                               self.df.tp.Tdump_ref)
        self.assertAlmostEqual(self.df.snapTime.max(timeSize),
                               self.df.tp.Tsnap)

        # Kernel times can be BL-dependent
        rboxes = RegionBoxes([self.df.binBaselines, self.df.kernelPredTime])
        self.assertAlmostEqual(rboxes.max(timeSize),
                               tp.Tkernel_predict(tp.Bmax_bins[0]))
        self.assertAlmostEqual(-rboxes.max(ntimeSize),
                               tp.Tkernel_predict(tp.Bmax_bins[-1]))

        rboxes = RegionBoxes([self.df.binBaselines, self.df.kernelBackTime])
        self.assertAlmostEqual(rboxes.max(timeSize),
                               tp.Tkernel_backward(tp.Bmax_bins[0]))
        self.assertAlmostEqual(-rboxes.max(ntimeSize),
                               tp.Tkernel_backward(tp.Bmax_bins[-1]))

    def test_ingest(self):
        if self.df.tp.pipeline != Pipelines.Ingest: return
        (ingest, rfi, demix, average, _) = self.df.create_ingest()
        self._assertEqualProduct(ingest, Products.Receive)
        self._assertEqualProduct(rfi, Products.Flag)
        self._assertEqualProduct(demix, Products.Demix)
        self._assertEqualProduct(average, Products.Average)

    def test_flag(self):
        if self.df.tp.pipeline == Pipelines.Ingest: return
        rfi = self.df.create_flagging(None)
        self._assertEqualProduct(rfi, Products.Flag)

    def test_grid(self):
        (fftPred, gcfPred, gridPred) = self.df.create_predict(None, None)
        (fftBack, gcfBack, gridBack) = self.df.create_backward(None, None)
        self._assertEqualProduct(fftBack, Products.FFT)
        self._assertEqualProduct(fftPred, Products.IFFT)
        self._assertEqualProduct(gridPred, Products.Degrid)
        self._assertEqualProduct(gridBack, Products.Grid)
        self._assertEqualProduct(gcfPred, Products.Degridding_Kernel_Update)
        self._assertEqualProduct(gcfBack, Products.Gridding_Kernel_Update)

    def test_rotate(self):
        rotBack = self.df.create_rotate(None,backward=True)
        rotPred = self.df.create_rotate(None,backward=False)
        self._assertEqualProduct(rotBack, Products.PhaseRotation)
        self._assertEqualProduct(rotPred, Products.PhaseRotationPredict)

    def test_project(self):
        if Products.Reprojection in self.df.tp.products:
            proj = self.df.create_project(None, self.df.eachLoop)
            self._assertEqualProduct(proj, Products.Reprojection)

    def test_calibrate(self):
        (source_find, dft, solve, app) = self.df.create_calibrate(None,None)
        self._assertEqualProduct(source_find, Products.Source_Find)
        self._assertEqualProduct(dft, Products.DFT)
        self._assertEqualProduct(solve, Products.Solve)

    def test_subtract(self):
        subtract = self.df.create_subtract(None, None)
        self._assertEqualProduct(subtract, Products.Subtract_Visibility)

    def test_clean(self):
        (spectral_fit, identify, subtract) = self.df.create_clean(None)
        self._assertEqualProduct(spectral_fit, Products.Image_Spectral_Fitting)
        self._assertEqualProduct(identify, Products.Identify_Component)
        self._assertEqualProduct(subtract, Products.Subtract_Image_Component)

    def test_imaging(self):

        # Check whole pipeline
        self.assertIn(self.df.tp.pipeline, Pipelines.imaging)
        self._assertPipelineComplete(self.df.create_imaging())

class PipelineTestsRCAL(PipelineTestsImaging):
    def setUp(self):
        self._loadTelParams(Pipelines.RCAL)

class PipelineTestsFastImg(PipelineTestsImaging):
    def setUp(self):
        self._loadTelParams(Pipelines.Fast_Img)

class PipelineTestsDPrepA(PipelineTestsImaging):
    def setUp(self):
        self._loadTelParams(Pipelines.DPrepA)

class PipelineTestsDPrepA_Image(PipelineTestsImaging):
    def setUp(self):
        self._loadTelParams(Pipelines.DPrepA_Image)

class PipelineTestsDPrepC(PipelineTestsImaging):
    def setUp(self):
        self._loadTelParams(Pipelines.DPrepC)

class PipelineTestsIngest(PipelineTestsBase):
    def setUp(self):
        self._loadTelParams(Pipelines.Ingest)

    def test_ingest(self):

        # Check ingest pipeline
        self.assertEqual(self.df.tp.pipeline, Pipelines.Ingest)
        self._assertPipelineComplete(self.df.create_ingest())

if __name__ == '__main__':
    unittest.main()
