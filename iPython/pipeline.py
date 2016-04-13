
from parameter_definitions import *
from implementation import Implementation as imp, PipelineConfig

from dataflow import *
from sympy import Lambda

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
        self.allBaselines = self.baseline.region(usedBls)
        self.binBaselines = self.allBaselines.split(len(tp.Bmax_bins), props={
            'bmax': lambda i: tp.Bmax_bins[i],
            'size': lambda i: tp.nbaselines_full * tp.frac_bins[i]
        })

        # Make time domain
        self.time = Domain('Time', 's')
        self.obsTime = self.time.region(tp.Tobs)
        self.dumpTime = self.obsTime.split(tp.Tobs / tp.Tdump_ref)
        self.snapTime = self.obsTime.split(tp.Tobs / tp.Tsnap)
        self.kernelPredTime = self.obsTime.split(
            lambda rbox: tp.Tobs / tp.Tkernel_predict(rbox(self.baseline,'bmax')))
        self.kernelBackTime = self.obsTime.split(
            lambda rbox: tp.Tobs / tp.Tkernel_backward(rbox(self.baseline,'bmax')))

        # Make frequency domain
        self.frequency = Domain('Frequency', 'ch')
        self.allFreqs = self.frequency.region(tp.Nf_max)
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

        # Make beam domain
        self.beam = Domain('Beam')
        self.allBeams = self.beam.region(tp.Nbeam)
        self.eachBeam = self.allBeams.split(tp.Nbeam)

        # Make polarisation domain
        self.polar = Domain('Polarisation')
        self.iquvPolars = self.polar.region(tp.Npp)
        self.iquvPolar = self.iquvPolars.split(tp.Npp)
        self.xyPolars = self.polar.region(tp.Npp)
        self.xyPolar = self.xyPolars.split(tp.Npp)

        # Make (major) loop domain
        self.loop = Domain('Major Loop')
        self.allLoops = self.loop.region(tp.Nmajortotal)
        self.eachSelfCal = self.allLoops.split(tp.Nselfcal + 1)
        self.eachLoop = self.allLoops.split(tp.Nmajortotal)

        # Make facet domain
        self.facet = Domain('Facet')
        self.allFacets = self.facet.region(tp.Nfacet**2)
        self.eachFacet = self.allFacets.split(tp.Nfacet**2)

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
            'Ingest',
            [self.eachBeam, self.dumpTime,
             self.islandFreqs, self.xyPolars, self.allBaselines],
            deps = [self.tm, self.corr],
            cluster='ingest',
            costs = self._costs_from_product(Products.Receive))

        rfi = Flow(
            'Ingest Flagging',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [ingest],
            cluster='ingest',
            costs = self._costs_from_product(Products.Flag))

        demix = Flow(
            'Demix',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars],
            deps = [rfi],
            cluster='ingest',
            costs = self._costs_from_product(Products.Demix))

        average = Flow(
            'Average',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.binBaselines],
            deps = [demix],
            cluster='ingest',
            costs = self._costs_from_product(Products.Average))

        buf = Flow('Buffer', [self.islandFreqs], deps=[average])

        return (ingest, rfi, demix, average, buf)

    def create_flagging(self,vis):

        return Flow(
            'Flagging',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [vis],
            costs = self._costs_from_product(Products.Flag))

    def create_calibrate(self,vis,model):

        # Source finding
        source_find = Flow(
            'Source Finding',
            [self.eachSelfCal],
            costs = self._costs_from_product(Products.Source_Find),
            deps = [model], cluster='calibrate',
        )

        # DFT Predict
        dft = Flow(
            'DFT',
            [self.eachBeam, self.eachSelfCal, self.xyPolar, self.snapTime, self.visFreq],
            costs = self._costs_from_product(Products.DFT),
            deps = [source_find], cluster='calibrate',
        )

        # Solve
        solve = Flow(
            'Calibration',
            [self.eachBeam, self.eachSelfCal, self.snapTime, self.allFreqs],
            costs = self._costs_from_product(Products.Solve),
            deps = [dft, vis], cluster='calibrate',
        )

        return (source_find, dft, solve)

    def create_apply(self,vis,calibration):

        # "Subtract"?
        app = Flow(
            'Apply Calibration',
            [self.eachBeam, self.eachLoop, self.xyPolar],
            deps = [vis, calibration], cluster='calibrate'
        )

        return app

    def create_predict(self,vis,model):
        """ Predicts visibilities from a model """

        if self.tp.scale_predict_by_facet:
            predictFacets = self.eachFacet
        else:
            predictFacets = self.allFacets

        # Predict
        fft = Flow(
            'FFT',
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
             self.snapTime, self.predFreqs, self.binBaselines],
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

    def create_project(self,facets, regLoop, freqs):
        """ Reprojects facets so they can be seen as part of the same image plane """

        return Flow(
            'Project',
            [self.eachBeam, regLoop, self.xyPolar,
             self.snapTime, freqs, self.eachFacet],
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
             self.snapTime, self.backFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Grid),
            deps = [vis, gcf], cluster = 'backward',
        )

        fft = Flow(
            'IFFT',
            # FIXME: eachLoop and xyPolar might be wrong?
            [self.eachBeam, self.eachLoop, self.eachFacet,
             self.xyPolar, self.snapTime, self.fftBackFreqs],
            costs = self._costs_from_product(Products.FFT),
            deps = [grid], cluster = 'backward',
        )

        return (fft, gcf, grid)

    def create_clean(self, dirty):

        if self.tp.pipeline != Pipelines.DPrepA_Image:
            spectral_fit = dirty
        else:
            spectral_fit = Flow(
                'Spectral Fitting',
                [self.eachBeam, self.eachLoop, self.eachFacet,
                 self.xyPolar, self.snapTime, self.fftBackFreqs],
                costs = self._costs_from_product(Products.FFT),
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
            'Subtract',
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

    def create_continuum(self):

        # LSM might get updated per major loop
        lsm = self.lsm
        if self.tp.Nmajor > 1:
            lsm = self.create_update(lsm)

        # UVWs are supposed to come from TM
        uvws = self.uvw
        # Visibilities from buffer, flagged
        vis = self.create_flagging(self.buf)

        # Calibrate
        calibration = self.create_calibrate(vis, lsm)[-1]
        visc = self.create_apply(vis, calibration)

        # Predict
        degrid = self.create_predict(uvws, lsm)[-1]
        if self.tp.scale_predict_by_facet:
            predVis = self.create_rotate(degrid, False)
        else:
            predVis = degrid

        # Subtract
        subtract = self.create_subtract(visc, predVis)

        # Phase rotate
        rotate = self.create_rotate(subtract, True)

        # Intermediate loops
        fftBack = self.create_backward(rotate, uvws)[0]
        project = self.create_project(fftBack, self.eachLoop, self.fftBackFreqs)
        (_, _, clean) = self.create_clean(project)
        lsm.depend(clean, 0)

        return clean.recursiveDeps()

class PipelineTests(unittest.TestCase):
    """Tests the data flows constructed from the parametric model for
    consistency. This means we both sanity-check the construction as
    well as whether our predictions match the ones on the parametric
    model where there's overlap. This should allow us to ensure that
    we inferred the right architecture.
    """

    def setUp(self):
        self._loadTelParams(Pipelines.ICAL)

    def _loadTelParams(self, pipeline):

        cfg = PipelineConfig(telescope=Telescopes.SKA1_Mid,
                             band=Bands.Mid1,
                             pipeline=pipeline,
                             max_baseline=150000,
                             Nf_max='default',
                             blcoal=True,
                             on_the_fly=True
                             scale_predict_by_facet=True)
        adjusts = {
            'Nfacet': 8,
            'Tsnap': 40
        }

        tp = imp.calc_tel_params(cfg, adjusts=adjusts)
        self.df = Pipeline(tp)

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

    def _assertEqualProduct(self, flow, product):
        self.assertAlmostEqual(
            float(flow.cost('compute')/self.df.tp.Tobs),
            float(self.df.tp.products[product]['Rflop']),
            delta = 20)
        self.assertAlmostEqual(
            float(flow.cost('transfer')/self.df.tp.Tobs),
            float(self.df.tp.products[product]['Rout']),
            delta = 10000)

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
        proj = self.df.create_project(None, self.df.eachLoop, self.df.fftBackFreqs)
        self._assertEqualProduct(proj, Products.Reprojection)

    def test_calibrate(self):
        (source_find, dft, solve) = self.df.create_calibrate(None,None)
        self._assertEqualProduct(source_find, Products.Source_Find)
        self._assertEqualProduct(dft, Products.DFT)
        self._assertEqualProduct(solve, Products.Solve)

    def test_subtract(self):
        subtract = self.df.create_subtract(None, None)
        self._assertEqualProduct(subtract, Products.Subtract_Visibility)

    def test_clean(self):
        (spectral_fit, identify, subtract) = self.df.create_clean(None)
        if self.df.tp.pipeline == Pipelines.DPrepA_Image:
            self._assertEqualProduct(spectral_fit, Products.Spectral_Fitting)
        self._assertEqualProduct(identify, Products.Identify_Component)
        self._assertEqualProduct(subtract, Products.Subtract_Image_Component)

    def test_continuum(self):

        # Create whole pipeline
        contFlows = self.df.create_continuum()

        # Sum up all costs, make sure it matches expected cost sum
        costSum = 0
        for flow in contFlows:
            costSum += float(flow.cost('compute'))
        self.assertAlmostEqual(
            float(costSum/self.df.tp.Tobs),
            float(self.df.tp.Rflop),
            delta = 1)

class PipelineTestsFastImg(PipelineTests):
    def setUp(self):
        self._loadTelParams(Pipelines.Fast_Img)

class PipelineTestsDPrepC(PipelineTests):
    def setUp(self):
        self._loadTelParams(Pipelines.DPrepC)

if __name__ == '__main__':
    unittest.main()
