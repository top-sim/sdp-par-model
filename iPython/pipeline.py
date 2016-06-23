
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
            self.binBaselines = self.allBaselines.split(tp.nbaselines, props={
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
        self.projPredFreqs = self.allFreqs.split(tp.Nf_proj_predict)
        self.projBackFreqs = self.allFreqs.split(tp.Nf_proj_backward)

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
        self.eachLoop = self.allLoops.split(tp.Nmajortotal)
        self.allSelfCals = self.loop.regions(tp.Nselfcal + 1)
        self.eachSelfCal = self.allSelfCals.split(tp.Nselfcal + 1)

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

        # We want to completely remove taylor terms for pipelines that
        # don't actually use them.
        if tp.Ntaylor_predict == 1:
            self.maybePredTaylor = []
        else:
            self.maybePredTaylor = [self.predTaylor]
        if tp.Ntaylor_backward == 1:
            self.maybeBackTaylor = []
        else:
            self.maybeBackTaylor = [self.backTaylor]

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
                        cluster = 'interface')

        self.uvw = Flow(
            'UVW generation',
            [self.eachBeam, self.snapTime, self.islandFreqs, self.allBaselines],
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
             self.eachFreq, self.xyPolar, self.allBaselines],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)})

        # TODO - buffer contains averaged visibilities...
        self.buf = Flow(
            'Visibility Buffer',
            [self.eachBeam, self.islandFreqs, self.allBaselines, self.snapTime, self.xyPolars],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)}
        )

    def _cost_from_product(self, rbox, product, cost):

        # Get time field
        product_costs = self.tp.products.get(product, {})
        t = product_costs.get("T", 0)

        # Baseline-dependent task?
        if cost + "_task" in product_costs:
            task_cost = product_costs[cost + "_task"]
            # Pass baseline length, multiply by number of baselines
            return unbldep(t * task_cost, rbox(self.baseline,'bmax'), rbox(self.baseline,SIZE_PROP))
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
            Products.Receive,
            [self.eachBeam, self.snapTime,
             self.islandFreqs, self.xyPolar, self.allBaselines],
            deps = [self.tm, self.corr],
            cluster='ingest',
            costs = self._costs_from_product(Products.Receive))

        demix = Flow(
            Products.Demix,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar],
            deps = [ingest],
            cluster='ingest',
            costs = self._costs_from_product(Products.Demix))

        average = Flow(
            Products.Average,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar, self.allBaselines],
            deps = [demix],
            cluster='ingest',
            costs = self._costs_from_product(Products.Average))

        rfi = Flow(
            Products.Flag,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolar, self.allBaselines],
            deps = [average],
            cluster='ingest',
            costs = self._costs_from_product(Products.Flag))

        buf = Flow('Buffer', [self.eachBeam, self.islandFreqs], deps=[rfi])

        return buf

    def create_flagging(self,vis):

        return Flow(
            Products.Flag,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [vis], cluster='calibrate',
            costs = self._costs_from_product(Products.Flag))

    def create_calibrate(self,vis,modelVis):

        # No calibration?
        if Products.Solve in self.tp.products:

            # Solve
            solve = Flow(
                Products.Solve,
                [self.eachBeam, self.eachSelfCal, self.snapTime, self.allFreqs],
                costs = self._costs_from_product(Products.Solve),
                deps = [vis, modelVis], cluster='calibrate',
            )
            calibration = solve

        else:

            solve = None

            # Load calibration
            calibration = Flow(
                'Calibration Buffer',
                [self.eachBeam, self.eachSelfCal, self.snapTime, self.allFreqs],
                cluster='calibrate',
                costs = {
                    'transfer': lambda rb:
                      self.tp.Mjones * self.tp.Na * self.tp.Nf_max * self.tp.Tsnap / self.tp.Tdump_ref
                }
            )

        # Apply the calibration
        app = Flow(
            Products.Correct,
            [self.snapTime, self.islandFreqs, self.eachBeam,
             self.eachLoop, self.xyPolar],
            deps = [vis, calibration], cluster='calibrate',
            costs = self._costs_from_product(Products.Correct)
        )

        return app

    def create_predict(self,uvw,model):
        """ Predicts visibilities from a model """

        # Extract relevant components from LSM
        extract = Flow(
            Products.Extract_LSM,
            [self.eachBeam, self.eachLoop],
            costs = self._costs_from_product(Products.Extract_LSM),
            deps = [model], cluster='predict',
        )

        # Assume we do both FFT + DFT for every predict
        degrid = self.create_predict_grid(uvw,extract)
        dft = self.create_predict_direct(uvw,extract)

        # Sum up both
        add = Flow(
            "Sum visibilities",
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.islandFreqs, self.allBaselines] +
             self.maybePredTaylor,
            deps = [dft, degrid], cluster='predict',
            costs = {
                'transfer': self._transfer_cost_vis(self.tp.Tdump_scaled)
            }
        )

        return add

    def create_predict_direct(self,uvw,sources):
        """ Predict visibilities by DFT """

        return Flow(
            Products.DFT,
            [self.eachBeam, self.eachLoop, self.xyPolars,
             self.snapTime, self.visFreq, self.allBaselines],
            costs = self._costs_from_product(Products.DFT),
            deps = [sources], cluster='predict',
        )

    def create_predict_grid(self,uvw,sources):
        """ Predict visibilities by FFT, then degrid. """

        # Do faceting?
        if self.tp.scale_predict_by_facet:
            predictFacets = self.eachFacet
        else:
            predictFacets = self.allFacets

        # Reproject to generate model image
        if Products.ReprojectionPredict in self.tp.products:
            modelImage = Flow(
                Products.ReprojectionPredict,
                [self.eachBeam, self.eachLoop, self.xyPolar,
                 self.snapTime, self.projPredFreqs, self.eachFacet],
                costs = self._costs_from_product(Products.ReprojectionPredict),
                deps = [sources],
                cluster = 'predict'
            )
        else:
            modelImage = sources

        # FFT to make a uv-grid
        fft = Flow(
            Products.IFFT,
            # FIXME: eachLoop and xyPolar might be wrong
            [self.eachBeam, self.eachLoop, predictFacets, self.xyPolar,
             self.snapTime, self.fftPredFreqs],
            costs = self._costs_from_product(Products.IFFT),
            deps = [modelImage], cluster='predict',
        )

        # Degrid
        gcf = Flow(
            Products.Degridding_Kernel_Update,
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.kernelPredTime, self.gcfPredFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Degridding_Kernel_Update),
            deps = [uvw], cluster='predict'
        )
        degrid = Flow(
            Products.Degrid,
            [self.eachBeam, self.eachLoop, predictFacets, self.xyPolar,
             self.snapTime, self.predFreqs, self.binBaselines] +
             self.maybePredTaylor,
            costs = self._costs_from_product(Products.Degrid),
            deps = [fft, gcf], cluster='predict'
        )

        # Rotate if we are doing faceting
        if self.tp.scale_predict_by_facet:
            return Flow(
                Products.PhaseRotationPredict,
                [self.eachBeam, self.eachLoop, self.allFacets, self.xyPolar,
                 self.snapTime, self.islandFreqs, self.binBaselines] +
                 self.maybePredTaylor,
                costs = self._costs_from_product(Products.PhaseRotationPredict),
                deps = [degrid], cluster = 'predict'
            )
        else:
            return degrid

    def create_project(self, facets):
        """ Reprojects facets so they can be seen as part of the same image plane """

        # No reprojection?
        if not Products.Reprojection in self.tp.products:
            return facets

        return Flow(
            Products.Reprojection,
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.projBackFreqs, self.eachFacet],
            costs = self._costs_from_product(Products.Reprojection),
            deps = [facets],
            cluster = 'backward'
        )

    def create_subtract(self, vis, model_vis):
        """ Subtracts model visibilities from measured visibilities """

        return Flow(
            Products.Subtract_Visibility,
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.islandFreqs, self.allBaselines],
            costs = self._costs_from_product(Products.Subtract_Visibility),
            deps = [vis, model_vis], cluster='calibrate'
        )

    def create_backward(self, vis, uvw):
        """ Creates dirty image from visibilities """

        # TODO: This product is for backward *and* forward!
        rotate = Flow(
            Products.PhaseRotation,
            [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
             self.snapTime, self.islandFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.PhaseRotation),
            deps = [vis], cluster = 'backward'
        )

        gcf = Flow(
            Products.Gridding_Kernel_Update,
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.kernelBackTime, self.gcfBackFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Gridding_Kernel_Update),
            deps = [uvw], cluster = 'backward',
        )

        grid = Flow(
            Products.Grid,
            [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
             self.snapTime, self.backFreqs, self.binBaselines] +
             self.maybeBackTaylor,
            costs = self._costs_from_product(Products.Grid),
            deps = [rotate, gcf], cluster = 'backward',
        )

        fft = Flow(
            Products.FFT,
            # FIXME: eachLoop and xyPolar might be wrong?
            [self.eachBeam, self.eachLoop, self.eachFacet,
             self.xyPolar, self.snapTime, self.fftBackFreqs],
            costs = self._costs_from_product(Products.FFT),
            deps = [grid], cluster = 'backward',
        )

        return fft

    def create_clean(self, dirty):

        # Skip for fast imaging
        if not Products.Subtract_Image_Component in self.tp.products:
            return dirty

        if self.tp.pipeline != Pipelines.DPrepA_Image:
            spectral_fit = dirty
        else:
            spectral_fit = Flow(
                Products.Image_Spectral_Fitting,
                [self.eachBeam, self.eachLoop,
                 self.xyPolar, self.obsTime, self.eachTaylor],
                costs = self._costs_from_product(Products.Image_Spectral_Fitting),
                deps = [dirty], cluster = 'deconvolve',
            )

        identify =  Flow(
            Products.Identify_Component,
            [self.eachBeam, self.eachLoop],
            costs = self._costs_from_product(Products.Identify_Component),
            deps = [spectral_fit],
            cluster = 'deconvolve'
        )

        subtract =  Flow(
            Products.Subtract_Image_Component,
            [self.eachBeam, self.eachLoop],
            costs = self._costs_from_product(Products.Subtract_Image_Component),
            deps = [identify],
            cluster = 'deconvolve'
        )
        identify.depend(subtract)

        # Is our result found sources?
        if Products.Source_Find in self.tp.products:
            return Flow(
                Products.Source_Find,
                [self.eachBeam, self.eachLoop],
                costs = self._costs_from_product(Products.Source_Find),
                deps = [identify], cluster='deconvolve',
            )
        else:
            # Residual?
            return subtract

    def create_update(self, lsm):

        return Flow(
            'Update LSM',
            [self.eachBeam, self.eachLoop],
            deps = [lsm],
        )

    def create_imaging(self):

        # LSM might get updated per major loop
        lsm = self.lsm
        has_clean = Products.Subtract_Image_Component in self.tp.products
        if has_clean:
            lsm = self.create_update(lsm)

        # Predict
        predict = self.create_predict(self.uvw, lsm)

        # Calibrate visibilities from buffer
        calibrated = self.create_calibrate(self.buf, predict)

        # Subtract & flag
        subtract = self.create_subtract(calibrated, predict)
        flagged = self.create_flagging(subtract)

        # Image
        dirty = self.create_backward(flagged, self.uvw)

        # Reproject
        project = self.create_project(dirty)

        # Clean
        if has_clean:
            clean = self.create_clean(project)
            lsm.depend(clean, 0)
            out = clean
        else:
            out = project

        return out

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

        # Look up flow in dependencies
        if not flow is None:
            flow = flow.getDep(product)

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

    def _assertPipelineComplete(self, flow):
        """Checks whether the given flow has the same compute and transfer
        rate as the product in our telescope parameters. If the
        product is undefined, the flow should be None.
        """

        # Sort costs, match descending
        def flowCost(f): return -f.cost('compute')
        flows = flow.recursiveDeps()
        flowsSorted = sorted(flows, key=flowCost)
        def productsCost(n_c): return -n_c[1]['Rflop']
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
                        map(lambda n_c: n_c[0], productsSorted))
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

    def test_flag(self):
        if self.df.tp.pipeline == Pipelines.Ingest: return
        rfi = self.df.create_flagging(None)
        self._assertEqualProduct(rfi, Products.Flag)

    def test_predict(self):
        predict = self.df.create_predict(None, None)
        self._assertEqualProduct(predict, Products.ReprojectionPredict)
        self._assertEqualProduct(predict, Products.DFT)
        self._assertEqualProduct(predict, Products.IFFT)
        self._assertEqualProduct(predict, Products.Degrid)
        self._assertEqualProduct(predict, Products.Degridding_Kernel_Update)
        self._assertEqualProduct(predict, Products.PhaseRotationPredict)

    def test_backward(self):
        backward = self.df.create_backward(None, None)
        self._assertEqualProduct(backward, Products.FFT)
        self._assertEqualProduct(backward, Products.Grid)
        self._assertEqualProduct(backward, Products.Gridding_Kernel_Update)
        self._assertEqualProduct(backward, Products.PhaseRotation)

    def test_project(self):
        if Products.Reprojection in self.df.tp.products:
            proj = self.df.create_project(None)
            self._assertEqualProduct(proj, Products.Reprojection)

    def test_calibrate(self):
        calibrate = self.df.create_calibrate(None,None)
        self._assertEqualProduct(calibrate, Products.Solve)
        self._assertEqualProduct(calibrate, Products.Correct)

    def test_subtract(self):
        subtract = self.df.create_subtract(None, None)
        self._assertEqualProduct(subtract, Products.Subtract_Visibility)

    def test_clean(self):
        clean = self.df.create_clean(None)
        self._assertEqualProduct(clean, Products.Image_Spectral_Fitting)
        self._assertEqualProduct(clean, Products.Identify_Component)
        self._assertEqualProduct(clean, Products.Subtract_Image_Component)
        self._assertEqualProduct(clean, Products.Source_Find)

    def test_imaging(self):

        # Check whole pipeline
        self.assertIn(self.df.tp.pipeline, Pipelines.imaging)
        self._assertPipelineComplete(self.df.create_imaging())

    def test_dot(self):

        dot = flowsToDot(self.df.create_imaging(), self.df.tp.Tobs)

        # Do some rough sanity checks. Probably not exactly robust,
        # but there's little we know about the contents...
        self.assertGreater(len(dot.source), 4000)
        self.assertGreater(dot.source.count('['), 40)
        self.assertGreater(dot.source.count(']'), 40)

    def test_baseline_split(self):

        # Check that the baseline split actually is done for the
        # appropriate flows.
        imaging = self.df.create_imaging()
        for product in [Products.Grid, Products.Degrid]:
            dep = imaging.getDep(product)
            def baseline_count_prop(rb):
                return rb(self.df.baseline, 'count')
            self.assertEqual(dep.the(baseline_count_prop),
                             self.df.binBaselines.the(baseline_count_prop))

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

        ingest = self.df.create_ingest()
        self._assertEqualProduct(ingest, Products.Receive)
        self._assertEqualProduct(ingest, Products.Flag)
        self._assertEqualProduct(ingest, Products.Demix)
        self._assertEqualProduct(ingest, Products.Average)

        # Check ingest pipeline
        self.assertEqual(self.df.tp.pipeline, Pipelines.Ingest)
        self._assertPipelineComplete(self.df.create_ingest())

if __name__ == '__main__':
    unittest.main()
