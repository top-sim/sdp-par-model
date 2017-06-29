
import unittest

from sympy import Lambda, Symbol

from ..parameters.definitions import Pipelines, Products
from ..parameters.container import BLDep
from ..config import PipelineConfig
from .dataflow import *

class Pipeline:
    """Dataflow generator for a SDP pipeline configuration."""

    def __init__(self, tp, nerf_time=1, nerf_freq=1, nerf_loop=1):

        # Set telescope parameters
        self.tp = tp

        self.Nisland = tp.Nf_min
        self.Mdbl = 8
        self.Mcpx = self.Mdbl * 2

        self._create_domains(nerf_time, nerf_freq, nerf_loop)
        self._create_dataflow()

    def _to_domainexpr(self, expr):
        if isinstance(expr, BLDep):
            return DomainExpr(lambda rb:
                expr(b = rb.eval(self.baseline, 'bmax'),
                     bcount = rb.eval(self.baseline, 'size')))
        else:
            return DomainExpr(expr)

    def _create_domains(self, nerf_time=1, nerf_freq=1, nerf_loop=1):
        tp = self.tp

        # Make baseline domain
        self.baseline = Domain('Baseline')
        usedBls = tp.Nbl
        self.allBaselines = self.baseline.regions(usedBls)
        if isinstance(tp.bl_bins, tuple):
            b = Symbol("b")
            self.binBaselines = self.allBaselines.split(tp.Nbl, props={
                'bmax': Lambda(b, Symbol("B_max")(b)),
                'size': 1
            })
        else:
            self.binBaselines = self.allBaselines.split(len(tp.bl_bins), props={
                'bmax': lambda i: tp.bl_bins[i]['b'],
                'size': lambda i: tp.bl_bins[i]['bcount']
            })
        if tp.NAProducts == 'all':
            self.kernelBaselines = self.binBaselines
        else:
            self.kernelBaselines = self.allBaselines.split(1, props={
                'bmax': tp.Bmax,
                'size': tp.NAProducts
            })

        # Make time domain
        tobs = tp.Tobs / nerf_time
        self.time = Domain('Time', 's', priority=7)
        self.obsTime = self.time.regions(tobs)
        self.dumpTime = self.obsTime.split(max(1, tobs / tp.Tint_min))
        self.snapTime = self.obsTime.split(max(1, tobs / tp.Tsnap))
        self.kernelPredTime = self.obsTime.split(
            tobs / self._to_domainexpr(tp.Tkernel_predict))
        self.kernelBackTime = self.obsTime.split(
            tobs / self._to_domainexpr(tp.Tkernel_backward))
        if tp.pipeline == Pipelines.ICAL:
            self.Tsolve = tp.tICAL_G
        elif tp.pipeline == Pipelines.RCAL:
            self.Tsolve = tp.tRCAL_G
        else:
            self.Tsolve = tp.Tsnap
        self.solveTime = self.obsTime.split(max(1, tobs / self.Tsolve))

        # Make frequency domain
        self.frequency = Domain('Frequency', 'ch', priority=8)
        self.allFreqs = self.frequency.regions(tp.Nf_max / nerf_freq)
        self.eachFreq = self.allFreqs.split(tp.Nf_max / nerf_freq)
        self.visFreq = self.allFreqs.split(
            self._to_domainexpr(tp.Nf_vis) / nerf_freq)
        self.outFreqs = self.allFreqs.split(tp.Nf_out / nerf_freq)
        self.islandFreqs = self.allFreqs.split(self.Nisland / nerf_freq)
        self.granFreqs = self.allFreqs.split(tp.Nf_min_gran / nerf_freq)
        self.predFreqs = self.allFreqs.split(
            self._to_domainexpr(tp.Nf_vis_predict) / nerf_freq)
        self.backFreqs = self.allFreqs.split(
            self._to_domainexpr(tp.Nf_vis_backward) / nerf_freq)
        self.gcfPredFreqs = self.allFreqs.split(
            self._to_domainexpr(tp.Nf_gcf_predict) / nerf_freq)
        self.gcfBackFreqs = self.allFreqs.split(
            self._to_domainexpr(tp.Nf_gcf_backward) / nerf_freq)
        self.fftPredFreqs = self.allFreqs.split(tp.Nf_FFT_predict / nerf_freq)
        self.fftBackFreqs = self.allFreqs.split(tp.Nf_FFT_backward / nerf_freq)
        self.projPredFreqs = self.allFreqs.split(tp.Nf_proj_predict / nerf_freq)
        self.projBackFreqs = self.allFreqs.split(tp.Nf_proj_backward / nerf_freq)
        self.cleanFreqs = self.allFreqs.split(max(1, tp.Nf_identify / nerf_freq))

        # Make beam domain
        self.beam = Domain('Beam', priority=10)
        self.allBeams = self.beam.regions(tp.Nbeam)
        self.eachBeam = self.allBeams.split(tp.Nbeam)

        # Make polarisation domain
        self.polar = Domain('Polarisation')
        self.iquvPolars = self.polar.regions(tp.Npp)
        self.iquvPolar = self.iquvPolars.split(tp.Npp)
        self.xyPolars = self.polar.regions(tp.Npp)
        self.xyPolar = self.xyPolars.split(tp.Npp)

        # Make (major) loop domain
        self.loop = Domain('Major Loop', priority=9)
        self.allLoops = self.loop.regions(tp.Nmajortotal / nerf_loop)
        self.eachLoop = self.allLoops.split(Max(1, tp.Nmajortotal / nerf_loop))
        self.allSelfCals = self.loop.regions(Max(1, (tp.Nselfcal + 1) / nerf_loop))
        self.eachSelfCal = self.allSelfCals.split(Max(1, (tp.Nselfcal + 1) / nerf_loop))

        # Make facet domain
        self.facet = Domain('Facet', priority=6)
        self.allFacets = self.facet.regions(tp.Nfacet**2)
        self.eachFacet = self.allFacets.split(tp.Nfacet**2)

        # Make taylor term domain
        self.taylor = Domain('Taylor', priority=5)
        self.allTaylor = self.taylor.regions(tp.Ntt)
        self.eachTaylor = self.allTaylor.split(tp.Ntt)
        self.predTaylor = self.allTaylor.split(tp.Ntt_predict)
        self.backTaylor = self.allTaylor.split(tp.Ntt_backward)

        # We want to completely remove taylor terms for pipelines that
        # don't actually use them.
        if tp.Ntt_predict == 1:
            self.maybePredTaylor = []
        else:
            self.maybePredTaylor = [self.predTaylor]
        if tp.Ntt_backward == 1:
            self.maybeBackTaylor = []
        else:
            self.maybeBackTaylor = [self.backTaylor]

    def _transfer_cost_vis(self, Tdump):
        """Utility transfer cost function for visibility data. Multiplies out
        frequency, baselines, polarisations and time given a dump time"""

        return (self.tp.Mvis * self.frequency('size')
                             * self.baseline('size')
                             * self.polar('size')
                             * self.time('size')
                             / self._to_domainexpr(Tdump))

    def _create_dataflow(self):
        """ Creates common data flow nodes """

        self.tm = Flow('Telescope Management', cluster = 'interface')

        self.lsm = Flow('Local Sky Model', attrs = {'pos':'0 0'},
                        cluster = 'interface')

        self.uvw = Flow(
            'Telescope Data',
            [self.eachBeam, self.snapTime, self.islandFreqs, self.allBaselines],
            deps = [self.tm],
            costs = {
                #'transfer': lambda rbox:
                #  3 * 8 * rbox(self.frequency, 'size')
                #        * rbox(self.baseline, 'size')
                #        * rbox(self.time, 'size')
                #        / self.tp.Tint_used
            })

        self.corr = Flow(
            'Correlator',
            [self.allBeams, self.dumpTime,
             self.eachFreq, self.xyPolar, self.allBaselines],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tint_min)})

        # TODO - buffer contains averaged visibilities...
        self.buf = Flow(
            'Visibility Buffer',
            [self.eachBeam, self.granFreqs, self.allBaselines, self.snapTime, self.xyPolars],
            cluster = 'interface',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tint_min)}
        )

    def _cost_from_product(self, product, cost):

        # Get time field
        product_costs = self.tp.products.get(product, {})
        t = product_costs.get("T", 0)

        # Baseline-dependent task?
        if cost + "_task" in product_costs:
            task_cost = product_costs[cost + "_task"]
            # Pass baseline length, multiply by number of baselines
            return self._to_domainexpr(t * task_cost)
        else:
            # Otherwise simply return the value as-is
            return t * product_costs.get(cost, 0) / product_costs.get("N", 1)

    def _costs_from_product(self, product):
        return {
            'compute': self._cost_from_product(product, 'Rflop'),
            'transfer': self._cost_from_product(product, 'Rout'),
        }

    def create_ingest(self):

        ingest = Flow(
            Products.Receive,
            [self.eachBeam, self.snapTime,
             self.islandFreqs, self.xyPolars, self.allBaselines],
            deps = [self.tm, self.corr],
            cluster='ingest',
            costs = self._costs_from_product(Products.Receive))

        demix = Flow(
            Products.Demix,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [ingest],
            cluster='ingest',
            costs = self._costs_from_product(Products.Demix))

        rfi = Flow(
            Products.Flag,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.allBaselines],
            deps = [demix],
            cluster='ingest',
            costs = self._costs_from_product(Products.Flag))

        average = Flow(
            Products.Average,
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.binBaselines],
            deps = [rfi],
            cluster='ingest',
            costs = self._costs_from_product(Products.Average))

        buf = Flow('Buffer', [self.eachBeam, self.islandFreqs], deps=[average])

        return buf

    def create_flagging(self,vis):

        return Flow(
            Products.Flag,
            [self.eachBeam, self.snapTime, self.granFreqs,
             self.xyPolars, self.binBaselines, self.eachLoop],
            deps = [vis], cluster='calibrate',
            costs = self._costs_from_product(Products.Flag))

    def create_calibrate(self,vis,modelVis):

        # No calibration?
        if Products.Solve in self.tp.products:

            # Collect island-local data for a time slot together and
            # average it before sending it off-island
            slots = Flow(
                "Timeslots",
                [self.eachBeam, self.eachSelfCal, self.solveTime, self.islandFreqs, self.xyPolars,
                 self.binBaselines],
                # maybe twice this, because we need to transfer
                # visibility + model...?
                costs = { 'transfer': self._transfer_cost_vis(self.Tsolve) },
                deps = [vis, modelVis], cluster='calibrate'
            )

            # Solve
            solve = Flow(
                Products.Solve,
                [self.eachBeam, self.eachSelfCal, self.solveTime, self.allFreqs, self.xyPolars,
                 self.allBaselines],
                costs = self._costs_from_product(Products.Solve),
                deps = [slots], cluster='calibrate',
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
                      self.tp.Mjones * self.tp.Na * self.tp.Nf_max * self.tp.Tsnap / self.tp.Tint_min
                }
            )

        # Apply the calibration
        app = Flow(
            Products.Correct,
            [self.snapTime, self.granFreqs, self.eachBeam,
             self.eachLoop, self.xyPolar, self.binBaselines],
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
            [self.eachBeam, self.eachLoop, self.xyPolars,
             self.snapTime, self.granFreqs, self.binBaselines] +
             self.maybePredTaylor,
            deps = [dft, degrid], cluster='predict',
            costs = {
                #'compute': lambda rb: 2 * (1 + self.tp.Nfacet) * self._transfer_cost_vis(self.tp.Tint_used)(rb),
                'transfer': self._transfer_cost_vis(self.tp.Tint_used)
            }
        )

        return add

    def create_predict_direct(self,uvw,sources):
        """ Predict visibilities by DFT """

        return Flow(
            Products.DFT,
            [self.eachBeam, self.eachLoop, self.xyPolars,
             self.snapTime, self.granFreqs, self.binBaselines],
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
            [self.eachBeam, self.eachLoop, predictFacets, self.xyPolar,
             self.kernelPredTime, self.gcfPredFreqs, self.kernelBaselines],
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
                [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
                 self.snapTime, self.granFreqs, self.binBaselines] +
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

        reproject = Flow(
            Products.Reprojection,
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.projBackFreqs, self.eachFacet],
            costs = self._costs_from_product(Products.Reprojection),
            deps = [facets],
            cluster = 'backward'
        )

        # Also sum up reprojected results across snapshots
        summed = Flow(
            "Sum Facets",
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.obsTime, self.projBackFreqs, self.eachFacet],
            costs = {
                'transfer': self.tp.Mpx * self.tp.Npix_linear**2
            },
            deps = [reproject],
            cluster = 'backward'
        )

        return summed

    def create_subtract(self, vis, model_vis):
        """ Subtracts model visibilities from measured visibilities """

        return Flow(
            Products.Subtract_Visibility,
            [self.eachBeam, self.eachLoop, self.xyPolars,
             self.snapTime, self.granFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.Subtract_Visibility),
            deps = [vis, model_vis], cluster='calibrate'
        )

    def create_backward(self, vis, uvw):
        """ Creates dirty image from visibilities """

        # TODO: This product is for backward *and* forward!
        rotate = Flow(
            Products.PhaseRotation,
            [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
             self.snapTime, self.granFreqs, self.binBaselines],
            costs = self._costs_from_product(Products.PhaseRotation),
            deps = [vis], cluster = 'backward'
        )

        gcf = Flow(
            Products.Gridding_Kernel_Update,
            [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
             self.kernelBackTime, self.gcfBackFreqs, self.kernelBaselines],
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
            [self.eachBeam, self.eachLoop, self.cleanFreqs, self.eachFacet],
            costs = self._costs_from_product(Products.Identify_Component),
            deps = [spectral_fit],
            cluster = 'deconvolve'
        )

        subtract =  Flow(
            Products.Subtract_Image_Component,
            [self.eachBeam, self.eachLoop, self.cleanFreqs, self.eachFacet],
            costs = self._costs_from_product(Products.Subtract_Image_Component),
            deps = [identify, spectral_fit],
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

    def perform_imaging_merges(self, root):

        # List of changes to apply
        merges = [
            ([Products.ReprojectionPredict,Products.IFFT],
             [self.eachBeam, self.eachLoop, self.xyPolar, self.snapTime, self.projPredFreqs, self.eachFacet],
             None),
            ([Products.Degridding_Kernel_Update, Products.Degrid],
             [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
              self.snapTime, self.granFreqs],
             None),
            ([Products.PhaseRotationPredict, Products.DFT, "Sum visibilities"],
             [self.eachBeam, self.eachLoop, self.xyPolars,
              self.snapTime, self.granFreqs, self.allBaselines],
             None),
            (["Timeslots"],
             [self.eachBeam, self.eachSelfCal, self.solveTime, self.islandFreqs, self.xyPolars, self.allBaselines],
             None),
            ([Products.Solve],
             [self.eachBeam, self.eachSelfCal, self.solveTime, self.allFreqs, self.xyPolars, self.allBaselines],
             None),
            ([Products.Correct,Products.Subtract_Visibility,Products.Flag],
             [self.eachBeam, self.snapTime, self.granFreqs, self.xyPolars, self.allBaselines, self.eachLoop],
             None),
            ([Products.Gridding_Kernel_Update,
              Products.PhaseRotation,Products.Grid, Products.FFT, Products.Reprojection],
             [self.eachBeam, self.eachLoop, self.xyPolar,self.snapTime, self.projBackFreqs, self.eachFacet],
             None)
        ]
        for flows_to_merge, gran, split_gran in merges:

            # Merge
            merged = mergeFlows(root, root.getDeps(flows_to_merge), gran)

            # Split output
            if not split_gran is None:
                makeSplitFlow(root, merged, split_gran)

    def create_pipeline(self, performMerges=False):
        """
        Create the pipeline for the telescope parameters this class got
        constructed with.

        :param performMerges: Do not return individual functions,
          instead merge and split such that the returned flows have
          a realistic granularity.
        """

        if self.tp.pipeline in Pipelines.imaging:

            # Create imaging pipeline
            root = self.create_imaging()

            # Perform merges, if requested
            if performMerges:
                self.perform_imaging_merges(root)

            return root

        if self.tp.pipeline == Pipelines.Ingest:
            return self.create_ingest()
