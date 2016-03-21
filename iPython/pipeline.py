
from parameter_definitions import *
from implementation import Implementation as imp, PipelineConfig

from dataflow import *
from sympy import Lambda

import unittest

class Pipeline:

    def __init__(self, tp):

        # Set telescope parameters
        self.tp = tp

        self.Nisland = 225
        self.Mdbl = 8
        self.Mcpx = self.Mdbl * 2

        self._create_domains()
        self._create_dataflow()

    def _create_domains(self):
        tp = self.tp

        # Make baseline domain
        self.baseline = Domain('Baseline')
        usedBls = tp.nbaselines * sum(tp.frac_bins)
        self.allBaselines = self.baseline.region(usedBls)
        self.binBaselines = self.allBaselines.split(len(tp.Bmax_bins), props={
            'bmax': lambda i: tp.Bmax_bins[i],
            'size': lambda i: tp.nbaselines * tp.frac_bins[i]
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
        self.outFreqs = self.allFreqs.split(tp.Nf_out)
        self.islandFreqs = self.allFreqs.split(self.Nisland)
        self.predFreqs = self.allFreqs.split(tp.Nf_vis_predict)
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
        self.allLoops = self.loop.region(tp.Nmajor)
        self.eachLoop = self.allLoops.split(tp.Nmajor)

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

        self.tm = Flow('Telescope Management', cluster='input')

        self.lsm = Flow('Local Sky Model', cluster='input', attrs = {'pos':'0 0'},
                        deps = [self.tm])

        self.corr = Flow(
            'Correlator',
            [self.allBeams, self.dumpTime,
             self.eachFreq, self.xyPolars, self.allBaselines],
            cluster='input',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)})

        self.ingest = Flow(
            'Ingest',
            [self.eachBeam, self.dumpTime,
             self.islandFreqs, self.xyPolars, self.allBaselines],
            deps = [self.tm, self.corr],
            cluster='input',
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tdump_ref)})

        self.rfi = Flow(
            'Flagging / Integration',
            [self.eachBeam, self.snapTime, self.islandFreqs,
             self.xyPolars, self.binBaselines], deps = [self.ingest],
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tcoal_skipper)})

        self.uvw = Flow(
            'UVW Rotation',
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

    def create_predict(self,vis,model):
        """ Predicts visibilities from a model """

        # Predict
        fft = Flow(
            'FFT',
            # FIXME: eachLoop and xyPolar might be wrong
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.fftPredFreqs],
            costs = {
                'compute': self.tp.Rfft_predict * self.tp.Tsnap,
                'transfer': self.Mcpx * self.tp.Nfacet_x_Npix * (self.tp.Nfacet_x_Npix / 2 + 1)
            },
            deps = [model], cluster='predict',
        )

        gcf = Flow(
            'Degrid w-kernels',
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.kernelPredTime, self.gcfPredFreqs, self.binBaselines],
            costs = {
                'compute': lambda rbox:
                  rbox(self.baseline,'size') * self.tp.Rccf_predict_task(rbox(self.baseline,'bmax')),
                'transfer': lambda rbox:
                  8 * self.tp.Qgcf**3 * self.tp.Ngw_predict(rbox(self.baseline,'bmax'))**3
            },
            deps = [vis], cluster='predict'
        )

        degrid = Flow(
            "Degrid",
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.predFreqs, self.binBaselines],
            costs = {
                'compute': lambda rbox:
                  self.tp.Rgrid_predict_task(rbox(self.baseline, 'size'),
                                             rbox(self.baseline, 'bmax')),
                'transfer': lambda rbox:
                  self.tp.Mvis * rbox(self.baseline, 'size') * rbox(self.time, 'size')
            },
            deps = [fft, gcf], cluster='predict'
        )

        return (fft, gcf, degrid)

    def create_rotate(self,vis,backward):
        """ Rotates the phase center of visibilities for facetting """

        # Select the right cost function
        Rflop_phrot_task = self.tp.Rflop_phrot_predict_task
        freqs = self.predFreqs
        if backward:
            Rflop_phrot_task = self.tp.Rflop_phrot_backward_task
            freqs = self.backFreqs

        rotate = Flow(
            "Rotate Phase",
            [self.eachBeam, self.eachLoop, self.eachFacet, self.xyPolar,
             self.snapTime, freqs, self.binBaselines],
            costs = {
                'compute': lambda rbox: Rflop_phrot_task(rbox(self.baseline,'size'),
                                                         rbox(self.baseline,'bmax')),
                'transfer': self._transfer_cost_vis(self.tp.Tcoal_skipper)
            },
            deps = [vis],
        )

        return rotate

    def create_project(self,facets, regLoop, freqs):
        """ Reprojects facets so they can be seen as part of the same image plane """

        return Flow(
            'Project',
            [self.eachBeam, regLoop, self.xyPolar,
             self.snapTime, freqs],
            costs = {
                'compute': self.tp.Rrp * self.tp.Tsnap,
                'transfer': self.Mdbl * self.tp.Nfacet_x_Npix**2
            },
            deps = [facets],
            cluster = 'backend'
        )

    def create_subtract(self, vis, model_vis):
        """ Subtracts model visibilities from measured visibilities """

        return Flow(
            "Subtract",
            [self.eachBeam, self.eachLoop, self.xyPolar,
             self.snapTime, self.predFreqs, self.binBaselines],
            deps = [vis, model_vis],
            costs = {'transfer': self._transfer_cost_vis(self.tp.Tcoal_skipper)}
        )

    def create_backward(self, vis, uvw):
        """ Creates dirty image from visibilities """

        regLoop = self.eachLoop
        fftFreqs = self.fftBackFreqs

        tp = self.tp
        gcf = Flow(
            'Grid w-kernels',
            [self.eachBeam, regLoop, self.eachFacet, self.xyPolar,
             self.kernelBackTime, self.gcfBackFreqs, self.binBaselines],
            costs = {
                'compute': lambda rbox:
                  rbox(self.baseline,'size') * tp.Rccf_backward_task(rbox(self.baseline,'bmax')),
                'transfer': lambda rbox:
                  8 * self.tp.Qgcf**3 * self.tp.Ngw_backward(rbox(self.baseline,'bmax'))**3
            },
            deps = [uvw], cluster = 'backward',
        )

        grid = Flow(
            'Grid',
            [self.eachBeam, regLoop, self.eachFacet, self.xyPolar,
             self.snapTime, self.backFreqs, self.binBaselines],
            costs = {
                'compute': lambda rbox: tp.Rgrid_backward_task(rbox(self.baseline,'size'),
                                                               rbox(self.baseline,'bmax')),
                'transfer': lambda rbox:
                  self.Mcpx * self.tp.Npix_linear * (self.tp.Npix_linear / 2 + 1)
            },
            deps = [vis, gcf], cluster = 'backward',
        )

        fft = Flow(
            'IFFT',
            # FIXME: eachLoop and xyPolar might be wrong?
            [self.eachBeam, regLoop, self.eachFacet, self.xyPolar,
             self.snapTime, fftFreqs],
            costs = {
                # Rfft_backward is cost for all facets together
                'compute': tp.Rfft_backward * tp.Tsnap / tp.Nfacet ** 2,
                'transfer': self.Mdbl * self.tp.Npix_linear**2
            },
            deps = [grid], cluster = 'backward',
        )

        return (fft, gcf, grid)

    def create_clean(self, dirty, loops):

        return Flow(
            'Clean',
            [self.eachBeam, loops, self.xyPolars,
             self.obsTime, self.allFreqs],
            deps = [dirty],
            cluster = 'backend'
        )

    def create_update(self, lsm):

        return Flow(
            'Update',
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
        # Visibilities from RFI
        vis = self.rfi

        # Predict
        (fftPred, gcfPred, degrid) = self.create_predict(uvws, lsm)

        # Subtract
        subtract = self.create_subtract(vis, degrid)

        # Phase rotate
        rotate = self.create_rotate(subtract, True)

        # Intermediate loops
        (fftBack, gcfBack, grid) = self.create_backward(rotate, uvws)
        project = self.create_project(fftBack, self.eachLoop, self.fftBackFreqs)
        clean = self.create_clean(project, self.eachLoop)
        lsm.depend(clean)

        return clean.recursiveDeps()

class PipelineTests(unittest.TestCase):
    """Tests the data flows constructed from the parametric model for
    consistency. This means we both sanity-check the construction as
    well as whether our predictions match the ones on the parametric
    model where there's overlap. This should allow us to ensure that
    we inferred the right architecture.
    """

    def setUp(self):
        self._loadTelParams(ImagingModes.Continuum)

    def _loadTelParams(self, mode):

        cfg = PipelineConfig(telescope=Telescopes.SKA1_Mid,
                             band=Bands.Mid1,
                             mode=mode,
                             max_baseline=60000,
                             Nf_max='default',
                             bldta=True,
                             on_the_fly=True)
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

    def test_fft(self):

        tp = self.df.tp
        (fftPred, _, _) = self.df.create_predict(None, None)
        (fftBack, _, _) = self.df.create_backward(None, None)

        # Check individual compute sums
        self.assertAlmostEqual(
            float(fftPred.cost('compute')/tp.Tobs),
            float(tp.Npp * tp.Nbeam * tp.Nmajor * tp.Nf_FFT_predict * tp.Rfft_predict),
            delta = 1)
        self.assertAlmostEqual(
            float(fftBack.cost('compute')/tp.Tobs),
            float(tp.Npp * tp.Nbeam * tp.Nmajor * tp.Nf_FFT_backward * tp.Rfft_backward),
            delta = 1)

        # Check total compute sum against official result
        fftCost = fftPred.cost('compute') + fftBack.cost('compute')
        self.assertAlmostEqual(float(fftCost/self.df.tp.Tobs),
                               float(self.df.tp.Rflop_fft),
                               delta = 1)

    def test_gcf(self):

        tp = self.df.tp
        (_, gcfPred, _) = self.df.create_predict(None, None)
        (_, gcfBack, _) = self.df.create_backward(None, None)

        self.assertAlmostEqual(
            float(gcfPred.cost('compute')/self.df.tp.Tobs),
            float(tp.Rccf_predict),
            delta = 1)
        self.assertAlmostEqual(
            float(gcfBack.cost('compute') / tp.Tobs),
            float(tp.Rccf_backward),
            delta = 1)

    def test_grid(self):

        tp = self.df.tp
        (_, _, gridPred) = self.df.create_predict(None, None)
        (_, _, gridBack) = self.df.create_backward(None, None)

        self.assertAlmostEqual(
            float(gridPred.cost('compute')/self.df.tp.Tobs),
            float(tp.Rgrid_predict),
            delta = 1)
        self.assertAlmostEqual(
            float(gridBack.cost('compute')/tp.Tobs),
            float(tp.Rgrid_backward),
            delta = 1)

    def test_rotate(self):

        rotPred = self.df.create_rotate(None,backward=False)
        rotBack = self.df.create_rotate(None,backward=True)

        self.assertAlmostEqual(
            float(rotBack.cost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop_phrot),
            delta = 1)

    def test_project(self):

        proj = self.df.create_project(None, self.df.eachLoop, self.df.fftBackFreqs)

        self.assertAlmostEqual(
            float(proj.cost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop_proj),
            delta = 1)

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
        self._loadTelParams(ImagingModes.FastImg)

class PipelineTestsSpectral(PipelineTests):
    def setUp(self):
        self._loadTelParams(ImagingModes.Spectral)

if __name__ == '__main__':
    unittest.main()
