
import unittest

from sdp_par_model.config import PipelineConfig
from sdp_par_model.dataflow.pipeline import *
from sdp_par_model.parameters.definitions import Telescopes, Bands

class PipelineTestsBase(unittest.TestCase):
    """ Common helpers for pipeline test cases """

    def _loadTelParams(self, pipeline):

        # A hard-coded example configuration. Testing all of them
        # would take way too long.
        cfg = PipelineConfig(telescope=Telescopes.SKA1_Mid,
                             band=Bands.Mid1,
                             pipeline=pipeline)
        adjusts = {
            'Nfacet': 8,
            'Tsnap': 40
        }

        tp = cfg.calc_tel_params(adjusts=adjusts)
        return Pipeline(tp)

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
            costSum += fcost
            self.assertAlmostEqual(
                fcost, pcost, delta=pcost/1e10,
                msg="Flow %s cost does not match product %s (%f != %f, factor %f)!\n\n"
                    "This often means that some products don't "
                    "have a matching flow (or vice-versa)\n"
                    "Flow list: %s\nProduct list: %s" % (
                        flow.name, pname, fcost, pcost, pcost/max(fcost, 0.001),
                        list(map(lambda f: f.name, flowsSorted)),
                        list(map(lambda n_c: n_c[0], productsSorted)))
            )

class PipelineTestsImaging(PipelineTestsBase):
    """Tests the data flows constructed from the parametric model for
    consistency. This means we both sanity-check the construction as
    well as whether our predictions match the ones on the parametric
    model where there's overlap. This should allow us to ensure that
    we inferred the right architecture.
    """

    def setUp(self):
        self.df = self._loadTelParams(Pipelines.ICAL)

    def test_baseline_domain(self):

        # Check that size of baseline region sets sums up to all
        # baselines. This is a fairly important requirement for the
        # math to work out...
        blSize = self.df.baseline('size')
        self.assertEqual(self.df.allBaselines.sum(blSize),
                         self.df.allBaselines.size)
        self.assertAlmostEqual(self.df.binBaselines.sum(blSize),
                               self.df.allBaselines.size)

    def test_time_domain(self):

        timeSize = self.df.time('size')
        tp = self.df.tp

        # Dump time and snap time
        self.assertAlmostEqual(float(self.df.dumpTime.max(timeSize)),
                               tp.Tint_min)
        self.assertAlmostEqual(self.df.snapTime.max(timeSize),
                               tp.Tsnap)

        # Kernel times can be BL-dependent
        rboxes = RegionBoxes([self.df.binBaselines, self.df.kernelPredTime])
        self.assertAlmostEqual(rboxes.max(timeSize),
                               tp.Tkernel_predict(tp.Bmax_bins[0]))
        self.assertAlmostEqual(-rboxes.max(-timeSize),
                               tp.Tkernel_predict(tp.Bmax_bins[-1]))

        rboxes = RegionBoxes([self.df.binBaselines, self.df.kernelBackTime])
        self.assertAlmostEqual(rboxes.max(timeSize),
                               tp.Tkernel_backward(tp.Bmax_bins[0]))
        self.assertAlmostEqual(-rboxes.max(-timeSize),
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

        # Sum should match no matter whether we merge or not
        self.assertAlmostEqual(
            float(self.df.create_pipeline(performMerges=False)
                  .recursiveCost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop),
            delta=float(self.df.tp.Rflop/1e10))
        self.assertAlmostEqual(
            float(self.df.create_pipeline(performMerges=False)
                  .recursiveCost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop),
            delta=float(self.df.tp.Rflop/1e10))

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
            baseline_count_prop = self.df.baseline('count')
            self.assertEqual(dep.the(baseline_count_prop),
                             self.df.binBaselines.the(baseline_count_prop))

    def test_nerf(self):

        # Nerfing domains should result in a straightforward cost
        # reduction relative to the vanilla predictions of the
        # parametric model.
        def _test(nerf, name):
            pip_fnerf = Pipeline(self.df.tp, **{name:nerf}).create_pipeline()
            self.assertAlmostEqual(
                float(pip_fnerf.recursiveCost('compute')/self.df.tp.Tobs),
                float(self.df.tp.Rflop/nerf),
                delta=float(self.df.tp.Rflop/nerf/10))

        # However note that this can be off substantially both due to
        # "constant" costs (e.g. cleaning) as well as rounding
        # issues. Therefore we only check that it stays within 25%,
        # and explicitly disregard a few below. This might need review
        # in future.
        _test(self.df.tp.Nmajortotal, 'nerf_loop')
        if not self.df.tp.pipeline in [Pipelines.ICAL, Pipelines.RCAL]:
            _test(self.df.tp.Nf_min, 'nerf_freq')
        if not self.df.tp.pipeline in [Pipelines.ICAL, Pipelines.DPrepC, Pipelines.DPrepA, Pipelines.DPrepA_Image]:
            _test(self.df.tp.Tobs / self.df.tp.Tsnap, 'nerf_time')

class PipelineTestsRCAL(PipelineTestsImaging):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.RCAL)

class PipelineTestsFastImg(PipelineTestsImaging):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.Fast_Img)

class PipelineTestsDPrepA(PipelineTestsImaging):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.DPrepA)

class PipelineTestsDPrepA_Image(PipelineTestsImaging):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.DPrepA_Image)

class PipelineTestsDPrepC(PipelineTestsImaging):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.DPrepC)

class PipelineTestsIngest(PipelineTestsBase):
    def setUp(self):
        self.df = self._loadTelParams(Pipelines.Ingest)

    def test_ingest(self):

        ingest = self.df.create_ingest()
        self._assertEqualProduct(ingest, Products.Receive)
        self._assertEqualProduct(ingest, Products.Flag)
        self._assertEqualProduct(ingest, Products.Demix)
        self._assertEqualProduct(ingest, Products.Average)

        # Check ingest pipeline
        self.assertEqual(self.df.tp.pipeline, Pipelines.Ingest)
        self._assertPipelineComplete(self.df.create_ingest())

        self.assertAlmostEqual(
            float(self.df.create_pipeline(performMerges=False)
                  .recursiveCost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop),
            delta=float(self.df.tp.Rflop/1e10))
        self.assertAlmostEqual(
            float(self.df.create_pipeline(performMerges=True)
                  .recursiveCost('compute')/self.df.tp.Tobs),
            float(self.df.tp.Rflop),
            delta=float(self.df.tp.Rflop/1e10))

if __name__ == '__main__':
    unittest.main()
