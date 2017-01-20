
import unittest

from sdp_par_model.dataflow.dataflow import *

class DataFlowTests(unittest.TestCase):

    def setUp(self):

        # Set up a domain with four different splits - all, each,
        # split into 4 regions and split into 3 regions. All sizes
        # given as floating point numbers, as we should be able to
        # somewhat consistently work with that.
        self.size1 = 12.
        self.dom1 = Domain("Domain 1")
        props = {'dummy': lambda rb: 0 }
        self.reg1 = self.dom1.regions(self.size1, props=props)
        self.split1 = self.reg1.split(self.size1, props=props)
        self.size1a = 4.
        self.split1a = self.reg1.split(self.size1a, props=props)
        self.size1b = 3.
        self.split1b = self.reg1.split(self.size1b, props=props)

        # Assumed by tests
        assert self.size1a > self.size1b and self.size1a < 2*self.size1b

        # Second domain, with splits "all" and "each".
        self.size2 = 7.
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.regions(self.size2, props=props)
        self.split2 = self.reg2.split(self.size2, props=props)

        # Smaller region for the second domain, with splits "all" and "each".
        self.size3 = 5.
        self.dom3 = self.dom2
        self.reg3 = self.dom3.regions(self.size3, props=props)
        self.split3 = self.reg3.split(self.size3, props=props)

    def test_size(self):

        # Expected region properties
        self.assertEqual(self.reg1.size, self.size1)
        self.assertEqual(self.split1.size, self.size1)
        self.assertEqual(self.split1a.size, self.size1)
        self.assertEqual(self.split1b.size, self.size1)
        countProp = self.dom1(COUNT_PROP)
        self.assertEqual(self.reg1.the(countProp), 1)
        self.assertEqual(self.split1.the(countProp), self.size1)
        self.assertEqual(self.split1a.the(countProp), self.size1a)
        self.assertEqual(self.split1b.the(countProp), self.size1b)

    def test_rboxes_1(self):
        self._test_rboxes_1(self.reg1, 1, self.size1)
        self._test_rboxes_1(self.split1, self.size1, 1)
        self._test_rboxes_1(self.split1a, self.size1a, self.size1 / self.size1a)
        self._test_rboxes_1(self.split1b, self.size1b, self.size1 / self.size1b)

    def _test_rboxes_1(self, regs, count, boxSize):
        rboxes = RegionBoxes([regs])
        self.assertEqual(rboxes.count(), count)

        # Summing up a value of one per box should yield the box count
        self.assertEqual(rboxes.sum(lambda rb: 1), count)
        countProp = self.dom1(COUNT_PROP)
        self.assertEqual(rboxes.the(countProp), count)

        # Summing up the box sizes should give us the region size
        sizeProp = self.dom1(SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1)

        # Should all still work when we force the boxes to be
        # enumerated by depending on the dummy property
        dummySizeProp = self.dom1('dummy') + self.dom1(SIZE_PROP)
        self.assertFalse(self.dom1 in rboxes.enumDomains)
        self.assertEqual(rboxes.the(dummySizeProp), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp), self.size1)
        self.assertTrue(self.dom1 in rboxes.enumDomains)
        self.assertEqual(rboxes.the(countProp), count)

        # Summing up indices should work as expected
        indexProp = self.dom1(INDEX_PROP)
        self.assertEqual(rboxes.max(indexProp), count-1)
        self.assertEqual(rboxes.sum(indexProp), count * (count-1) // 2)
        offsetProp = self.dom1(OFFSET_PROP)
        self.assertEqual(rboxes.max(offsetProp), (count-1) * boxSize)
        self.assertEqual(rboxes.sum(offsetProp), (count * (count-1) // 2) * boxSize)

    def test_rboxes_2(self):

        # Same tests as rboxes_1, but with two nested domains. We
        # especially check that domain order does not make a difference.
        self._test_rboxes_2([self.reg1, self.reg2], 1, self.size1 * self.size2)
        self._test_rboxes_2([self.reg1, self.split2], self.size2, self.size1)
        self._test_rboxes_2([self.split1, self.reg2], self.size1, self.size2)
        self._test_rboxes_2([self.split1, self.split2], self.size1 * self.size2, 1)
        self._test_rboxes_2([self.split1a, self.reg2], self.size1a, self.size2 * self.size1 / self.size1a)
        self._test_rboxes_2([self.split1a, self.split2], self.size1a * self.size2, self.size1 / self.size1a)
        self._test_rboxes_2([self.split1b, self.reg2], self.size1b, self.size2 * self.size1 / self.size1b)
        self._test_rboxes_2([self.split1b, self.split2], self.size1b * self.size2, self.size1 / self.size1b)
        self._test_rboxes_2([self.reg2, self.reg1], 1, self.size1 * self.size2)
        self._test_rboxes_2([self.split2, self.reg1], self.size2, self.size1)
        self._test_rboxes_2([self.reg2, self.split1], self.size1, self.size2)
        self._test_rboxes_2([self.split2, self.split1], self.size1 * self.size2, 1)
        self._test_rboxes_2([self.reg2, self.split1a], self.size1a, self.size2 * self.size1 / self.size1a)
        self._test_rboxes_2([self.split2, self.split1a], self.size1a * self.size2, self.size1 / self.size1a)
        self._test_rboxes_2([self.reg2, self.split1b], self.size1b, self.size2 * self.size1 / self.size1b)
        self._test_rboxes_2([self.split2, self.split1b], self.size1b * self.size2, self.size1 / self.size1b)

    def _test_rboxes_2(self, regss, count, boxSize):
        rboxes = RegionBoxes(regss)
        self.assertEqual(rboxes.count(), count)
        self.assertEqual(rboxes.sum(lambda rb: 1), count)
        sizeProp = self.dom1(SIZE_PROP) * self.dom2(SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1*self.size2)
        dummySizeProp1 = self.dom1('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp1), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp1), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        dummySizeProp2 = self.dom2('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp2), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp2), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        dummySizeProp12 = self.dom1('dummy') + self.dom2('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp12), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp12), self.size1*self.size2)
        self.assertTrue(self.dom1 in rboxes.enumDomains)
        self.assertTrue(self.dom2 in rboxes.enumDomains)

    def test_rboxes_zip(self):

        # Edges between regions of the same domain grow linearly
        self._test_rboxes_zip([self.reg1],   [self.reg1],    self.dom1, self.dom1, 1)
        self._test_rboxes_zip([self.reg1],   [self.split1],  self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.reg1],   [self.split1a], self.dom1, self.dom1, self.size1a)
        self._test_rboxes_zip([self.reg1],   [self.split1b], self.dom1, self.dom1, self.size1b)
        self._test_rboxes_zip([self.split1], [self.reg1],    self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1],  self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1a], self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1b], self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1a], [self.split1b], self.dom1, self.dom1, 2 * self.size1b)
        self._test_rboxes_zip([self.split1b], [self.split1a], self.dom1, self.dom1, 2 * self.size1b)

        # Edges between regions of different domains grow quadradically
        self._test_rboxes_zip([self.reg1],   [self.reg2],   self.dom1, self.dom2, 1)
        self._test_rboxes_zip([self.reg1],   [self.split2], self.dom1, self.dom2, self.size2)
        self._test_rboxes_zip([self.split1], [self.reg2],   self.dom1, self.dom2, self.size1)
        self._test_rboxes_zip([self.split1], [self.split2], self.dom1, self.dom2, self.size1 * self.size2)
        self._test_rboxes_zip([self.split1a],[self.split2], self.dom1, self.dom2, self.size1a * self.size2)
        self._test_rboxes_zip([self.split2], [self.split1a],self.dom2, self.dom1, self.size1a * self.size2)
        self._test_rboxes_zip([self.reg2],   [self.split1a],self.dom2, self.dom1, self.size1a)
        self._test_rboxes_zip([self.split1a],[self.reg2],   self.dom1, self.dom2, self.size1a)
        self._test_rboxes_zip([self.reg2],   [self.split1b],self.dom2, self.dom1, self.size1b)
        self._test_rboxes_zip([self.split1b],[self.reg2],   self.dom1, self.dom2, self.size1b)

        # Multidimensional boxes make edge count go up where you would
        # expect. The number of possible cases explodes here, we just
        # test a few representative ones.
        self._test_rboxes_zip([self.split2, self.reg1],   [self.reg2],   self.dom1, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.reg2],   self.dom2, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.split2], self.dom1, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.split2], self.dom2, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.reg2],   self.dom1, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.reg2],   self.dom2, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.split2], self.dom1, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.split2], self.dom2, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.split2], self.dom1, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.split2], self.dom2, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1a],self.dom1, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1b],self.dom2, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1a],self.dom1, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1b],self.dom2, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split1a, self.reg2],  [self.split1b],self.dom2, self.dom1,
                              2 * self.size1b)
        self._test_rboxes_zip([self.split1b, self.reg2],  [self.split1a],self.dom1, self.dom1,
                              2 * self.size1b)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom1, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom2, self.dom2,
                              self.size1a * self.size2)

    def test_rboxes_zip_different_root(self):

        # Domain 3 is the same as domain 2, however has a different
        # root region size. This means that a few regions split from
        # the larger root region might end up without edges.
        self._test_rboxes_zip([self.reg2],   [self.reg3],   self.dom2, self.dom2, 1)
        self._test_rboxes_zip([self.reg2],   [self.split3], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split2], [self.reg3],   self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split2], [self.split3], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.reg3],   [self.reg2],   self.dom2, self.dom2, 1)
        self._test_rboxes_zip([self.split3], [self.reg2],   self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.reg3],   [self.split2], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split3], [self.split2], self.dom2, self.dom2, self.size3)

    def _test_rboxes_zip(self, regsA, regsB, domA, domB, edges):
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)

        # Summing one per edge should give us the number of
        # edges. Enumerating shouldn't make a difference, as before.
        def oneProp(rbA, rbB): return 1
        self.assertEqual(rboxesA.zipSum(rboxesB, oneProp), edges)
        def onePropA(rbA, rbB): return 1 + rbA.eval(domA, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropA), edges)
        self.assertTrue(domA in rboxesA.enumDomains)
        self.assertFalse(domA in rboxesB.enumDomains)
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)
        def onePropB(rbA, rbB): return 1 + rbB.eval(domB, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropB), edges)
        self.assertFalse(domB in rboxesA.enumDomains)
        self.assertTrue(domB in rboxesB.enumDomains)
        def onePropAB(rbA, rbB): return 1 + rbA.eval(domA, 'dummy') + rbB.eval(domB, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropAB), edges)

    def test_rboxes_cross_zip_simple(self):

        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # No edges cross if the cross region is coarser than the
        # region with the finest split.
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [split1],  [reg1], 0)
        self._test_rboxes_cross_zip([split1], [reg1],    [reg1], 0)
        self._test_rboxes_cross_zip([split1], [split1],  [reg1], 0)

        # All of them cross if we go in or out of the cross domain
        self._test_rboxes_cross_zip([reg2],   [reg1],    [reg1], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [reg2], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [split1], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [split2], 1)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [split1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [split1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split2], 0)
        self._test_rboxes_cross_zip([reg2],   [split1],  [reg1], self.size1)
        self._test_rboxes_cross_zip([reg2],   [split1],  [reg2], self.size1)
        self._test_rboxes_cross_zip([split2], [split1],  [reg1], self.size1 * self.size2)
        self._test_rboxes_cross_zip([split2], [split1],  [reg2], self.size1 * self.size2)

        # However, all edges but one cross if we make it finer
        self._test_rboxes_cross_zip([split1], [reg1],    [split1], self.size1 - 1)
        self._test_rboxes_cross_zip([reg1],   [split1],  [split1], self.size1 - 1)

        # Yet there is no crossing unless we have diagonal edges
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split1], 0)

        # For coarser splits, a few more edges end up non-crossing
        self._test_rboxes_cross_zip([split1], [reg1],    [split1a], self.size1 - self.size1 / self.size1a)
        self._test_rboxes_cross_zip([split1], [reg1],    [split1b], self.size1 - self.size1 / self.size1b)

        # Furthermore, a N-to-1 split will produce (N-1) crossing
        # edges - basically all but the horizontal edges.
        self._test_rboxes_cross_zip([split1], [split1a], [split1], self.size1a * (self.size1 / self.size1a - 1))
        self._test_rboxes_cross_zip([split1], [split1b], [split1], self.size1b * (self.size1 / self.size1b - 1))

        # Things get more complicated if we use a disagreeing split
        # number for detecting crossings. Not 100% sure whether the
        # formulas match reality, but I am pretty sure that the
        # (numerical) results of 3 and 6 respectively are correct.
        self._test_rboxes_cross_zip([split1], [split1a], [split1b],
                                    (self.size1 / self.size1a) * (self.size1 / self.size1a - 1) / 2)
        self._test_rboxes_cross_zip([split1], [split1b], [split1a],
                                    (self.size1 / self.size1b) * (self.size1 / self.size1b - 1) / 2)

        # Adding extra domains into the mix multiplies things up as should be expected
        self._test_rboxes_cross_zip([split1, reg2], [reg1], [split1a],
                                    self.size1 - self.size1 / self.size1a)
        self._test_rboxes_cross_zip([split1, reg2], [reg1], [split1b],
                                    self.size1 - self.size1 / self.size1b)
        self._test_rboxes_cross_zip([split1, split2], [reg1], [split1a],
                                    self.size2 * (self.size1 - self.size1 / self.size1a))
        self._test_rboxes_cross_zip([split1, split2], [reg1], [split1b],
                                    self.size2 * (self.size1 - self.size1 / self.size1b))

    def test_rboxes_cross_zip_multiple1(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # Simple cases for splitting by multiple domains
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1],       [reg1, reg2], 1)
        self._test_rboxes_cross_zip([split1, reg2], [reg1],       [reg1, reg2], self.size1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1],       [reg1, reg2], self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [reg1, reg2], self.size1 * self.size2)

    def test_rboxes_cross_zip_multiple2(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # If a split domain doesn't exist on one side, all existing edges cross.
        self._test_rboxes_cross_zip([reg1,split2],  [reg1],       [reg1, split2], self.size2)
        self._test_rboxes_cross_zip([split1,reg2],  [reg1],       [reg1, split2], self.size1)
        self._test_rboxes_cross_zip([split1,reg2],  [reg1],       [split1, reg2], self.size1)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [reg1, split2], self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [split1,split2],self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [split1, reg2], self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[split1],     [split1, reg2], self.size1*self.size2)

    def test_rboxes_cross_zip_multiple3(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # If they both exist, things multiply out
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [split1, reg2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [split1, reg2], self.size1 - 1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [split1, reg2], 0)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [split1, reg2], (self.size1 - 1) * self.size2)
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [reg1, split2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [reg1, split2], 0)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [reg1, split2], self.size2 - 1)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [reg1, split2], self.size1 * (self.size2 - 1))
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [split1,split2],0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [split1,split2],self.size1 - 1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [split1,split2],self.size2 - 1)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [split1,split2],self.size1 * self.size2 - 1)

    def _test_rboxes_cross_zip(self, regsA, regsB, regsC, edges):
        import itertools
        def enums(regs):
            doms = map(lambda rs: rs.domain, regs)
            return itertools.product(doms, [False, True])
        for (domA, enumA), (domB, enumB), (domC, enumC) in \
                itertools.product(enums(regsA), enums(regsB), enums(regsC)):

            # Construct region boxes
            rboxesA = RegionBoxes(regsA)
            rboxesB = RegionBoxes(regsB)
            rboxesC = RegionBoxes(regsC)

            # Make sure the right domains end up enumerated
            if enumA: rboxesA.sum(1 + domA('dummy'))
            if enumB: rboxesB.sum(1 + domB('dummy'))
            if enumC: rboxesC.sum(1 + domC('dummy'))

            # Check that bounds is consistent, as
            # zipCrossSum will depend on it heavily
            def toOffSize(reg): return (reg[OFFSET_PROP], reg[SIZE_PROP])
            def toOffSizes(regs): return list(map(toOffSize, regs))
            self.assertEqual(list(map(toOffSizes, rboxesA.regions(domA))), list(rboxesA.bounds(domA)))
            self.assertEqual(list(map(toOffSizes, rboxesB.regions(domB))), list(rboxesB.bounds(domB)))
            self.assertEqual(list(map(toOffSizes, rboxesC.regions(domC))), list(rboxesC.bounds(domC)))

            # Summing one per edge should give us the number of
            # edges, no matter whether we ask for the dummy
            # property or not.
            msg = "cross sum mismatch for regions %s vs %s, enumerating [%s %s %s]" % \
                (regsA, regsB,
                 domA.name if enumA else '',
                 domB.name if enumB else '',
                 domC.name if enumC else '')
            def onePropEdge(rbA, rbB): return 1
            self.assertEqual(rboxesA.zipCrossSum(rboxesB, rboxesC, onePropEdge),edges,msg=msg)
            self.assertEqual(rboxesB.zipCrossSum(rboxesA, rboxesC, onePropEdge),edges,msg='reversed ' + msg)

class DataFlowTestsSymbol(unittest.TestCase):

    def test_symbols(self):

        # Make a region with entirely symbolised properties
        dom = Domain('Symbolized')
        size = Symbol('size')
        regs = dom.regions(size)
        i = Symbol("i")
        split = regs.split(size, props = {
            'p1': 1,
            'p2': Lambda(i, 1),
            'p3': Lambda(i, i) })

        # Check size
        rbox = RegionBoxes([split])
        self.assertEqual(rbox.count(), size)
        self.assertEqual(rbox.sum(dom('p1')), size)
        self.assertEqual(rbox.sum(dom('p2')).doit(), size)
        self.assertEqual(rbox.sum(dom('p3')).doit(), size**2/2-size/2)

class DataFlowTestsFlow(unittest.TestCase):

    def setUp(self):

        # Again, set up some domains and splits so we can build a
        # meaningful flow network.
        self.size1 = 12.
        self.dom1 = Domain("Domain 1")
        self.reg1 = self.dom1.regions(self.size1)
        self.split1 = self.reg1.split(self.size1)
        self.size2 = 7.
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.regions(self.size2, props={'testCost': 100 })
        self.split2 = self.reg2.split(self.size2, props={'testCost': 100 })
        self.size3 = 5.
        self.dom3 = self.dom2
        self.reg3 = self.dom3.regions(self.size3)
        self.split3 = self.reg3.split(self.size3)

    def _makeFlows(self):

        # Make some input nodes without dependencies
        input1 = Flow('Global Input', [], cluster='input', costs = { 'transfer': 12 })
        input2 = Flow('Input Domain 1',
                      [self.split1], cluster='input',
                      costs = { 'transfer': self.dom1('size') })
        input3 = Flow('Input Domain 1 + 2',
                      [self.reg1, self.split2], cluster='input',
                      costs = { 'transfer': self.dom2('size') })

        # Make some intermediate nodes
        interg = Flow('Intermediate Global', [],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [input1, input3]
        )
        self.inter1_name = 'Intermediate Domain 1'
        inter1 = Flow(self.inter1_name, [self.split1],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [interg, input2, input3]
        )
        self.inter2_name = 'Intermediate Domain 2'
        inter2 = Flow(self.inter2_name, [self.split2],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [interg, inter1, input3]
        )
        self.inter3_name = 'Intermediate Domain 3'
        inter3 = Flow(self.inter3_name, [self.split3],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [inter1, inter2, input3]
        )
        inter4 = Flow('Intermediate Domain 1+3', [self.split1, self.split3],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [inter1, inter2, input3]
        )

        # Tie them together
        self.gather_name = 'Gather'
        gather = Flow(self.gather_name, [self.reg2],
                      costs = {
                          'compute': 1000,
                          'transfer': self.dom2('testCost') * self.dom2('size')
                      },
                      deps = [inter3, inter4]
        )

        # Do a loop
        self.loop_name = 'Loop'
        loop = Flow(self.loop_name, [self.split3],
                    costs = { 'compute': 1000, 'transfer': 1000},
                    deps = [gather]
        )
        inter2.depend(loop)

        # Add an output node
        output = Flow('Output', [self.split2],
                      costs = {}, deps = [gather])
        return output

    def testFlows(self):

        root = self._makeFlows()

        # Make sure we have all flows in the recursive dependencies
        self.assertEqual(len(root.recursiveDeps()), 11)

        # Make sure all vanish if we remove the main dependency
        dep = root.deps[0]
        root.removeDepend(dep)
        self.assertEqual(len(root.recursiveDeps()), 1)

        # Make sure weights are consistent
        self.assertEqual(len(root.deps), len(root.weights))

    def testDot(self):

        # Run GraphViz generation with all optional features
        root = self._makeFlows()
        dot = flowsToDot(root, t=1.0, computeSpeed=Constants.tera, showTaskRates=True)
        self.assertGreater(len(dot.source), 3000)

        # Analyse crossings for split2
        dot = flowsToDot(root, t=1.0, cross_regs=[self.split2])
        self.assertGreater(len(dot.source), 3000)

        # We would expect a number of different crossing degrees:
        #  '0.0'  - we have edges that stay fully inside
        #  '80.0' - for split3->reg2 (keep in mind that split3 is the same domain)
        #  '85.7' - for reg2->split2
        #  '100'  - for all cases where domain 2 is not present
        # (this check might be a bit fragile...)
        import re
        self.assertEqual(len(set(re.findall("crossing: ([0-9.]+)%", dot.source))), 4)

    def testCost(self):

        # Make sure the cost sum is what we expect
        root = self._makeFlows()
        self.assertEqual(root.recursiveCost('compute'), 91000)
        self.assertEqual(root.recursiveCost('transfer'), 90731)

    def testMerge(self):

        root = self._makeFlows()
        gather = root.getDep(self.gather_name)
        loop = root.getDep(self.loop_name)
        inter1 = root.getDep(self.inter1_name)
        inter2 = root.getDep(self.inter2_name)
        inter3 = root.getDep(self.inter3_name)
        compute = root.recursiveCost('compute')
        produce = root.recursiveCost('transfer')
        transfer = root.recursiveEdgeCost('transfer')

        # First test a No-Op (replace with a Flow of the same granularity)
        gather_i = mergeFlows(root, [gather], [self.reg2])
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # If we change to coarser granularity produced data as well as
        # transfered data should reduce - we have removed (size3-1)
        # tasks and their input edges from "Gather".
        mergeFlows(root, [loop], [self.reg3], reevalTransfer=True)
        produce -= (self.size3 - 1) * 1000
        transfer -= (self.size3 - 1) * (100 * self.size2)
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # Try merging something into the gather. This will remove some
        # transfer from the overall network (working out the details
        # takes a bit of work with pen and paper, but it checks out.)
        gather_m = mergeFlows(root, [gather_i, inter3], [self.reg2])
        produce -= 1000 * self.size3
        transfer -= 1000 * (self.size1 * self.size3 + 2 * self.size3) + self.size3
        transfer += 1000 * (self.size1 + self.size2) + self.size2
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # This should be the same as if we remove the domain, in this
        # case.
        gather_m2 = mergeFlows(root, [gather_m], [])
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

    def testSplit(self):

        root = self._makeFlows()
        gather = root.getDep(self.gather_name)
        compute = root.recursiveCost('compute')
        produce = root.recursiveCost('transfer')
        transfer = root.recursiveEdgeCost('transfer')

        # Make a splitter node after the gather flow. Total data
        # produced will stay the same, but transfer goes down because
        # we have stopped sending the whole task result to the "loop"
        # and "output" nodes.
        makeSplitFlow(root, gather, [self.split2])
        transfer -= (self.size2 + self.size3) * self.size2 * 100
        transfer += (self.size2 + self.size3) * 100
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

if __name__ == '__main__':
    unittest.main()
