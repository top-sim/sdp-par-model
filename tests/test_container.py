
import unittest

from sdp_par_model.parameters.container import *
from sympy import Mul

class ContainerTests(unittest.TestCase):

    def test_bldep(self):

        b = Symbol('_b')
        b2 = Symbol('_b2')
        bcount = Symbol('bcount')

        bb = BLDep(b, b)
        self.assertEqual(bb(1000), 1000)
        self.assertEqual(BLDep({'b': b}, b)(2000), 2000)
        self.assertEqual(BLDep(b2, b2)(2000), 2000)

        # Sum should multiply in baseline count
        bs = blsum(b, b)
        self.assertEqual(bs(2000, bcount=1), 2000)
        self.assertEqual(bs(2000, bcount=2), 4000)
        self.assertEqual(bs(1000, bcount=10), 10000)
        self.assertEqual(blsum(b, b)(1000, bcount=10), 10000)
        self.assertEqual(blsum({'b':b}, b)(1000, bcount=10), 10000)

        # Check operators
        self.assertEqual((2 * bb)(1000), 2000)
        self.assertEqual((2 / bb)(1000), 1/500)
        self.assertEqual((bb * 2)(1000), 2000)
        self.assertEqual((bb / 2)(1000), 500)
        self.assertEqual((bb * bb)(1000), 1000000)
        self.assertEqual((bb / bb)(1000), 1)

        # Check free_symbols
        free = Symbol('free')
        self.assertEqual(bb.free_symbols, set())
        self.assertEqual(BLDep(b, free).free_symbols, set([free]))
        self.assertEqual(BLDep(b, b + free).free_symbols, set([free]))

        # Check sum
        bins = [
            { 'b': 10000, 'bcount': 10 },
            { 'b': 20000, 'bcount': 20 },
            { 'b': 40000, 'bcount': 10 },
            { 'b': 60000, 'bcount': 5 },
            ]
        self.assertEqual(bb.eval_sum(bins).doit(), 130000)
        self.assertEqual(bs.eval_sum(bins).doit(), 1200000)
        self.assertEqual(blsum(b, 1).eval_sum(bins).doit(), 45)
        self.assertEqual(blsum(b, free).eval_sum(bins).doit(), free*45)
        self.assertEqual(blsum(b, free).eval_sum(bins).doit(), free*45)
        # Make sure it moved the free variable outside
        self.assertTrue(isinstance(blsum(b, free).eval_sum(bins), Mul))

        # Check symbolic sum
        i = Symbol('i')
        n = Symbol('n')
        B = Symbol('B')
        Bcount = Symbol('bcount')
        Bsum = Symbol('Bsum')
        symbins = (i, 1, n, {'b': B(i), 'bcount': Bcount(i) })
        known_sum = { Bcount: Bsum }
        self.assertEqual(bb.eval_sum(symbins).doit(), Sum(B(i), (i, 1, n)))
        self.assertEqual(bs.eval_sum(symbins).doit(), Sum(Bcount(i) * B(i), (i, 1, n)))
        self.assertEqual(BLDep(b, 1).eval_sum(symbins).doit(), n)
        self.assertEqual(blsum(b, 1).eval_sum(symbins).doit(), Sum(Bcount(i), (i, 1, n)))
        self.assertEqual(blsum(b, 1).eval_sum(symbins, known_sum).doit(), Bsum)
        self.assertEqual(BLDep(b, free).eval_sum(symbins).doit(), free * n)
        self.assertEqual(blsum(b, free*b).eval_sum(symbins).doit(), free * Sum(Bcount(i) * B(i), (i, 1, n)))
        self.assertEqual(BLDep(b, free).subs({free: b}).eval_sum(symbins).doit(), Sum(B(i), (i, 1, n)))

if __name__ == '__main__':
    unittest.main()
