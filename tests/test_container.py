
import unittest

from sdp_par_model.parameters.container import *
from sympy import Mul

class ContainerTests(unittest.TestCase):

    def test_bldep(self):

        b = Symbol('_b')
        b2 = Symbol('_b2')
        bcount = Symbol('bcount')

        self.assertEqual(BLDep(b, b)(1000), 1000)
        self.assertEqual(BLDep(b2, b2)(2000), 2000)
        self.assertEqual(BLDep({'b': b}, b)(2000), 2000)
        self.assertEqual(BLDep({'c': b}, b)(c=2000), 2000)
        self.assertEqual(BLDep({'b': b, 'c': b2}, b)(b=0, c=2000), 0)
        self.assertEqual(BLDep({'b': b, 'c': b2}, b2)(b=0, c=2000), 2000)
        self.assertEqual(BLDep({'b': b, 'c': b2}, b * b2)(b=20, c=2000), 40000)
        self.assertEqual(BLDep(b,b,defaults={'b':1000})(), 1000)
        self.assertEqual(BLDep({'b': b, 'c': b2}, b * b2, defaults={'b':20})(c=2000), 40000)

    def test_blsum(self):

        # Sum should multiply in baseline count
        b = Symbol('_b')
        b2 = Symbol('_b2')
        bs = blsum(b, b)
        self.assertEqual(bs(2000, bcount=1), 2000)
        self.assertEqual(bs(2000, bcount=2), 4000)
        self.assertEqual(bs(1000, bcount=10), 10000)
        self.assertEqual(blsum(b, b)(1000), 1000)
        self.assertEqual(blsum(b, b)(1000, bcount=10), 10000)
        self.assertEqual(blsum(b2, bs(b2))(1000, bcount=10), 10000)
        self.assertEqual(blsum({'b':b}, b)(1000, bcount=10), 10000)

    def test_operators(self):

        # Check operators
        b = Symbol('_b')
        bb = BLDep(b,b)
        self.assertEqual((2 * bb)(1000), 2000)
        self.assertEqual((2 / bb)(1000), 1/500)
        self.assertEqual((bb * 2)(1000), 2000)
        self.assertEqual((bb / 2)(1000), 500)
        self.assertEqual((bb * bb)(1000), 1000000)
        self.assertEqual((bb / bb)(1000), 1)

    def test_free(self):

        # Check free_symbols
        b = Symbol('_b')
        free = Symbol('free')
        self.assertEqual(BLDep(b, b).free_symbols, set())
        self.assertEqual(BLDep(b, free).free_symbols, set([free]))
        self.assertEqual(BLDep(b, b + free).free_symbols, set([free]))

    def test_sum(self):
        b = Symbol('_b')
        bb = BLDep(b,b)
        bs = blsum(b, b)
        free = Symbol('free')

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

    def test_symbolify(self):

        tp = ParameterContainer()

        b = Symbol('b')
        tp.Nbl = 5
        tp.Ntest = 5
        tp.Nbldep = BLDep(b, b)
        tp.Nblsum = blsum(b, b)
        tp.products['Test'] = { 'Rflop': 20 }

        tp.symbolify()
        ib = tp.baseline_bins[0]
        Nbl = tp.Nbl

        # Everything should be replaced by symbolified versions now
        self.assertEqual(str(tp.Ntest), "N_test")
        self.assertEqual(str(tp.products['Test']['Rflop']), "R_flop,Test")
        self.assertEqual(tp.baseline_bins[:3], (ib, 1, Nbl))
        self.assertEqual(str(tp.Nbldep(1)), "N_bldep(1)")
        self.assertEqual(str(tp.Nblsum(2)), "N_blsum(2, 1)")

        # Evaluating a new baseline-dependent sum should yield a sympy expression
        Bmax = tp.baseline_bins[3]['b']
        self.assertEqual(BLDep(b,b).eval_sum(tp.baseline_bins),
                         Sum(Bmax, (ib, 1, Nbl)))
        self.assertEqual(blsum(b,3*b).eval_sum(tp.baseline_bins),
                         3*Sum(Bmax, (ib, 1, Nbl)))

if __name__ == '__main__':
    unittest.main()
