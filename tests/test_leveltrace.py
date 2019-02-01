
import unittest

from sdp_par_model import *
from sdp_par_model.scheduling.level_trace import *

class TestLevelTrace(unittest.TestCase):

    def test_init(self):
        self.assertEqual(LevelTrace(), LevelTrace())
        self.assertEqual(LevelTrace({}), LevelTrace())
        self.assertEqual(LevelTrace(LevelTrace()), LevelTrace())
        self.assertEqual(eval(repr(LevelTrace())), LevelTrace())
        trace = LevelTrace({0:1, 1:0})
        self.assertNotEqual(trace, LevelTrace())
        self.assertEqual(LevelTrace(trace), trace)
        self.assertEqual(eval(repr(trace)), trace)
        self.assertRaises(ValueError, LevelTrace, {1:1})
        self.assertRaises(ValueError, LevelTrace, {1:1, 0:0})

    def test_get(self):
        trace = LevelTrace({0:1,1:0,2:1,3:0})
        self.assertEqual(trace.get(-1e15), 0)
        self.assertEqual(trace.get(-0.0001), 0)
        self.assertEqual(trace.get(0), 1)
        self.assertEqual(trace.get(0.5), 1)
        self.assertEqual(trace.get(1), 0)
        self.assertEqual(trace.get(2), 1)
        self.assertEqual(trace.get(3), 0)
        self.assertEqual(trace.get(1e15), 0)

    def test_add(self):
        trace = LevelTrace()

        trace.add(0, 1, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0}))
        trace.add(1, 2, 1)
        self.assertEqual(trace, LevelTrace({0:1,2:0}))
        trace.add(0, 2, -1)
        self.assertEqual(trace, LevelTrace())

        trace.add(0, 1, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0}))
        trace.add(2, 3, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0,2:1,3:0}))
        trace.add(1, 2, 1)
        self.assertEqual(trace, LevelTrace({0:1,3:0}))
        trace.add(0, 3, -1)
        self.assertEqual(trace, LevelTrace())

        trace.add(0, 1, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0}))
        trace.add(2, 3, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0,2:1,3:0}))
        trace.add(1.5, 2.5, 1)
        self.assertEqual(trace, LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0}))
        trace.add(0, 3, -1)
        self.assertEqual(trace, LevelTrace({1:-1,1.5:0,2:1,2.5:0}))

    def test_minmax(self):
        trace = LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0})

        self.assertEqual(trace.maximum(0,3), 2)
        self.assertEqual(trace.maximum(-1,0), 0)
        self.assertEqual(trace.maximum(-1,0.1), 1)
        self.assertEqual(trace.maximum(0,1), 1)
        self.assertEqual(trace.maximum(0,1.5), 1)
        self.assertEqual(trace.maximum(0.9,1.5), 1)
        self.assertEqual(trace.maximum(1,1.5), 0)
        self.assertEqual(trace.maximum(1,1.6), 1)

        self.assertEqual(trace.minimum(0,3), 0)
        self.assertEqual(trace.minimum(-1,0.1), 0)
        self.assertEqual(trace.minimum(0,1), 1)
        self.assertEqual(trace.minimum(0,1.1), 0)
        self.assertEqual(trace.minimum(1.5,2), 1)
        self.assertEqual(trace.minimum(2,2.5), 2)
        self.assertEqual(trace.minimum(1,2.6), 0)

    def test_integrate(self):
        trace = LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0})

        self.assertEqual(trace.integrate(0, 1), 1)
        self.assertEqual(trace.integrate(-1, 1.5), 1)
        self.assertEqual(trace.integrate(1, 2), 0.5)
        self.assertEqual(trace.integrate(1, 2.125), 0.75)
        self.assertEqual(trace.integrate(0.75, 2.125), 1)
        self.assertEqual(trace.integrate(0.75, 2.5), 1.75)
        self.assertEqual(trace.integrate(0.75, 3), 2.25)
        self.assertEqual(trace.integrate(0, 3), 3)
        self.assertEqual(trace.integrate(0, 4), 3)
        self.assertEqual(trace.average(0, 3), 1)

    def test_above_below(self):
        trace = LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0})

        self.assertEqual(trace.find_above(-1,0), -1)
        self.assertIsNone(trace.find_above(-1,2.01))
        self.assertEqual(trace.find_above(0,0), 0)
        self.assertEqual(trace.find_above(0,1), 0)
        self.assertEqual(trace.find_above(1,1), 1.5)
        self.assertEqual(trace.find_above(0,2), 2)
        self.assertIsNone(trace.find_above(2.5,2))
        self.assertEqual(trace.find_above(2.6,1), 2.6)
        self.assertIsNone(trace.find_above(3,0.1))
        self.assertEqual(trace.find_above(4,0), 4)

        self.assertEqual(trace.find_below(-1,0), -1)
        self.assertIsNone(trace.find_below(-1,-0.1))
        self.assertEqual(trace.find_below(1,0), 1)
        self.assertEqual(trace.find_below(2,1), 2.5)
        self.assertEqual(trace.find_below(2,0), 3)

        self.assertEqual(trace.find_below_backward(4, 0), 4)
        self.assertEqual(trace.find_below_backward(3, 0), 1.5)
        self.assertEqual(trace.find_below_backward(3, 1), 3)
        self.assertEqual(trace.find_below_backward(4, 1), 4)
        self.assertEqual(trace.find_below_backward(2.6, 1), 2.6)
        self.assertEqual(trace.find_below_backward(2.5, 1), 2)
        self.assertEqual(trace.find_below_backward(2.4, 1), 2)
        self.assertEqual(trace.find_below_backward(1.5, 0), 1.5)
        self.assertEqual(trace.find_below_backward(1.6, 0), 1.5)
        self.assertEqual(trace.find_below_backward(1.4, 0), 1.4)
        self.assertEqual(trace.find_below_backward(1, 0), 0)
        self.assertEqual(trace.find_below_backward(1, 1), 1)
        self.assertEqual(trace.find_below_backward(0.1, 0), 0)
        self.assertEqual(trace.find_below_backward(0, 0), 0)
        self.assertEqual(trace.find_below_backward(-0.1, 0), -0.1)
        self.assertIsNone(trace.find_below_backward(4, -0.1))
        self.assertIsNone(trace.find_below_backward(-1, -0.1))
        trace.add(1.2,2.2,-2)
        self.assertEqual(trace.find_below_backward(4, -0.1), 2)
        self.assertEqual(trace.find_below_backward(4, -2), 1.5)
        self.assertIsNone(trace.find_below_backward(1.2, -2))
        self.assertEqual(trace.find_below_backward(1.5, -2), 1.5)

    def test_period(self):
        trace = LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0})
        self.assertEqual(trace.find_period_below(-1, 0, 0, 1), -1)
        self.assertEqual(trace.find_period_below(0, 1, 1, 1), 0)
        self.assertEqual(trace.find_period_below(-1, 1, 0, 1), -1)
        self.assertIsNone(trace.find_period_below(-1, 1, 0, 1.1))
        self.assertEqual(trace.find_period_below(0, 2, 0, 0.5), 1)
        self.assertEqual(trace.find_period_below(1.5, 3, 1, 0.5), 1.5)
        self.assertEqual(trace.find_period_below(1.6, 3, 1, 0.5), 2.5)
        self.assertEqual(trace.find_period_below(1.6, 5, 0, 0.5), 3)
        self.assertEqual(trace.find_period_below(0, 2, 1, 2), 0)
        self.assertIsNone(trace.find_period_below(0, 4.5, 1, 2.125))
        self.assertEqual(trace.find_period_below(0, 4.625, 1, 2.125), 2.5)
        self.assertEqual((-trace).find_period_below(0, 4, -1, 1.5), 1.5)

    def test_zipwith(self):

        trace = LevelTrace({0:1,1:0,1.5:1,2:2,2.5:1,3:0})
        self.assertEqual(trace - trace, LevelTrace())
        self.assertEqual(trace + (-trace), LevelTrace())
        self.assertEqual(trace + trace, trace.map(lambda x: x*2))
        for x in [1,2,3,4]:
            trace2 = LevelTrace({2.5-x:-1, x+1:0})
            trace3 = LevelTrace(trace)
            trace3.add(2.5-x, x+1, -1)
            self.assertEqual(trace + trace2, trace3)
            self.assertEqual(trace2 + trace, trace3)
            self.assertEqual(-trace2 - trace, -trace3)
            self.assertEqual(-trace - trace2, -trace3)
            self.assertEqual(trace - trace3, -trace2)
            self.assertEqual(-trace3 + trace, -trace2)
            self.assertEqual(-trace + trace3, trace2)
            self.assertEqual(trace3 - trace, trace2)

if __name__ == '__main__':
    unittest.main()
