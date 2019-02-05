
import unittest
import random

from sdp_par_model.scheduling.graph import *
from sdp_par_model.scheduling.scheduler import *

class TestLevelTrace(unittest.TestCase):

    def setUp(self):
        self.tasks = {
            i : Task(str(i), '', '', 1, {}, {})
            for i in range(10)
        }
        self.tasks[1].depend(self.tasks[0])
        self.tasks[2].depend(self.tasks[0])
        self.tasks[3].depend(self.tasks[1])
        self.tasks[4].depend(self.tasks[1])
        self.tasks[4].depend(self.tasks[2])
        self.tasks[5].depend(self.tasks[0])
        self.tasks[6].depend(self.tasks[4])
        self.tasks[7].depend(self.tasks[5])
        self.tasks[7].depend(self.tasks[6])
        self.tasks[8].depend(self.tasks[5])
        self.tasks[9].depend(self.tasks[7])
        self.tasks[9].depend(self.tasks[8])

    def _assertTopoSorted(self, tasks):
        for i, t in enumerate(tasks):
            for d in t.deps:
                if d in tasks:
                    self.assertIn(d, tasks[:i])

    def test_toposort(self):
        # Tasks are already topologically sorted
        tasks = list(self.tasks.values())
        self._assertTopoSorted(tasks)
        self.assertEqual(toposort(tasks), tasks)
        # Reverse, check property
        tasks = list(reversed(list(self.tasks.values())))
        self._assertTopoSorted(toposort(tasks))
        # Shuffle, check property
        for i in range(len(self.tasks)):
            tasks = list(self.tasks.values())[i:]
            for _ in range(100):
                random.shuffle(tasks)
                self._assertTopoSorted(toposort(tasks))

    def test_active_deps(self):

        active = list(self.tasks.values())
        for t, res in [(0, [0,1,2,3,4,5,6,7,8,9]),
                       (1, [1,3,4,6,7,9]), (5,[5,7,8,9]), (3,[3])]:
            ad = active_dependencies([self.tasks[t]], active)
            self._assertTopoSorted(ad)
            self.assertEqual(set(ad), set([self.tasks[x] for x in res]))

        active = [self.tasks[x] for x in [0,1,5,3,4]]
        for t, res in [(0, [0,1,3,5,4]),
                       (1, [1,3,4]), (5,[5]), (3,[3])]:
            ad = active_dependencies([self.tasks[t]], active)
            self._assertTopoSorted(ad)
            self.assertEqual(set(ad), set([self.tasks[x] for x in res]))

if __name__ == '__main__':
    unittest.main()
