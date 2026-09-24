"""Fixed diagnostic grid and three-phase action-overlay checks."""

import unittest

import numpy as np

from rl.defense_three_phase_grid import COMMANDS, DELAYS, HOLDS, grid, phase_plan


class ThreePhaseGridTests(unittest.TestCase):
    def test_fixed_direction_symmetric_cardinality_and_unique_plans(self):
        self.assertEqual(len(COMMANDS), 10)
        self.assertEqual(len(DELAYS)*len(HOLDS)**3*len(COMMANDS)**3, 81000)
        self.assertEqual(len(set(grid())), 81000)

    def test_three_contiguous_phases_leave_prefix_and_suffix_intact(self):
        baseline = np.arange(24, dtype=np.uint8)
        plan = phase_plan(baseline, 2, 3, 4, 2, 5, 4, 6)
        np.testing.assert_array_equal(plan[:2], baseline[:2])
        np.testing.assert_array_equal(plan[2:5], 4)
        np.testing.assert_array_equal(plan[5:7], 5)
        np.testing.assert_array_equal(plan[7:11], 6)
        np.testing.assert_array_equal(plan[11:], baseline[11:])
        np.testing.assert_array_equal(baseline, np.arange(24, dtype=np.uint8))
        with self.assertRaises(ValueError):
            phase_plan(baseline, 20, 3, 4, 2, 5, 4, 6)
        with self.assertRaises(ValueError):
            phase_plan(baseline, 2, 3, 4, 2, 5, 4, 20)


if __name__ == "__main__":
    unittest.main()
