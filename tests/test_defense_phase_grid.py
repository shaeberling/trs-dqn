import unittest

import numpy as np

from rl.defense_phase_grid import COMMANDS, DELAYS, HOLDS, grid, phase_plan


class DefensePhaseGridTests(unittest.TestCase):
    def test_fixed_grid_is_exhaustive_and_symmetric(self):
        plans = list(grid())
        self.assertEqual(len(plans), 7500)
        self.assertEqual(len(set(plans)), len(plans))
        self.assertEqual(COMMANDS, tuple(range(10)))
        self.assertEqual(DELAYS, (0, 8, 16))
        self.assertEqual(HOLDS, (4, 8, 16, 24, 32))
        self.assertEqual({plan[2] for plan in plans}, set(COMMANDS))
        self.assertEqual({plan[4] for plan in plans}, set(COMMANDS))

    def test_phase_replacement_preserves_source_outside_both_segments(self):
        baseline = np.arange(100, dtype=np.uint8)
        result = phase_plan(baseline, 8, 16, 4, 24, 7)
        np.testing.assert_array_equal(result[:8], baseline[:8])
        np.testing.assert_array_equal(result[8:24], np.full(16, 4, np.uint8))
        np.testing.assert_array_equal(result[24:48], np.full(24, 7, np.uint8))
        np.testing.assert_array_equal(result[48:], baseline[48:])
        np.testing.assert_array_equal(baseline, np.arange(100, dtype=np.uint8))
        with self.assertRaises(ValueError):
            phase_plan(baseline, 80, 16, 4, 24, 7)
        with self.assertRaises(ValueError):
            phase_plan(baseline, 8, 16, 20, 24, 7)
        with self.assertRaises(ValueError):
            list(grid(commands=(0, 0)))


if __name__ == "__main__":
    unittest.main()
