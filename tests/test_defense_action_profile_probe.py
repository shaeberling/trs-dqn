"""The diagnostic action subset is fixed and leaves stage-one controls usable."""

import unittest

import numpy as np

from rl.defense import action_names
from rl.defense_action_profile_probe import distinct_stage_one_logits, UNIQUE_STAGE_ONE_COMMANDS


class DistinctCommandProbeTests(unittest.TestCase):
    def test_fixed_subset_retains_noop_all_directions_and_fire(self):
        self.assertEqual(UNIQUE_STAGE_ONE_COMMANDS, tuple(range(10)))
        self.assertEqual(action_names()[:10], (
            "NOOP", "UP", "DOWN", "LEFT", "RIGHT", "UP+LEFT", "UP+RIGHT",
            "DOWN+LEFT", "DOWN+RIGHT", "SPACE"))
        logits = np.arange(40, dtype=np.float32).reshape(2, 20)
        result = distinct_stage_one_logits(logits)
        np.testing.assert_array_equal(result, logits[:, :10])
        self.assertEqual(result.shape, (2, 10))

    def test_rejects_bad_logits(self):
        for values in (np.zeros((20,), np.float32), np.zeros((1, 9), np.float32),
                       np.full((1, 20), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                distinct_stage_one_logits(values)


if __name__ == "__main__":
    unittest.main()
