"""Pure action-group likelihood checks for the read-only Defense audit."""

import unittest

import numpy as np

from rl.defense_route_likelihood import physical_indices, physical_log_probs, likelihood_summary


class DefenseRouteLikelihoodTests(unittest.TestCase):
    def test_stage_one_fire_aliases_are_one_physical_action(self):
        actions = np.arange(20, dtype=np.int32)
        expected = np.array([*range(10), *([9] * 8), 10, 11], np.int32)
        np.testing.assert_array_equal(physical_indices(actions), expected)

    def test_grouped_probability_sums_fire_aliases(self):
        logits = np.zeros((2, 20), np.float32)
        logp = physical_log_probs(logits, np.array([9, 17], np.int32))
        np.testing.assert_allclose(np.exp(logp), np.array([9/20, 9/20]), rtol=1e-6)
        self.assertEqual(likelihood_summary(logp, np.array([9, 17]))["physical_command_counts"][9], 2)

    def test_invalid_action_and_length_fail_closed(self):
        with self.assertRaises(ValueError):
            physical_indices(np.array([20]))
        with self.assertRaises(ValueError):
            physical_log_probs(np.zeros((2, 20)), np.array([0]))


if __name__ == "__main__":
    unittest.main()
