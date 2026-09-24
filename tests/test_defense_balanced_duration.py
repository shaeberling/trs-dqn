import unittest

import numpy as np

from rl.defense_balanced_duration import (balanced_duration_initial_bias,
                                          group_duration_logits_numpy,
                                          grouped_duration_physical_actions,
                                          grouped_duration_log_probs_numpy)


class BalancedDurationInitialBiasTests(unittest.TestCase):
    def test_equal_physical_mass_within_each_duration(self):
        weights = (.88, .08, .03, .01)
        logits = balanced_duration_initial_bias((1, 4, 16, 64), weights)
        self.assertEqual(logits.shape, (80,))
        probabilities = np.exp(logits - np.logaddexp.reduce(logits)).reshape(4, 20)
        for index, expected in enumerate(weights):
            physical = np.concatenate((probabilities[index, :9],
                                       [probabilities[index, 9:18].sum()],
                                       probabilities[index, 18:]))
            np.testing.assert_allclose(physical, np.full(12, expected / 12), atol=1e-7)
            self.assertAlmostEqual(float(probabilities[index].sum()), expected, delta=2e-6)

    def test_weights_are_normalized_without_directional_change(self):
        scaled = balanced_duration_initial_bias((1, 4), (9, 1))
        normalized = balanced_duration_initial_bias((1, 4), (.9, .1))
        np.testing.assert_array_equal(scaled, normalized)
        original = np.array((9., 1.))
        balanced_duration_initial_bias((1, 4), original)
        np.testing.assert_array_equal(original, np.array((9., 1.)))

    def test_duration_rows_group_fire_independently(self):
        logits = np.arange(40, dtype=np.float32).reshape(1, 40) / 11
        grouped = group_duration_logits_numpy(logits, 2)
        self.assertEqual(grouped.shape, (1, 24))
        np.testing.assert_array_equal(grouped[0, :9], logits[0, :9])
        np.testing.assert_array_equal(grouped[0, 10:12], logits[0, 18:20])
        np.testing.assert_array_equal(grouped[0, 12:21], logits[0, 20:29])
        self.assertAlmostEqual(float(grouped[0, 9]),
                               float(np.logaddexp.reduce(logits[0, 9:18])), places=6)
        self.assertAlmostEqual(float(grouped[0, 21]),
                               float(np.logaddexp.reduce(logits[0, 29:38])), places=6)

    def test_grouped_executor_uses_original_keyboard_indices(self):
        np.testing.assert_array_equal(grouped_duration_physical_actions(np.arange(12)),
                                      np.array((*range(10), 18, 19)))
        with self.assertRaises(ValueError):
            grouped_duration_physical_actions(np.array([12]))

    def test_mixture_preserves_physical_key_marginal(self):
        logits = np.arange(48, dtype=np.float32).reshape(1, 48) / 13
        original = np.exp(grouped_duration_log_probs_numpy(logits, 4, 0)).reshape(1, 4, 12)
        mixed = np.exp(grouped_duration_log_probs_numpy(logits, 4, .2)).reshape(1, 4, 12)
        np.testing.assert_allclose(mixed.sum(axis=1), original.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(mixed.sum(axis=(1, 2)), np.ones(1), atol=1e-6)
        self.assertGreater(mixed[0, 0, 0], original[0, 0, 0])

    def test_invalid_grouped_likelihood_inputs(self):
        for mix in (-.1, 1, float("nan")):
            with self.subTest(mix=mix):
                with self.assertRaises(ValueError):
                    grouped_duration_log_probs_numpy(np.zeros((1, 24)), 2, mix)
        with self.assertRaises(ValueError):
            grouped_duration_log_probs_numpy(np.zeros((1, 20)), 2, .1)

    def test_invalid_spec_and_weights(self):
        for durations, weights in (((1, 4), (1,)),
                                   ((1, 4), (1, 0)),
                                   ((1, 4), (1, float("nan"))),
                                   ((1, 4), (1e308, 1e308)),
                                   ((1, 4), (1e-300, 1e300)),
                                   ((4, 16), (1, 1))):
            with self.subTest(durations=durations, weights=weights):
                with self.assertRaises(ValueError):
                    balanced_duration_initial_bias(durations, weights)


if __name__ == "__main__":
    unittest.main()
