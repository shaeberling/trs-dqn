"""Canonical fire grouping preserves mass without a game-state rule."""

import unittest

import numpy as np

from rl.defense_action_group_probe import COMMAND_MAP, group_fire_logits


class ActionGroupProbeTests(unittest.TestCase):
    def test_fire_mass_and_other_command_mass_preserved(self):
        self.assertEqual(COMMAND_MAP.tolist(), [*range(10), 18, 19])
        logits = np.arange(40, dtype=np.float64).reshape(2, 20)/10
        grouped = group_fire_logits(logits)
        self.assertEqual(grouped.shape, (2, 12))
        original = np.exp(logits-np.logaddexp.reduce(logits, axis=1, keepdims=True))
        compact = np.exp(grouped-np.logaddexp.reduce(grouped, axis=1, keepdims=True))
        np.testing.assert_allclose(compact[:, :9], original[:, :9], atol=1e-14)
        np.testing.assert_allclose(compact[:, 9], original[:, 9:18].sum(axis=1), atol=1e-14)
        np.testing.assert_allclose(compact[:, 10:], original[:, 18:], atol=1e-14)

    def test_rejects_bad_logits(self):
        for values in (np.zeros((20,), np.float32), np.zeros((1, 12), np.float32),
                       np.full((1, 20), np.inf, np.float32)):
            with self.assertRaises(ValueError):
                group_fire_logits(values)


if __name__ == "__main__":
    unittest.main()
