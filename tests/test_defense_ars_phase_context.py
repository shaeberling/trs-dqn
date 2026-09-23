import unittest
from pathlib import Path

import numpy as np

from rl.defense_ars_phase_context import phase_basis, visible_phase_subspace
from rl.model import QNetwork


class PhaseContextTests(unittest.TestCase):
    def test_two_own_screen_phase_changes_are_independent_and_centered(self):
        rng = np.random.default_rng(12)
        earlier = rng.normal(size=(12, 4)).astype(np.float32)
        middle = earlier + np.array([1, 0, 0, 0], np.float32)
        later = middle + np.array([0, 2, 0, 0], np.float32)
        basis, report = phase_basis(earlier, middle, later)
        self.assertEqual(basis.shape, (2, 5))
        self.assertEqual(report['components'], 2)
        means = [batch@basis[:, :-1].T+basis[:, -1]
                 for batch in (earlier, middle, later)]
        self.assertAlmostEqual(float(means[0][:, 0].mean()), 0., places=5)
        self.assertAlmostEqual(float(means[1][:, 0].mean()), 1., places=5)
        self.assertAlmostEqual(float(means[1][:, 1].mean()), 0., places=5)
        self.assertAlmostEqual(float(means[2][:, 1].mean()), 1., places=5)
        self.assertAlmostEqual(float(np.dot(basis[0, :-1], basis[1, :-1])), 0., places=5)

    def test_rejects_no_independent_late_change_and_invalid_features(self):
        early = np.arange(40, dtype=np.float32).reshape(10, 4)
        middle = early + np.array([1, 0, 0, 0], np.float32)
        with self.assertRaises(ValueError):
            phase_basis(early, middle, middle + np.array([2, 0, 0, 0], np.float32))
        bad = middle.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            phase_basis(early, bad, middle)

    def test_verified_real_own_screens_build_frozen_encoder_phase_basis(self):
        checkpoint = Path('results/defense/training/ars-early-91/milestone-000005')
        archive = Path('results/defense/training/ars-bottleneck-source-92')
        model = QNetwork(action_count=20)
        model.load_weights(str(checkpoint/'model.safetensors'))
        basis, provenance = visible_phase_subspace(model, archive)
        self.assertEqual(basis.shape, (2, 257))
        self.assertTrue(np.isfinite(basis).all())
        self.assertEqual(provenance['diagnostics']['examples'], 46)
        self.assertEqual(len(provenance['omitted_own_lives']), 2)
        self.assertFalse(provenance['native_snapshot_read'])
        self.assertFalse(provenance['action_target_read'])


if __name__ == '__main__':
    unittest.main()
