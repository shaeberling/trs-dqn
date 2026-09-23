from pathlib import Path
import unittest

import mlx.core as mx
import numpy as np

from rl.defense_ars_context import (approach_subspace, contrast_basis,
                                    visible_context, visible_subspace)
from rl.defense_ars_early_context import visible_early_subspace
from rl.model import QNetwork


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'results/defense/training/ars-59-head-search/run/generation-000000/model.safetensors'
ARCHIVE = ROOT/'results/defense/training/ars-score-gated-69/run/own-loss-states'
PILOT_CHECKPOINT = ROOT/'results/defense/training/ars-subspace-pilot-80/generation-000001/model.safetensors'
EARLY_ARCHIVE = ROOT/'results/defense/training/ars-early-source-89'


class VisibleContextTests(unittest.TestCase):
    def test_affine_contrast_centers_early_and_scales_later(self):
        early = np.array([[0, 0], [2, 0]], np.float32)
        middle = np.array([[3, 2], [3, 2]], np.float32)
        late = np.array([[3, 2], [3, 2]], np.float32)
        basis, stats = contrast_basis(early, middle, late)
        self.assertEqual(basis.shape, (3,))
        self.assertAlmostEqual(stats['earlier']['mean'], 0, places=6)
        self.assertAlmostEqual(stats['middle']['mean'], 1, places=6)
        self.assertAlmostEqual(stats['later']['mean'], 1, places=6)
        with self.assertRaises(ValueError):
            contrast_basis(early, early, early)

    def test_only_own_visible_training_screens_supply_basis(self):
        model = QNetwork(action_count=20)
        model.load_weights(str(CHECKPOINT))
        mx.eval(model.state)
        basis, record = visible_context(model, ARCHIVE,
            '125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171')
        self.assertEqual(basis.shape, (257,))
        self.assertEqual(record['diagnostics']['examples'], 48)
        self.assertLess(abs(record['diagnostics']['earlier']['mean']), 1e-5)
        self.assertAlmostEqual((record['diagnostics']['middle']['mean']+
                                record['diagnostics']['later']['mean'])/2, 1, places=5)
        self.assertFalse(record['native_snapshot_read'])
        with self.assertRaises(ValueError):
            visible_context(model, ARCHIVE, 'wrong-model')

    def test_approach_subspace_is_centered_and_reproducible(self):
        rng = np.random.default_rng(17)
        early = rng.normal(size=(20, 12)).astype(np.float32)
        drift = np.zeros((20, 12), np.float32)
        drift[:, 0] = 2
        drift[:, 1:6] = rng.normal(size=(20, 5))
        approach = np.stack((early+drift, early+1.5*drift, early+2*drift))
        basis, record = approach_subspace(early, approach, components=4)
        self.assertEqual(basis.shape, (5, 13))
        self.assertEqual(record['examples'], 20)
        np.testing.assert_allclose(basis, approach_subspace(early, approach, 4)[0])
        np.testing.assert_allclose((early@basis[:, :-1].T+basis[:, -1]).mean(axis=0),
                                   np.zeros(5), atol=1e-5)
        self.assertAlmostEqual(float(((approach[0]@basis[0, :-1]+basis[0, -1]).mean()+
                                      (approach[1]@basis[0, :-1]+basis[0, -1]).mean())/2),
                               1, places=5)
        with self.assertRaises(ValueError):
            approach_subspace(early, approach[:, :, :10], 4)

    def test_verified_visible_subspace_never_reads_native_snapshot(self):
        model = QNetwork(action_count=20)
        model.load_weights(str(CHECKPOINT))
        mx.eval(model.state)
        basis, record = visible_subspace(model, ARCHIVE,
            '125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171')
        self.assertEqual(basis.shape, (5, 257))
        self.assertEqual(record['diagnostics']['examples'], 48)
        self.assertFalse(record['native_snapshot_read'])
        self.assertTrue(record['frozen_encoder_match_source'])
        model.load_weights(str(PILOT_CHECKPOINT))
        mx.eval(model.state)
        continued, lineage = visible_subspace(model, ARCHIVE,
            '125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171')
        np.testing.assert_array_equal(continued, basis)
        self.assertTrue(lineage['frozen_encoder_match_source'])
        wide, wide_record = visible_subspace(model, ARCHIVE,
            '125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171',
            components=12)
        self.assertEqual(wide.shape, (13, 257))
        self.assertEqual(wide_record['subspace_components'], 12)
        self.assertFalse(wide_record['native_snapshot_read'])
        with self.assertRaises(ValueError):
            visible_subspace(model, ARCHIVE, 'wrong-model')

    def test_verified_early_own_screens_supply_earlier_proposals(self):
        model = QNetwork(action_count=20)
        model.load_weights(str(PILOT_CHECKPOINT))
        mx.eval(model.state)
        basis, record = visible_early_subspace(model, EARLY_ARCHIVE, components=12)
        self.assertEqual(basis.shape, (13, 257))
        self.assertEqual(record['diagnostics']['examples'], 48)
        self.assertFalse(record['native_snapshot_read'])
        self.assertFalse(record['action_target_read'])
        self.assertTrue(record['frozen_encoder_match_source'])
        self.assertAlmostEqual((record['diagnostics']['mean_contrast']['middle']['mean']+
                                record['diagnostics']['mean_contrast']['later']['mean'])/2,
                               1, places=5)


if __name__ == '__main__':
    unittest.main()
