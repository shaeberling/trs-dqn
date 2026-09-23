from pathlib import Path
import unittest

import mlx.core as mx
import numpy as np

from rl.defense_ars_context import contrast_basis, visible_context
from rl.model import QNetwork


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'results/defense/training/ars-59-head-search/run/generation-000000/model.safetensors'
ARCHIVE = ROOT/'results/defense/training/ars-score-gated-69/run/own-loss-states'


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


if __name__ == '__main__':
    unittest.main()
