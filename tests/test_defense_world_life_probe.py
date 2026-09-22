import unittest
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_life_probe import Readout, extract, features, metrics, sample, windows
from rl.defense_world_model import WorldModel


class LifeProbeTests(unittest.TestCase):
    def test_window_coverage_is_unique_and_includes_tail(self):
        for count in range(32, 151):
            indices = [index for start, first in windows(count)
                       for index in range(start+first, start+32)]
            self.assertEqual(indices, list(range(8, count)))
        with self.assertRaises(ValueError):
            list(windows(31))

    def test_metrics_ties_class_direction_and_perfect_prediction(self):
        labels = np.array([1., 1., 1., 0.])
        constant = metrics(np.zeros(4), labels)
        self.assertEqual(constant['loss_average_precision'], .25)
        self.assertEqual(constant['brier'], .25)
        self.assertEqual(constant['threshold_half']['true_positive'], 1)
        perfect = metrics(np.array([50., 50., 50., -50.]), labels)
        self.assertEqual(perfect['loss_average_precision'], 1.)
        self.assertLess(perfect['brier'], 1e-20)
        self.assertEqual(perfect['threshold_half'], dict(true_positive=1, false_positive=0, false_negative=0))
        # Two tied highest scores contain one of the two positives.
        tied = metrics(np.array([-2., -2., 0., 1.]), np.array([0., 1., 0., 1.]))
        self.assertAlmostEqual(tied['loss_average_precision'], (1/2+2/3)/2)

    def test_stratified_weights_preserve_empirical_objective(self):
        labels = np.array([0., 1., 1., 1.], np.float32)
        chosen, weights = sample(labels, np.random.default_rng(1), 100, True)
        self.assertEqual(int((labels[chosen] == 0).sum()), 50)
        np.testing.assert_array_equal(weights[:50], .5)
        np.testing.assert_array_equal(weights[50:], 1.5)
        # Any class-constant loss has exactly the same weighted expectation.
        losses = np.where(labels == 0, 7., 2.)
        self.assertEqual(float(np.mean(losses[chosen]*weights)), float(losses.mean()))
        _, uniform = sample(labels, np.random.default_rng(1), 100, False)
        np.testing.assert_array_equal(uniform, 1.)
        with self.assertRaises(ValueError):
            sample(np.ones(4), np.random.default_rng(0), 4, True)

    def test_prior_feature_never_observes_its_arrival(self):
        mx.random.seed(9)
        model = WorldModel()
        rng = np.random.default_rng(7)
        frames = rng.integers(128, 192, (1, 6, 16, 64), dtype=np.uint8)
        actions = mx.array([[1, 2, 3, 4, 5]])
        original = [np.array(v) for v in features(model, mx.array(frames), actions, 1)]
        frames[:, 3:] = 32
        changed = [np.array(v) for v in features(model, mx.array(frames), actions, 1)]
        np.testing.assert_array_equal(original[1][:, :2], changed[1][:, :2])
        self.assertFalse(np.array_equal(original[0][:, 1], changed[0][:, 1]))

    def test_extraction_alignment_and_readout_does_not_modify_parent(self):
        mx.random.seed(6)
        model = WorldModel()
        rng = np.random.default_rng(5)
        frames = rng.integers(128, 192, (41, 16, 64), dtype=np.uint8)
        actions = rng.integers(20, size=40, dtype=np.int32)
        continuation = np.ones(40, np.float32)
        continuation[[12, 39]] = 0
        dataset = SimpleNamespace(length=32, episodes=[(frames, actions, np.zeros(40), continuation)])
        rows = extract(model, dataset, batch=2)
        self.assertEqual(rows['observed'].shape, (32, 160))
        np.testing.assert_array_equal(rows['action'], np.arange(8, 40))
        np.testing.assert_array_equal(rows['continuation'], continuation[8:])
        original = {k: np.array(v).copy() for k, v in tree_flatten(model.parameters())}
        first, second = Readout(model.continue_logit), Readout(model.continue_logit)
        for (k, a), (l, b) in zip(tree_flatten(first.head.parameters()), tree_flatten(second.head.parameters()), strict=True):
            self.assertEqual(k, l)
            np.testing.assert_array_equal(np.array(a), np.array(b))
        before = np.array(first.head(mx.array(rows['observed'])))
        for _ in range(3):
            stats = first.train(rows['observed'], rows['continuation'], np.ones(32, np.float32))
            self.assertTrue(np.isfinite(list(stats.values())).all())
        for key, value in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(original[key], np.array(value))
        np.testing.assert_array_equal(before, np.array(second.head(mx.array(rows['observed']))))
        self.assertFalse(np.array_equal(before, np.array(first.head(mx.array(rows['observed'])))))


if __name__ == '__main__':
    unittest.main()
