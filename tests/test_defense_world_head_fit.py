from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_head_fit import HeadLearner, immutable_arrays
from rl.defense_world_model import WorldLearner


class HeadFitTests(unittest.TestCase):
    def test_partial_update_preserves_other_moments_and_exact_resume(self):
        base = WorldLearner(seed=5, burn=1)
        rng = np.random.default_rng(5)
        # Populate all Adam moments with a real full-model update first.
        frames = rng.integers(128, 192, (1, 5, 16, 64), dtype=np.uint8)
        base.train((frames, np.array([[1, 2, 3, 4]], np.int32),
                    np.array([[0., .1, 0., .2]], np.float32), np.array([[1., 1., 1., 0.]], np.float32)))
        before = immutable_arrays(base)
        head_before = {k: np.array(v).copy() for k, v in tree_flatten(base.model.continue_logit.parameters())}
        learner = HeadLearner(base)
        x = rng.normal(size=(12, 160)).astype(np.float32)
        y = np.array([0., 1.]*6, np.float32)
        weights = np.ones(12, np.float32)
        learner.train(x, y, weights)
        self.assertEqual(base.updates, 2)
        self.assertEqual(int(base.optimizer.step), 2)
        for key, value in immutable_arrays(base).items():
            np.testing.assert_array_equal(value, before[key])
        self.assertTrue(any(not np.array_equal(head_before[k], np.array(v))
                            for k, v in tree_flatten(base.model.continue_logit.parameters())))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'checkpoint'
            base.save(directory, rng, dict(training_phase='head-only test'))
            restored = WorldLearner(seed=23, burn=1)
            restored_rng = np.random.default_rng(99)
            restored.restore(directory, restored_rng)
            other = HeadLearner(restored)
            self.assertEqual(learner.train(x, y, weights), other.train(x, y, weights))
            for tree, new in ((base.model.parameters(), restored.model.parameters()),
                              (base.optimizer.state, restored.optimizer.state)):
                a, b = dict(tree_flatten(tree)), dict(tree_flatten(new))
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            np.testing.assert_array_equal(rng.integers(1000, size=20), restored_rng.integers(1000, size=20))


if __name__ == '__main__':
    unittest.main()
