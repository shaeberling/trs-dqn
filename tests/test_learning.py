"""Numerical regression checks; requires an accessible Metal GPU."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.env import SHAPE
from rl.model import Learner, QNetwork, render_table


class LearningTests(unittest.TestCase):
    def test_graphics_encoding_matches_trs(self):
        table = render_table()
        for code in range(128, 192):
            np.testing.assert_array_equal(table[code, :, :, 0].reshape(-1),
                                          [(code >> i) & 1 for i in range(6)])
        self.assertEqual(table[ord('A'), 0, 0, 1], np.float32(ord('A')/127))

    def test_gradient_target_freezing_and_checkpoint_roundtrip(self):
        learner = Learner(1e-3, 42)
        rng = np.random.default_rng(42)
        obs = rng.integers(128, 192, size=(8, *SHAPE), dtype=np.uint8)
        learner.actions(obs)  # Compile inference before any weight updates.
        target_before = {k: np.array(v) for k, v in tree_flatten(learner.target.parameters())}
        batch = (obs, np.arange(8, dtype=np.int32) % 6, np.ones(8, np.float32),
                 obs, np.zeros(8, np.float32), np.ones(8, np.float32))
        first = learner.train(batch)[0]
        for _ in range(30):
            final = learner.train(batch)[0]
        self.assertLess(final, first*0.2)
        np.testing.assert_array_equal(learner.actions(obs),
                                      np.array(mx.argmax(learner.online(mx.array(obs)), axis=1)))
        for key, value in tree_flatten(learner.target.parameters()):
            np.testing.assert_array_equal(value, target_before[key])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.safetensors"
            learner.online.save_weights(str(path))
            restored = QNetwork()
            restored.load_weights(str(path))
            np.testing.assert_allclose(np.array(restored(mx.array(obs))),
                                       np.array(learner.online(mx.array(obs))), atol=1e-6)
        learner.sync_target()
        np.testing.assert_allclose(np.array(learner.online(mx.array(obs))),
                                   np.array(learner.target(mx.array(obs))), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
