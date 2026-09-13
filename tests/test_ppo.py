import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.env import SHAPE
from rl.evaluate import load_policy
from rl.ppo import PPO, gae


class PPOTests(unittest.TestCase):
    def test_gae_does_not_cross_episode_boundaries(self):
        rewards = np.array([[1], [2], [3]], np.float32)
        values = np.full((3, 1), .5, np.float32)
        boundaries = np.array([[0], [1], [0]], np.float32)
        advantages, returns = gae(rewards, values, boundaries, np.array([4], np.float32), .9, 1)
        np.testing.assert_allclose(advantages[:, 0], [2.3, 1.5, 6.1], atol=1e-6)
        np.testing.assert_allclose(returns[:, 0], [2.8, 2, 6.6], atol=1e-6)

    def test_categorical_checkpoint_reproduces_seeded_policy(self):
        agent = PPO()
        obs = np.full((16, *SHAPE), 128, np.uint8)
        actions, logp, values = agent.act(obs, np.random.default_rng(21))
        loss, metrics = agent.update(mx.array(obs), mx.array(actions), mx.array(logp),
                                     mx.ones(16), mx.array(values+1))
        mx.eval(loss, metrics, agent.state)
        self.assertTrue(np.isfinite(loss.item()))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            agent.save(directory, {"config": {"algorithm": "ppo"}})
            restored = load_policy(directory/"model.safetensors")
            original = agent.policy()
            restored.reset_seed(24)
            original.reset_seed(24)
            for _ in range(5):
                np.testing.assert_array_equal(original(obs), restored(obs))


if __name__ == "__main__":
    unittest.main()
