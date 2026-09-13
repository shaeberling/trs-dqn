import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.env import SHAPE
from rl.evaluate import load_policy, summary
from rl.ppo import PPO, gae, update_selection, validation_protocol


class PPOTests(unittest.TestCase):
    def test_selection_updates_both_records_before_checkpoint_save(self):
        result = summary([{"score": 220, "level": 4, "terminated": True}])
        best, rank, improved_mean, improved_levels = update_selection(
            result, 5, 77.2, (0, 0, 1, 1, 70.0))
        self.assertEqual(best, 220)
        self.assertEqual(rank, (0, 1, 1, 1, 220.0))
        self.assertTrue(improved_mean)
        self.assertTrue(improved_levels)

    def test_selection_preserves_mean_when_only_level_rank_improves(self):
        result = summary([{"score": 140, "level": 3, "terminated": True},
                          {"score": 0, "level": 1, "terminated": True}])
        best, rank, improved_mean, improved_levels = update_selection(
            result, 5, 77.2, (0, 0, 0, 2, 77.2))
        self.assertEqual(best, 77.2)
        self.assertEqual(rank, (0, 0, 1, 1, 70.0))
        self.assertFalse(improved_mean)
        self.assertTrue(improved_levels)

    def test_selection_rejects_incomplete_validation(self):
        result = summary([{"score": 500, "level": 6, "terminated": False}])
        previous_rank = (0, 0, 1, 8, 71.95)
        self.assertEqual(update_selection(result, 5, 77.2, previous_rank),
                         (77.2, previous_rank, False, False))

    def test_validation_records_only_compare_matching_protocols(self):
        original = {"eval_games": 5, "tstates": 100000}
        self.assertEqual(validation_protocol(original),
                         validation_protocol({**original, "eval_seed": 10000,
                                              "eval_max_steps": 20000}))
        for changed in ({"eval_games": 20}, {"eval_seed": 11000},
                        {"eval_max_steps": 0}, {"tstates": 50000},
                        {"environment_version": "normalized-game-over-v2"}):
            self.assertNotEqual(validation_protocol(original),
                                validation_protocol({**original, **changed}))

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
