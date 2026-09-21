import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from rl.env import SHAPE
from rl.evaluate import load_policy, summary
from rl.ppo import PPO, gae, update_selection, validation_protocol


class PPOTests(unittest.TestCase):
    def test_evaluation_only_resume_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/"average", root/"must-not-be-created"
            source.mkdir()
            for flag in ({"evaluation_only": True}, {"resume_supported": False}):
                (source/"state.json").write_text(json.dumps(flag))
                result = subprocess.run(
                    [sys.executable, "-m", "rl.ppo", "--run", str(output),
                     "--resume", str(source)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("evaluation-only checkpoint", result.stderr)
                self.assertFalse(output.exists())

    def test_invalid_reserved_boot_cli_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"must-not-be-created"
            for count in (-1, 2, 1):
                # Two workers: -1/2 are out of range; 1 requires shared curriculum.
                result = subprocess.run(
                    [sys.executable, "-m", "rl.ppo", "--run", str(directory),
                     "--envs", "2", "--curriculum-boot-envs", str(count)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("curriculum-boot-envs", result.stderr)
                self.assertFalse(directory.exists())

    def test_importing_entry_module_does_not_load_gpu_backend(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import rl.ppo; "
             "assert 'mlx.core' not in sys.modules; assert 'rl.model' not in sys.modules"],
            check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.stderr, "")

    def test_optimizer_resume_matches_uninterrupted_updates(self):
        learning_rate = 0.00005
        agent = PPO(seed=17, learning_rate=learning_rate, entropy=0.003)
        rng = np.random.default_rng(42)
        obs = rng.integers(128, 192, size=(8, *SHAPE), dtype=np.uint8)
        actions, logp, values = agent.act(obs, rng)
        batch = tuple(mx.array(x) for x in
                      (obs, actions, logp, np.linspace(-1, 1, 8, dtype=np.float32),
                       values + np.linspace(0.2, 2, 8, dtype=np.float32)))
        for _ in range(3):
            loss, metrics = agent.update(*batch)
            mx.eval(loss, metrics, agent.state)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            agent.save(directory, {"config": {"algorithm": "ppo"}})
            restored = PPO(seed=99, learning_rate=0.01, entropy=0.003)
            # Mirror the trainer's resume path, including replacement of Adam
            # state, learning-rate override, and recompilation after loading.
            restored.model.load_weights(str(directory/"model.safetensors"))
            restored.optimizer.state = tree_unflatten(list(
                mx.load(str(directory/"optimizer.npz")).items()))
            restored.optimizer.learning_rate = learning_rate
            restored.compile()
            for _ in range(3):
                for learner in (agent, restored):
                    loss, metrics = learner.update(*batch)
                    mx.eval(loss, metrics, learner.state)

            for original, resumed in ((agent.model.parameters(), restored.model.parameters()),
                                      (agent.optimizer.state, restored.optimizer.state)):
                expected, actual = dict(tree_flatten(original)), dict(tree_flatten(resumed))
                self.assertEqual(expected.keys(), actual.keys())
                for name in expected:
                    np.testing.assert_allclose(np.array(actual[name]), np.array(expected[name]),
                                               rtol=1e-6, atol=1e-7, err_msg=name)

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
                                              "eval_max_steps": 20000, "eval_envs": 1}))
        for changed in ({"eval_games": 20}, {"eval_seed": 11000},
                        {"eval_max_steps": 0}, {"tstates": 50000},
                        {"environment_version": "normalized-game-over-v2"}, {"eval_envs": 8},
                        {"observation_stride": 2}):
            self.assertNotEqual(validation_protocol(original),
                                validation_protocol({**original, **changed}))

    def test_gae_does_not_cross_episode_boundaries(self):
        rewards = np.array([[1], [2], [3]], np.float32)
        values = np.full((3, 1), .5, np.float32)
        boundaries = np.array([[0], [1], [0]], np.float32)
        advantages, returns = gae(rewards, values, boundaries, np.array([4], np.float32), .9, 1)
        np.testing.assert_allclose(advantages[:, 0], [2.3, 1.5, 6.1], atol=1e-6)
        np.testing.assert_allclose(returns[:, 0], [2.8, 2, 6.6], atol=1e-6)

    def test_gae_trace_decay_matches_analytic_delayed_reward(self):
        rewards = np.zeros((65, 1), np.float32)
        rewards[-1, 0] = 1
        values = np.zeros_like(rewards)
        boundaries = np.zeros_like(rewards)
        boundaries[-1] = 1
        for lam in (.95, .99):
            advantages, returns = gae(rewards, values, boundaries,
                                      np.zeros(1, np.float32), .995, lam)
            expected = (.995*lam)**np.arange(64, -1, -1)
            np.testing.assert_allclose(advantages[:, 0], expected, rtol=1e-5)
            np.testing.assert_array_equal(advantages, returns)
        boundaries[32] = 1
        advantages, _ = gae(rewards, values, boundaries,
                            np.zeros(1, np.float32), .995, .99)
        np.testing.assert_array_equal(advantages[:33], np.zeros((33, 1)))

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
            # The shared evaluator refactor must also match the trainer's
            # unchanged categorical sampler, not only the checkpoint loader.
            original.reset_seed(31)
            action_rng = np.random.default_rng(31)
            for _ in range(5):
                np.testing.assert_array_equal(original(obs), agent.act(obs, action_rng)[0])


if __name__ == "__main__":
    unittest.main()
