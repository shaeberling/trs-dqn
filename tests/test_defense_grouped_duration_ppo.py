import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_balanced_duration import (group_duration_logits_mlx,
                                          group_duration_logits_numpy,
                                          grouped_duration_log_probs_mlx,
                                          grouped_duration_log_probs_numpy)
from rl.defense_grouped_duration_ppo import GroupedDurationPPO
from rl.defense_learning import load_policy


class GroupedDurationPPOTests(unittest.TestCase):
    def test_grouped_option_likelihood_matches_mlx_and_preserves_key_marginal(self):
        import mlx.core as mx

        raw = np.random.default_rng(870).normal(size=(3, 80)).astype(np.float32)
        grouped = group_duration_logits_numpy(raw, 4)
        np.testing.assert_allclose(np.array(group_duration_logits_mlx(mx.array(raw), 4)),
                                   grouped, atol=1e-6)
        original = np.exp(grouped_duration_log_probs_numpy(grouped, 4, 0)).reshape(3, 4, 12)
        for mix in (0., .05, .5):
            expected = grouped_duration_log_probs_numpy(grouped, 4, mix)
            actual = np.array(grouped_duration_log_probs_mlx(mx.array(grouped), 4, mix))
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            probs = np.exp(expected).reshape(3, 4, 12)
            np.testing.assert_allclose(probs.sum(axis=1), original.sum(axis=1), atol=1e-6)

    def test_actor_and_loss_use_same_grouped_mixture(self):
        import mlx.core as mx

        agent = GroupedDurationPPO(seed=871, action_count=80, durations=(1, 4, 16, 64),
                                   entropy=.01, duration_explore_mix=.05)
        observations = np.zeros((3, 4, 16, 64), np.uint8)
        mask = np.array([True, False, True])
        actions, old_logp, values = agent.act(observations, np.random.default_rng(872),
                                               actor_mask=mask)
        raw = np.array(agent.predict(mx.array(observations))[0])
        grouped = group_duration_logits_numpy(raw, 4)
        expected = grouped_duration_log_probs_numpy(grouped, 4, .05)
        np.testing.assert_allclose(old_logp[mask],
                                   expected[np.flatnonzero(mask), actions[mask]], atol=1e-6)
        self.assertEqual(int(actions[1]), 0)
        self.assertEqual(float(old_logp[1]), 0.)
        loss, metrics = agent._loss(agent.model, mx.array(observations), mx.array(actions),
                                    mx.array(old_logp), mx.ones(3), mx.array(values),
                                    mx.array(mask.astype(np.float32)))
        mx.eval(loss, metrics)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertLess(abs(float(metrics[3].item())), 1e-5)
        loss, metrics = agent.update(mx.array(observations), mx.array(actions), mx.array(old_logp),
                                     mx.ones(3), mx.array(values),
                                     mx.array(mask.astype(np.float32)))
        mx.eval(loss, metrics, agent.state)
        self.assertTrue(np.isfinite(float(loss.item())))

    def test_missing_weights_rejected_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "invalid"
            result = subprocess.run([sys.executable, "-m", "rl.defense_train", "--run", str(run),
                                     "--learned-durations", "1", "4", "--grouped-duration",
                                     "--life-terminal"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(run.exists())

    def test_native_train_evaluate_replay_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            command = [sys.executable, "-u", "-m", "rl.defense_train", "--run", str(run),
                       "--artifacts", str(root / "artifacts"), "--learned-durations",
                       "1", "4", "16", "64", "--grouped-duration",
                       "--grouped-duration-weights", ".88", ".08", ".03", ".01",
                       "--duration-explore-mix", ".05", "--option-actor-gae",
                       "--life-terminal", "--steps", "32", "--seed", "43", "--envs", "2",
                       "--rollout", "16", "--batch-size", "32", "--epochs", "1",
                       "--eval-every", "32", "--eval-games", "2", "--eval-envs", "2",
                       "--mlx-cache-mb", "64"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = run / "step-000000000032"
            state = json.loads((checkpoint / "state.json").read_text())
            self.assertTrue(state["config"]["grouped_duration"])
            self.assertEqual(state["config"]["reward"],
                             "visible score difference only, constant scale for optimizer")
            evaluation = json.loads((checkpoint / "evaluation.json").read_text())
            self.assertEqual(evaluation["complete_games"], 2)
            stopped = json.loads((run / "status.json").read_text())
            self.assertEqual(stopped["event"], "stopped")
            self.assertEqual(stopped["duration_options"]["executed_base_actions"], 32)
            policy, _ = load_policy(checkpoint / "model.safetensors")
            sampled = policy.sample_with_rngs(np.zeros((2, 4, 16, 64), np.uint8),
                                               [np.random.default_rng(1), np.random.default_rng(2)])
            self.assertTrue(set(sampled).issubset(set((*range(10), 18, 19))))
            verification = json.loads((root / "artifacts/best/verification.json").read_text())
            self.assertTrue(verification["verified"])
            resumed = subprocess.run([sys.executable, "-u", "-m", "rl.defense_train", "--run",
                                      str(root / "resume"), "--artifacts", str(root / "resumed-artifacts"),
                                      "--resume", str(checkpoint), "--steps", "64"],
                                     capture_output=True, text=True, timeout=180)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            later = json.loads((root / "resume/latest/state.json").read_text())
            self.assertEqual(later["steps"], 64)
            self.assertTrue(later["config"]["grouped_duration"])


if __name__ == "__main__":
    unittest.main()
