"""Grouped PPO likelihoods, fixed action mapping and native checkpoint replay."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense_canonical_fire import (COMMAND_MAP, balanced_fire_initial_bias,
                                       group_logits_numpy)


class CanonicalFireTests(unittest.TestCase):
    def test_balanced_fresh_fire_bias_is_direction_neutral(self):
        offset = balanced_fire_initial_bias()
        self.assertEqual(offset.shape, (20,))
        self.assertEqual(offset.dtype, np.float32)
        self.assertTrue(np.all(offset[:9] == 0))
        self.assertTrue(np.all(offset[18:] == 0))
        np.testing.assert_allclose(offset[9:18], -np.log(9), atol=1e-6)
        np.testing.assert_allclose(group_logits_numpy(offset[None]),
                                   np.zeros((1, 12), np.float32), atol=1e-6)

    def test_grouped_probability_is_sum_of_fire_aliases(self):
        import mlx.core as mx
        from rl.defense_canonical_fire import group_logits_mlx, learning_indices

        logits = np.arange(60, dtype=np.float32).reshape(3, 20)/8
        grouped = group_logits_numpy(logits)
        np.testing.assert_allclose(np.array(group_logits_mlx(mx.array(logits))), grouped, atol=1e-6)
        original = np.exp(logits-np.logaddexp.reduce(logits, axis=1, keepdims=True))
        probabilities = np.exp(grouped-np.logaddexp.reduce(grouped, axis=1, keepdims=True))
        np.testing.assert_allclose(probabilities[:, :9], original[:, :9], atol=1e-6)
        np.testing.assert_allclose(probabilities[:, 9], original[:, 9:18].sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(probabilities[:, 10:], original[:, 18:], atol=1e-6)
        np.testing.assert_array_equal(np.array(learning_indices(mx.array(COMMAND_MAP))), np.arange(12))

    def test_actor_and_loss_use_same_grouped_likelihood(self):
        import mlx.core as mx
        from rl.ppo import PPO

        agent = PPO(seed=31, action_count=20, canonical_fire=True)
        observations = np.zeros((4, 4, 16, 64), np.uint8)
        actions, old_logp, values = agent.act(observations, np.random.default_rng(17))
        self.assertTrue(set(actions).issubset(set(COMMAND_MAP)))
        logits = np.array(agent.predict(mx.array(observations))[0])
        grouped = group_logits_numpy(logits)
        logp = grouped-np.logaddexp.reduce(grouped, axis=1, keepdims=True)
        indices = np.array([np.where(COMMAND_MAP == action)[0][0] for action in actions])
        np.testing.assert_allclose(old_logp, logp[np.arange(4), indices], atol=1e-5)
        loss, metrics = agent._loss(agent.model, mx.array(observations), mx.array(actions),
                                    mx.array(old_logp), mx.array(np.ones(4, np.float32)),
                                    mx.array(values))
        mx.eval(loss, metrics)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertAlmostEqual(float(metrics[3].item()), 0., places=5)

    def test_invalid_combinations_rejected_before_emulator_launch(self):
        from rl import defense_train

        for extra in (["--allow-enter"], ["--sil-updates", "1"], ["--policy-bias-noise", ".5"],
                      ["--policy-weight-noise", ".005"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", [
                    "defense_train", "--run", "/nonexistent/canonical-fire-test",
                    "--canonical-fire", *extra]), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_train.main()
            self.assertEqual(error.exception.code, 2)

    def test_balanced_initializer_changes_only_fire_alias_biases(self):
        import mlx.core as mx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [sys.executable, '-m', 'rl.defense_train', '--canonical-fire',
                    '--initialize-only', '--envs', '2', '--rollout', '16',
                    '--batch-size', '32', '--eval-every', '16', '--eval-games', '2',
                    '--eval-envs', '2', '--mlx-cache-mb', '64']
            for name, extra in (('control', []), ('balanced', ['--balanced-canonical-init'])):
                command = [*base, '--run', str(root/name), '--artifacts',
                           str(root/(name+'-artifacts')), *extra]
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                config = json.loads((root/name/'latest/state.json').read_text())['config']
                self.assertEqual(config['balanced_canonical_init'], name == 'balanced')
                self.assertEqual(config['reward'],
                                 'visible score difference only, constant scale for optimizer')
            control = mx.load(str(root/'control/latest/model.safetensors'))
            balanced = mx.load(str(root/'balanced/latest/model.safetensors'))
            self.assertEqual(control.keys(), balanced.keys())
            for key in control:
                difference = np.array(balanced[key] - control[key])
                np.testing.assert_allclose(
                    difference, balanced_fire_initial_bias() if key == 'advantage.bias'
                    else np.zeros_like(difference), atol=1e-6, err_msg=key)

    def test_native_training_evaluation_replay_and_resume(self):
        checkpoint = Path("results/defense/training/ppo-own-loss-107/checkpoint-000008407808")
        if not (checkpoint/"optimizer.npz").exists():
            self.skipTest("archived own PPO optimizer unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = [sys.executable, "-m", "rl.defense_train", "--run", str(root/"run"),
                     "--artifacts", str(root/"artifacts"), "--resume", str(checkpoint),
                     "--steps", "8411904", "--canonical-fire", "--envs", "2",
                     "--rollout", "16", "--batch-size", "32", "--epochs", "1",
                     "--curriculum-boot-envs", "1", "--eval-every", "4096",
                     "--eval-games", "2", "--eval-envs", "2", "--mlx-cache-mb", "64"]
            result = subprocess.run(first, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            saved = root/"run/step-000008411904"
            state = json.loads((saved/"state.json").read_text())
            self.assertTrue(state["config"]["canonical_fire"])
            self.assertIn("canonical_fire_source_sha256", state["config"])
            evaluation = json.loads((saved/"evaluation.json").read_text())
            self.assertEqual(evaluation["complete_games"], 2)
            from rl.defense_learning import load_policy
            policy, config = load_policy(saved/"model.safetensors")
            self.assertTrue(config["canonical_fire"])
            sampled = policy.sample_with_rngs(np.zeros((2, 4, 16, 64), np.uint8),
                                               [np.random.default_rng(1), np.random.default_rng(2)])
            self.assertTrue(set(sampled).issubset(set(COMMAND_MAP)))
            manifest = json.loads((root/"artifacts/best/manifest.json").read_text())
            self.assertEqual(manifest["rank"][1], 1)
            verification = json.loads((root/"artifacts/best/verification.json").read_text())
            self.assertTrue(verification["verified"])
            resumed = subprocess.run([sys.executable, "-m", "rl.defense_train", "--run",
                                      str(root/"resume"), "--artifacts", str(root/"resume-artifacts"),
                                      "--resume", str(saved), "--steps", "8411936"],
                                     capture_output=True, text=True, timeout=180)
            self.assertEqual(resumed.returncode, 0, resumed.stdout+resumed.stderr)
            after = json.loads((root/"resume/latest/state.json").read_text())
            self.assertEqual(after["steps"], 8411936)
            self.assertTrue(after["config"]["canonical_fire"])


if __name__ == "__main__":
    unittest.main()
