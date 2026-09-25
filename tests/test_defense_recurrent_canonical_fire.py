"""Grouped physical controls remain identical in recurrent training and replay."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.defense import action_names
from rl.defense_canonical_fire import COMMAND_MAP, balanced_fire_initial_bias, group_logits_numpy
from rl.defense_learning import load_policy, policy_description, record_game, verify_policy_trace
from rl.defense_recurrent import RecurrentPPO
from rl.recurrent_policy import (CanonicalRecurrentPolicy,
                                 OwnActionCanonicalRecurrentPolicy)


class RecurrentCanonicalFireTests(unittest.TestCase):
    def test_grouped_recurrent_policy_tracks_physical_key_and_game_memory(self):
        seen = []

        def infer(obs, hidden, previous):
            seen.extend(np.asarray(previous).tolist())
            logits = np.zeros((len(obs), 20), np.float32)
            logits[:, 9:18] = 4
            return logits, hidden+1

        policy = OwnActionCanonicalRecurrentPolicy(infer, 3)
        rngs = [np.random.default_rng(seed) for seed in (1, 2)]
        first = policy.sample_with_rngs(np.zeros((2, 1)), rngs)
        self.assertEqual(seen, [20, 20])
        self.assertTrue(set(first).issubset(set(COMMAND_MAP)))
        seen.clear()
        policy.sample_with_rngs(np.zeros((2, 1)), rngs[::-1])
        self.assertEqual(seen, [int(first[1]), int(first[0])])
        with self.assertRaises(ValueError):
            OwnActionCanonicalRecurrentPolicy(infer, 3, action_count=12)
        plain = CanonicalRecurrentPolicy(lambda obs, hidden: (
            np.zeros((len(obs), 20), np.float32), hidden+1), 3)
        self.assertTrue(set(plain.sample_with_rngs(np.zeros((2, 1)), rngs)).issubset(set(COMMAND_MAP)))

    def test_agent_action_and_loss_use_same_grouped_likelihood(self):
        agent = RecurrentPPO(seed=17, action_count=20, hidden_size=8,
                             own_action_input=True, canonical_fire=True)
        agent.reset_memory(2)
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        logits, values, _ = agent.predict(mx.array(obs), mx.zeros((2, 8)),
                                          mx.array([20, 20]))
        grouped = group_logits_numpy(np.array(logits))
        logp = grouped-np.logaddexp.reduce(grouped, axis=1, keepdims=True)
        actions, chosen_logp, actual_values = agent.act(obs, np.random.default_rng(17))
        self.assertTrue(set(actions).issubset(set(COMMAND_MAP)))
        choices = np.array([np.flatnonzero(COMMAND_MAP == action)[0] for action in actions])
        np.testing.assert_allclose(chosen_logp, logp[np.arange(2), choices], atol=1e-5)
        np.testing.assert_allclose(actual_values, np.array(values), atol=1e-5)
        np.testing.assert_array_equal(agent.previous_actions, actions)

        sequence = np.tile(obs[:, None], (1, 4, 1, 1, 1))
        starts = np.zeros((2, 4), bool)
        starts[:, 0] = True
        previous = np.full((2, 4), 20, np.int32)
        previous[:, 1:] = actions[:, None]
        raw, targets, _ = agent.model.sequence(mx.array(sequence), mx.zeros((2, 8)),
                                               mx.array(starts), mx.array(previous))
        grouped_seq = group_logits_numpy(np.array(raw).reshape(-1, 20)).reshape(2, 4, 12)
        logp_seq = grouped_seq-np.logaddexp.reduce(grouped_seq, axis=2, keepdims=True)
        old_logp = np.take_along_axis(logp_seq, choices[:, None, None].repeat(4, axis=1),
                                      axis=2)[..., 0]
        action_seq = np.tile(actions[:, None], (1, 4))
        loss, metrics = agent._loss(agent.model, mx.array(sequence), mx.array(action_seq),
                                    mx.array(old_logp), mx.ones((2, 4)), targets,
                                    mx.zeros((2, 8)), mx.array(starts), mx.array(previous))
        mx.eval(loss, metrics)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertAlmostEqual(float(metrics[3].item()), 0., places=5)

    def test_native_fresh_initialization_training_reload_replay_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [sys.executable, "-m", "rl.defense_train", "--canonical-fire",
                    "--balanced-canonical-init", "--recurrent-hidden", "8",
                    "--recurrent-own-action", "--sequence-length", "4", "--life-terminal",
                    "--envs", "2", "--rollout", "8", "--batch-size", "16",
                    "--epochs", "1", "--max-episode-steps", "3", "--eval-every", "32",
                    "--eval-games", "2", "--eval-envs", "2", "--mlx-cache-mb", "128"]

            def run(name, extra):
                completed = subprocess.run(base+["--run", str(root/name), "--artifacts",
                    str(root/(name+"-artifacts")), *extra], capture_output=True,
                    text=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
                return json.loads((root/name/"latest/state.json").read_text())

            initial = run("initial", ["--initialize-only", "--seed", "45"])
            control = run("control", ["--initialize-only", "--seed", "45", "--memory-scale", "0"])
            self.assertEqual(initial["config"]["policy"],
                             "learned recurrent categorical over fixed twelve command groups, sampled")
            left = mx.load(str(root/"initial/latest/model.safetensors"))
            right = mx.load(str(root/"control/latest/model.safetensors"))
            for key in left:
                np.testing.assert_array_equal(np.array(left[key]), np.array(right[key]))
            bias = np.array(left["base.advantage.bias"])
            np.testing.assert_allclose(group_logits_numpy(bias[None]),
                                       np.zeros((1, 12), np.float32), atol=.05)
            np.testing.assert_allclose(bias[9:18]-bias[:9].mean(),
                                       balanced_fire_initial_bias()[9:18], atol=.05)

            trained = run("trained", ["--seed", "45", "--steps", "32"])
            self.assertEqual(trained["steps"], 32)
            self.assertEqual(trained["config"]["reward"],
                             "visible score difference only, constant scale for optimizer")
            checkpoint = root/"trained/latest/model.safetensors"
            policy, config = load_policy(checkpoint)
            self.assertIn("fixed twelve-group", policy_description(config))
            self.assertIsInstance(policy, OwnActionCanonicalRecurrentPolicy)
            sample = policy.sample_with_rngs(np.zeros((2, 4, 16, 64), np.uint8),
                                             [np.random.default_rng(1), np.random.default_rng(2)])
            self.assertTrue(set(sample).issubset(set(COMMAND_MAP)))
            trace = record_game(policy, 10000, tstates=100000, max_steps=0)
            verification = verify_policy_trace(checkpoint, *trace[:4])
            self.assertTrue(verification["verified"])
            self.assertEqual(verification["verified_actions"], len(trace[1]))
            resumed = run("resumed", ["--resume", str(root/"trained/latest"), "--steps", "64"])
            self.assertEqual(resumed["steps"], 64)
            self.assertTrue(resumed["config"]["canonical_fire"])
            self.assertTrue(resumed["config"]["recurrent_own_action"])
            incompatible = subprocess.run(base+["--run", str(root/"incompatible"),
                "--artifacts", str(root/"incompatible-artifacts"), "--resume",
                str(root/"trained/latest"), "--no-canonical-fire", "--steps", "64"],
                capture_output=True, text=True, timeout=120)
            self.assertNotEqual(incompatible.returncode, 0)
            self.assertIn("Changing grouped-command profile", incompatible.stderr)
            self.assertFalse((root/"incompatible").exists())


if __name__ == "__main__":
    unittest.main()
