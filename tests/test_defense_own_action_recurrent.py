import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_learning import load_policy, policy_description, record_game, verify_policy_trace
from rl.defense_recurrent import RecurrentPPO, ResidualRecurrentNetwork
from rl.ppo import PPO
from rl.recurrent_policy import (OWN_ACTION_RECURRENT_ARCHITECTURE,
                                 OwnActionRecurrentPolicy)


class OwnActionRecurrentTests(unittest.TestCase):
    def test_sequence_matches_step_and_zero_residual_preserves_base(self):
        model = ResidualRecurrentNetwork(action_count=20, hidden_size=8,
                                         own_action_input=True)
        obs = np.random.default_rng(10).integers(128, 192, (2, 4, 4, 16, 64), dtype=np.uint8)
        actions = mx.array([[20, 4, 4, 7], [20, 1, 2, 20]])
        initial = mx.ones((2, 8))
        starts = mx.array([[True, False, False, False],
                           [True, False, False, True]])
        flattened = mx.array(obs.reshape(-1, 4, 16, 64))
        base = model.base.policy_value(flattened)
        current = model.step(flattened, mx.ones((8, 8)),
                             mx.array([20, 4, 4, 7, 20, 1, 2, 20]))
        for expected, actual in zip(base, current[:2], strict=True):
            np.testing.assert_array_equal(np.array(expected), np.array(actual))
        model.memory_actor.weight = mx.ones_like(model.memory_actor.weight)*.1
        model.memory_value.weight = mx.ones_like(model.memory_value.weight)*.2
        logits, values, final = model.sequence(mx.array(obs), initial, starts, actions)
        hidden = initial
        sequential_logits, sequential_values = [], []
        for index in range(4):
            hidden = hidden*(1-starts[:, index, None].astype(hidden.dtype))
            one_logits, one_value, hidden = model.step(
                mx.array(obs[:, index]), hidden, actions[:, index])
            sequential_logits.append(one_logits)
            sequential_values.append(one_value)
        np.testing.assert_allclose(np.array(logits),
                                   np.array(mx.stack(sequential_logits, axis=1)), atol=2e-6)
        np.testing.assert_allclose(np.array(values),
                                   np.array(mx.stack(sequential_values, axis=1)), atol=2e-6)
        np.testing.assert_allclose(np.array(final), np.array(hidden), atol=2e-6)
        with self.assertRaises(ValueError):
            model.step(flattened, mx.ones((8, 8)))

    def test_policy_tracks_own_action_by_game_identity(self):
        observed = []

        def infer(obs, hidden, previous):
            observed.extend(np.asarray(previous).tolist())
            logits = np.stack([np.zeros(len(obs)), np.full(len(obs), 2.),
                               np.full(len(obs), -2.)], axis=1)
            return logits, hidden+1

        policy = OwnActionRecurrentPolicy(infer, hidden_size=1, action_count=3)
        rngs = [np.random.default_rng(seed) for seed in (10, 11)]
        first = policy.sample_with_rngs(np.zeros((2, 1)), rngs)
        self.assertEqual(observed, [3, 3])
        observed.clear()
        policy.sample_with_rngs(np.zeros((2, 1)), rngs[::-1])
        self.assertEqual(observed, [int(first[1]), int(first[0])])
        policy.reset_seed(9)
        self.assertFalse(policy.previous_actions)
        observed.clear()
        policy(np.zeros((1, 1)))
        self.assertEqual(observed, [3])
        with self.assertRaises(ValueError):
            policy.sample_with_rngs(np.zeros((2, 1)), [rngs[0], rngs[0]])

    def test_training_resume_loader_and_native_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = PPO(seed=3, action_count=20)
            source.save(root/"base", dict(steps=4, config=dict(
                game="defense", algorithm="ppo", game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION,
                action_names=list(action_names()), tstates=100000,
                observation_stride=1)))
            base = [sys.executable, "-m", "rl.defense_train", "--envs", "2",
                    "--rollout", "8", "--batch-size", "16", "--epochs", "1",
                    "--recurrent-hidden", "8", "--recurrent-own-action",
                    "--sequence-length", "4", "--mlx-cache-mb", "128",
                    "--max-episode-steps", "3"]

            def run(name, extra):
                result = subprocess.run(base+["--run", str(root/name),
                                          "--artifacts", str(root/(name+"-artifacts")),
                                          *extra], capture_output=True, text=True,
                                        timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/"latest/state.json").read_text())

            first = run("first", ["--initialize-policy", str(root/"base"), "--steps", "32"])
            self.assertEqual(first["steps"], 32)
            self.assertEqual(first["config"]["architecture"],
                             OWN_ACTION_RECURRENT_ARCHITECTURE)
            self.assertEqual(first["config"]["recurrent_own_action"], True)
            checkpoint = root/"first/latest/model.safetensors"
            policy, config = load_policy(checkpoint)
            self.assertIn("own-previous-key", policy_description(config))
            trace = record_game(policy, 10000, tstates=100000, max_steps=0)
            verification = verify_policy_trace(checkpoint, *trace[:4])
            self.assertTrue(verification["verified"])
            self.assertEqual(verification["verified_actions"], len(trace[1]))
            later = run("later", ["--resume", str(root/"first/latest"), "--steps", "64"])
            self.assertEqual(later["steps"], 64)


if __name__ == "__main__":
    unittest.main()
