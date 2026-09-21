import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256
from rl.defense_learning import DQN_ALGORITHM, greedy_policy, load_policy, policy_description
from rl.model import Learner
from rl.replay import NStep, Replay
from rl.train import checkpoint


def configuration():
    return dict(game="defense", game_sha256=GAME_SHA256, algorithm=DQN_ALGORITHM,
                environment_version=ENVIRONMENT_VERSION, action_names=list(ACTION_NAMES),
                allow_enter=False, tstates=100000, eval_max_steps=0)


class DefenseDQNTests(unittest.TestCase):
    def test_greedy_policy_does_not_draw_randomness_and_rejects_invalid_values(self):
        policy = greedy_policy(lambda obs: np.tile(np.arange(20), (len(obs), 1)))
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        rngs = [np.random.default_rng(i) for i in range(2)]
        before = [r.bit_generator.state for r in rngs]
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, rngs), [19, 19])
        policy.reset_seed(10)
        np.testing.assert_array_equal(policy(obs), [19, 19])
        self.assertEqual(before, [r.bit_generator.state for r in rngs])
        with self.assertRaises(ValueError):
            policy.sample_with_rngs(obs, rngs[:1])
        for values in (np.zeros(2), np.zeros((3, 20)), np.full((2, 20), np.nan)):
            with self.assertRaises(ValueError):
                greedy_policy(lambda obs: values)(obs)

    def test_twenty_action_update_target_sync_and_greedy_checkpoint(self):
        agent = Learner(seed=9, action_count=20)
        rng = np.random.default_rng(8)
        obs = rng.integers(128, 192, (4, 4, 16, 64), dtype=np.uint8)
        before = {k: np.array(v) for k, v in tree_flatten(agent.target.parameters())}
        self.assertEqual(agent.online(mx.array(obs)).shape, (4, 20))
        batch = (obs, np.array([19, 18, 17, 16], np.int32),
                 np.full(4, 3, np.float32), obs, np.zeros(4, np.float32), np.ones(4, np.float32))
        loss, errors, q = agent.train(batch)
        self.assertTrue(np.isfinite([loss, q]).all())
        self.assertTrue(np.isfinite(errors).all())
        for k, v in tree_flatten(agent.target.parameters()):
            np.testing.assert_array_equal(before[k], np.array(v))
        self.assertTrue(any(not np.array_equal(before[k], np.array(v))
                            for k, v in tree_flatten(agent.online.parameters())))
        agent.sync_target()
        for (_, online), (_, target) in zip(tree_flatten(agent.online.parameters()),
                                           tree_flatten(agent.target.parameters()), strict=True):
            np.testing.assert_array_equal(np.array(online), np.array(target))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            checkpoint(agent, p, dict(config=configuration(), steps=8))
            policy, config = load_policy(p/"model.safetensors")
            np.testing.assert_array_equal(policy(obs), agent.actions(obs))
            self.assertEqual(policy_description(config), "learned Q-values, greedy")
            with self.assertRaisesRegex(ValueError, "Temperature"):
                load_policy(p/"model.safetensors", temperature=.5)
            config["algorithm"] = "unknown"
            (p/"state.json").write_text(json.dumps(dict(config=config)))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_policy(p/"model.safetensors")

    def test_double_q_selects_online_action_but_uses_target_value(self):
        agent = Learner(seed=7, action_count=20)
        # Online next-state argmax is action 1, while target argmax is action 0.
        class Values:
            def __init__(self, table):
                self.table = mx.array(table)

            def __call__(self, states):
                return self.table[states]

        online = Values([[1., 0.], [2., 3.]])
        agent.target = Values([[0., 0.], [100., 4.]])
        loss, (errors, q) = agent._loss(online, mx.array([0]), mx.array([0]),
                                      mx.array([2.]), mx.array([1]), mx.array([.5]), mx.array([1.]))
        # Target = 2 + .5 * 4 = 4; residual 1 - 4 = -3, Huber = 2.5.
        self.assertAlmostEqual(loss.item(), 2.5)
        np.testing.assert_allclose(np.array(errors), [3.])
        self.assertAlmostEqual(q.item(), 1.)

    def test_nstep_life_boundary_never_crosses_into_next_life_and_truncation_bootstraps(self):
        replay = Replay(8)
        nstep = NStep(replay, n=5, gamma=.9)
        frames = [np.full((4, 16, 64), 128+i, np.uint8) for i in range(4)]
        nstep.append(frames[0], 19, 1., frames[1], False, False)
        nstep.append(frames[1], 18, 2., frames[2], True, False)
        self.assertFalse(nstep.queue)
        np.testing.assert_allclose(replay.returns[:2], [2.8, 2.])
        np.testing.assert_array_equal(replay.discounts[:2], [0., 0.])
        np.testing.assert_array_equal(replay.next_obs[:2], [frames[2], frames[2]])
        nstep.append(frames[2], 17, .7, frames[3], False, True)
        self.assertFalse(nstep.queue)
        self.assertAlmostEqual(replay.returns[2], .7, places=6)
        self.assertAlmostEqual(replay.discounts[2], .9, places=6)
        np.testing.assert_array_equal(replay.next_obs[2], frames[3])

    def test_cli_rejects_invalid_configuration_and_cross_algorithm_resume(self):
        from rl import defense_dqn, defense_train
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/"state.json").write_text(json.dumps(dict(config=configuration())))
            with patch.object(sys, "argv", ["train", "--run", str(p/"new"), "--resume", str(p)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_train.main()
            self.assertEqual(error.exception.code, 2)
            base = ["dqn", "--run", str(p/"new"), "--artifacts", str(p/"artifacts")]
            for extra in (["--capacity", "0"], ["--warmup", "50001"], ["--gamma", "nan"],
                          ["--epsilon-final", "1.1"], ["--n-step", "0"], ["--steps", "-1"],
                          ["--resume", str(p)]):
                with self.subTest(extra=extra), patch.object(sys, "argv", base+extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    defense_dqn.main()
                self.assertEqual(error.exception.code, 2)
            cfg = configuration(); cfg["algorithm"] = "ppo"
            (p/"state.json").write_text(json.dumps(dict(config=cfg)))
            with patch.object(sys, "argv", base+["--resume", str(p)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_real_emulator_fresh_training_and_optimizer_target_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            command = [sys.executable, "-m", "rl.defense_dqn", "--run", str(p/"run"),
                       "--artifacts", str(p/"artifacts"), "--envs", "2", "--batch-size", "4",
                       "--capacity", "32", "--warmup", "4", "--n-step", "2", "--train-every", "4",
                       "--target-every", "2", "--steps", "32", "--eval-every", "1000",
                       "--max-episode-steps", "3", "--mlx-cache-mb", "64"]
            first = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
            state = json.loads((p/"run/latest/state.json").read_text())
            self.assertEqual(state["steps"], 32)
            self.assertGreater(state["updates"], 0)
            self.assertEqual(state["config"]["algorithm"], DQN_ALGORITHM)
            self.assertTrue((p/"run/latest/target.safetensors").is_file())
            rows = [json.loads(l) for l in (p/"run/metrics.jsonl").read_text().splitlines()]
            workers = next(r["workers"] for r in rows if r["event"] == "workers_started")
            self.assertTrue(all(not r["mlx_loaded"] for r in workers))
            # A zero-update resume must preserve online, lagged target and Adam
            # independently, not silently reinitialize or synchronize any of them.
            cloned = subprocess.run([sys.executable, "-m", "rl.defense_dqn", "--run", str(p/"clone"),
                                     "--artifacts", str(p/"clone-artifacts"),
                                     "--resume", str(p/"run/latest"), "--steps", "32"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(cloned.returncode, 0, cloned.stdout+cloned.stderr)
            clone_state = json.loads((p/"clone/latest/state.json").read_text())
            for field in ("steps", "updates", "episodes", "rng"):
                self.assertEqual(clone_state[field], state[field])
            for file in ("model.safetensors", "target.safetensors", "optimizer.npz"):
                original, copied = mx.load(str(p/"run/latest"/file)), mx.load(str(p/"clone/latest"/file))
                self.assertEqual(set(original), set(copied))
                for name in original:
                    np.testing.assert_array_equal(np.array(original[name]), np.array(copied[name]))
            resumed = subprocess.run([sys.executable, "-m", "rl.defense_dqn", "--run", str(p/"resumed"),
                                      "--artifacts", str(p/"resumed-artifacts"),
                                      "--resume", str(p/"run/latest"), "--steps", "64"],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout+resumed.stderr)
            later = json.loads((p/"resumed/latest/state.json").read_text())
            self.assertEqual(later["steps"], 64)
            self.assertGreater(later["updates"], state["updates"])
            self.assertEqual(later["config"]["envs"], 2)
            # Resume refills replay; target and optimizer are saved, not trajectories.
            self.assertFalse(any("replay" in x.name for x in (p/"resumed/latest").iterdir()))


if __name__ == "__main__":
    unittest.main()
