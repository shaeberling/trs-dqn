import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_duration_ppo import (DurationCategoricalPolicy, DurationPPO,
                                     duration_action_names, initialize_duration_policy,
                                     option_actor_gae)
from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace
from rl.defense_repeat import RepeatedActions
from rl.ppo import gae


class SequencePolicy:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def reset_seed(self, seed):
        self.seed = seed

    def sample_with_rngs(self, obs, rngs):
        self.calls.append(tuple(rngs))
        return np.asarray(next(self.batches), np.int32)

    def __call__(self, obs):
        return np.asarray(next(self.batches), np.int32)


class DefenseDurationPPOTests(unittest.TestCase):
    def test_complete_option_credit_reaches_delayed_score(self):
        rewards = np.zeros((64, 1), np.float32)
        rewards[-1, 0] = 1.
        values = np.zeros_like(rewards)
        boundaries = np.zeros_like(rewards, bool)
        boundaries[-1, 0] = True
        starts = np.zeros_like(rewards, bool)
        starts[0, 0] = True
        actor, mask = option_actor_gae(rewards, values, boundaries, starts,
                                      np.zeros(1, np.float32), np.array([False]), .997, .95)
        ordinary, _ = gae(rewards, values, boundaries.astype(np.float32),
                          np.zeros(1, np.float32), .997, .95)
        self.assertTrue(mask[0, 0])
        self.assertAlmostEqual(float(actor[0, 0]), .997**63, places=6)
        self.assertAlmostEqual(float(ordinary[0, 0]), (.997*.95)**63, places=6)
        self.assertGreater(float(actor[0, 0]), 20*float(ordinary[0, 0]))
        np.testing.assert_array_equal(actor[1:], 0)

    def test_option_credit_respects_boundaries_bootstrap_and_unfinished_hold(self):
        rewards = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], np.float32)
        values = np.zeros_like(rewards)
        boundaries = np.zeros_like(rewards, bool)
        boundaries[4, 0] = True
        starts = np.zeros_like(rewards, bool)
        starts[0] = True
        starts[2, 0] = True
        actor, mask = option_actor_gae(rewards, values, boundaries, starts,
                                      np.zeros(2, np.float32), np.array([False, True]), .9, .8)
        last = 3+.9*4+.9**2*5
        self.assertAlmostEqual(float(actor[2, 0]), last, places=6)
        self.assertAlmostEqual(float(actor[0, 0]), 1+.9*2+.9**2*.8*last, places=6)
        self.assertFalse(mask[0, 1])
        self.assertEqual(float(actor[0, 1]), 0.)
        self.assertEqual(int(mask.sum()), 2)
        with self.assertRaises(ValueError):
            changed = boundaries.copy()
            changed[0, 0] = True
            option_actor_gae(rewards, values, changed, starts,
                             np.zeros(2, np.float32), np.array([False, True]), .9, .8)
        bootstrapped, completed = option_actor_gae(
            np.array([[1.], [2.]], np.float32), np.zeros((2, 1), np.float32),
            np.zeros((2, 1), bool), np.array([[True], [False]]),
            np.array([7.], np.float32), np.array([False]), .9, .8)
        self.assertTrue(completed[0, 0])
        self.assertAlmostEqual(float(bootstrapped[0, 0]), 1+.9*2+.9**2*7, places=6)

    def test_option_actor_credit_requires_joint_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"unused"
            result = subprocess.run([sys.executable, "-m", "rl.defense_train",
                                     "--run", str(output), "--option-actor-gae"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--option-actor-gae requires learned durations", result.stderr)
            self.assertFalse(output.exists())

    def test_executor_and_parallel_policy_cancel_only_on_visible_boundaries(self):
        durations = (1, 4, 16)
        executor = RepeatedActions(2, durations=durations)
        np.testing.assert_array_equal(executor.select([24, 3]), [4, 3])
        self.assertEqual(executor.remaining.tolist(), [3, 0])
        np.testing.assert_array_equal(executor.select([0, 44]), [4, 4])
        self.assertEqual(executor.remaining.tolist(), [2, 15])
        executor.reset(np.array([True, False]))
        self.assertEqual(executor.remaining.tolist(), [0, 15])
        source = SequencePolicy([[24, 3], [44], [2]])
        policy = DurationCategoricalPolicy(source, durations)
        left, right = object(), object()
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [left, right]), [4, 3])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [right, left]), [4, 4])
        policy.observe_boundaries(np.array([True, False]), [right, left])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [left, right]), [4, 2])
        self.assertEqual(source.calls[1], (right,))
        self.assertEqual(source.calls[2], (right,))
        self.assertEqual(len(duration_action_names(durations)), 60)

    def test_initializer_preserves_parent_logits_and_value_with_neutral_priors(self):
        import mlx.core as mx
        from rl.model import QNetwork

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            original = QNetwork(action_count=20)
            original.save_weights(str(source/"model.safetensors"))
            config = dict(game="defense", game_sha256=GAME_SHA256,
                          environment_version=ENVIRONMENT_VERSION, algorithm="ppo",
                          action_names=list(action_names(False)), tstates=100000,
                          observation_stride=1)
            (source/"state.json").write_text(json.dumps(dict(config=config, steps=123)))
            extended = QNetwork(action_count=80)
            metadata = initialize_duration_policy(extended, source, (1, 4, 16, 64),
                                                  tstates=100000, observation_stride=1)
            observation = mx.array(np.zeros((2, 4, 16, 64), np.uint8))
            before, value_before = original.policy_value(observation)
            after, value_after = extended.policy_value(observation)
            for index, offset in enumerate((0., -2., -4., -6.)):
                np.testing.assert_allclose(np.array(after[:, index*20:(index+1)*20]),
                                           np.array(before)+offset, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array(value_after), np.array(value_before),
                                       rtol=1e-5, atol=1e-5)
            self.assertEqual(metadata["source_training_steps"], 123)
            self.assertEqual(metadata["duration_logit_offsets"], [0., -2., -4., -6.])
            self.assertEqual(metadata["duration_logit_spacing"], 2.)
            cautious = QNetwork(action_count=80)
            second = initialize_duration_policy(cautious, source, (1, 4, 16, 64),
                                                tstates=100000, observation_stride=1,
                                                logit_spacing=5.)
            self.assertEqual(second["duration_logit_offsets"], [0., -5., -10., -15.])
            with self.assertRaises(ValueError):
                initialize_duration_policy(QNetwork(80), source, (1, 4, 16, 64),
                                           tstates=50000, observation_stride=1)
            with self.assertRaises(ValueError):
                initialize_duration_policy(QNetwork(80), source, (1, 4, 16, 64),
                                           tstates=100000, observation_stride=1,
                                           logit_spacing=float('nan'))

    def test_mask_excludes_forced_steps_from_actor_objective(self):
        import mlx.core as mx

        agent = DurationPPO(action_count=80, entropy=.002)
        obs = mx.array(np.zeros((2, 4, 16, 64), np.uint8))
        actions = mx.array([4, 0], mx.int32)
        old = mx.array([0., 0.])
        advantages_a = mx.array([1., 1000.])
        advantages_b = mx.array([1., -1000.])
        returns = mx.array([1., 2.])
        mask = mx.array([1., 0.])
        loss_a, metrics_a = agent._loss(agent.model, obs, actions, old,
                                      advantages_a, returns, mask)
        loss_b, metrics_b = agent._loss(agent.model, obs, actions, old,
                                      advantages_b, returns, mask)
        np.testing.assert_allclose(float(loss_a.item()), float(loss_b.item()), atol=1e-6)
        np.testing.assert_allclose([float(x.item()) for x in metrics_a],
                                   [float(x.item()) for x in metrics_b], atol=1e-6)
        no_actor, metrics = agent._loss(agent.model, obs, actions, old,
                                       advantages_a, returns, mx.array([0., 0.]))
        self.assertTrue(np.isfinite(float(no_actor.item())))
        self.assertEqual(float(metrics[0].item()), 0.)
        self.assertEqual(float(metrics[2].item()), 0.)

    def test_real_native_parallel_and_serial_replay_match(self):
        import mlx.core as mx
        from rl.model import QNetwork

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model = QNetwork(action_count=80)
            model.save_weights(str(path/"model.safetensors"))
            config = dict(game="defense", game_sha256=GAME_SHA256,
                          environment_version=ENVIRONMENT_VERSION, algorithm="ppo",
                          action_names=list(action_names(False)), learned_durations=[1, 4, 16, 64],
                          policy_action_names=list(duration_action_names((1, 4, 16, 64))),
                          life_terminal=True, tstates=100000, observation_stride=1,
                          eval_max_steps=0, allow_enter=False)
            (path/"state.json").write_text(json.dumps(dict(config=config)))
            policy, loaded = load_policy(path/"model.safetensors")
            self.assertEqual(loaded["learned_durations"], [1, 4, 16, 64])
            result = evaluate(policy, [604800, 604801], envs=2)
            self.assertEqual(result["complete_games"], 2)
            policy, _ = load_policy(path/"model.safetensors")
            actual = record_game(policy, 604800, tstates=100000, max_steps=0)
            self.assertEqual(actual[3], result["games"][0])
            checked = verify_policy_trace(path/"model.safetensors", *actual[:4])
            self.assertTrue(checked["verified"])
            self.assertEqual(checked["verified_actions"], len(actual[1]))
            mx.eval(model.state)


if __name__ == "__main__":
    unittest.main()
