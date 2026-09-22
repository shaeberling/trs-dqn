import copy
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

from rl.defense_repeat import LearnedRepeatPolicy, OptionReturns, RepeatedActions, validate_spec


class Sink:
    def __init__(self):
        self.rows = []

    def add(self, *row):
        self.rows.append(copy.deepcopy(row))


class DefenseRepeatTests(unittest.TestCase):
    def test_spec_and_action_duration_mapping(self):
        for invalid in [(), (2,), (1, 1), (1, 0), (1, 257), (1, 2.5), (True, 4)]:
            with self.assertRaises(ValueError):
                validate_spec(20, invalid)
        controller = RepeatedActions(2, 3, (1, 4))
        np.testing.assert_array_equal(controller.select([4, 2]), [1, 2])
        for _ in range(3):
            np.testing.assert_array_equal(controller.select([0, 0]), [1, 0])
        np.testing.assert_array_equal(controller.remaining, [0, 0])
        np.testing.assert_array_equal(controller.select([2, 5]), [2, 2])
        controller.reset(np.array([False, True]))
        self.assertEqual(controller.cancelled_steps, 3)
        self.assertEqual(controller.stats()['executed_base_actions'], 10)
        self.assertEqual(controller.stats()['decisions'], 7)
        with self.assertRaises(ValueError):
            controller.select([6, 0])
        with self.assertRaises(ValueError):
            controller.reset([0, 1])

    def test_discounted_actual_duration_terminal_and_truncation(self):
        screen = np.zeros((4, 16, 64), np.uint8)
        sink = Sink()
        returns = OptionReturns(sink, gamma=.5, action_count=2, durations=(1, 4))
        returns.begin(screen, 3)
        screen[:] = 1  # The initial observation must be owned, not aliased.
        for i, reward in enumerate([1, 2, 4, 8]):
            self.assertEqual(returns.append(reward, screen, False, False), i == 3)
        first, option, reward, following, discount = sink.rows[-1]
        self.assertFalse(first.any())
        self.assertEqual(option, 3)
        self.assertEqual(reward, 4.)
        self.assertEqual(discount, .5**4)
        for terminal, truncated in [(True, False), (False, True)]:
            returns.begin(screen, 2)
            returns.append(2., screen, False, False)
            self.assertTrue(returns.append(6., screen, terminal, truncated))
            self.assertEqual(sink.rows[-1][2], 5.)
            self.assertEqual(sink.rows[-1][4], 0. if terminal else .25)
        self.assertEqual((returns.completed, returns.interrupted), (3, 2))
        with self.assertRaises(ValueError):
            returns.append(1., screen, False, False)

    def test_single_duration_matches_ordinary_one_step_returns(self):
        from rl.replay import NStep
        a, b = Sink(), Sink()
        option = OptionReturns(a, .997, 20, (1,))
        ordinary = NStep(b, 1, .997)
        rng = np.random.default_rng(19)
        for index in range(20):
            obs = rng.integers(0, 256, (4, 16, 64), dtype=np.uint8)
            following = rng.integers(0, 256, obs.shape, dtype=np.uint8)
            reward = float(rng.integers(100))
            terminal, truncated = index % 7 == 0, index % 9 == 0
            option.begin(obs, index)
            option.append(reward, following, terminal, truncated)
            ordinary.append(obs, index, reward, following, terminal, truncated)
        for aa, bb in zip(a.rows, b.rows, strict=True):
            for x, y in zip(aa, bb, strict=True):
                np.testing.assert_array_equal(x, y)

    def test_policy_game_identity_and_boundary_isolation_without_rng_draws(self):
        calls = []
        def infer(obs):
            calls.append(len(obs))
            values = np.zeros((len(obs), 6))
            values[np.arange(len(obs)), obs[:, 0, 0, 0].astype(int)] = 1.
            return values
        policy = LearnedRepeatPolicy(infer, 3, (1, 4))
        rngs = [np.random.default_rng(i) for i in range(2)]
        rng_states = [copy.deepcopy(r.bit_generator.state) for r in rngs]
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        obs[0] = 4
        obs[1] = 5
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, rngs), [1, 2])
        obs[:] = 0
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, rngs[::-1]), [2, 1])
        policy.observe_boundaries([True, False], rngs)
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, rngs), [0, 2])
        self.assertEqual(calls, [2, 1])
        self.assertEqual(rng_states, [r.bit_generator.state for r in rngs])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs[:1], rngs[1:]), [2])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs[:1], rngs[1:]), [0])
        policy.reset_seed(44)
        obs[:1] = 4
        self.assertEqual(policy(obs[:1])[0], 1)
        policy.observe_boundaries([True])
        obs[:1] = 0
        self.assertEqual(policy(obs[:1])[0], 0)
        with self.assertRaises(ValueError):
            policy.sample_with_rngs(obs, [rngs[0], rngs[0]])

    def test_cli_rejects_incompatible_combinations(self):
        from rl import defense_dqn
        for extra in [[], ['--n-step', '1', '--exploration-max-repeat', '64'],
                      ['--n-step', '1', '--no-life-terminal'], ['--n-step', '1', '--quantiles', '32'],
                      ['--n-step', '1', '--learned-repeats', ''],
                      ['--n-step', '1', '--learned-repeats', '1,4,4']]:
            with patch.object(sys, 'argv', ['dqn', '--run', '/nonexistent/repeat-test',
                    '--artifacts', '/nonexistent/repeat-artifacts', '--learned-repeats', '1,4'] + extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_own_scalar_initialization_and_learning(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from rl.defense_repeat_model import RepeatLearner
        from rl.model import QNetwork
        checkpoint = Path('results/defense/training/dqn-33-persistent-resets/step-000006962144/model.safetensors')
        parent = QNetwork(20)
        parent.load_weights(str(checkpoint))
        learner = RepeatLearner(durations=(1, 4, 16, 64))
        learner.initialize_scalar(checkpoint)
        obs = np.random.default_rng(9).integers(0, 256, (4, 4, 16, 64), dtype=np.uint8)
        expected = np.array(parent(mx.array(obs)))
        actual = np.array(learner.online(mx.array(obs))).reshape(4, 4, 20)
        for duration in range(4):
            np.testing.assert_allclose(actual[:, duration], expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(int(learner.optimizer.state['step'].item()), 0)
        for (name, online), (other, target) in zip(tree_flatten(learner.online.parameters()),
                tree_flatten(learner.target.parameters()), strict=True):
            self.assertEqual(name, other)
            np.testing.assert_array_equal(np.array(online), np.array(target))
        batch = (obs, np.array([0, 21, 42, 63]), np.arange(4, dtype=np.float32), obs,
                 np.array([.997, .997**4, .997**16, 0.], np.float32), np.ones(4, np.float32))
        loss, errors, q = learner.train(batch)
        self.assertTrue(np.isfinite([loss, q]).all() and np.isfinite(errors).all())
        self.assertEqual(int(learner.optimizer.state['step'].item()), 1)
        with self.assertRaises(ValueError):
            learner.initialize_scalar(checkpoint)

    def test_native_training_resume_parallel_recording_and_verification(self):
        from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/name),
                    '--artifacts', str(root/(name+'-artifacts'))] + extra, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            first = run('first', ['--learned-repeats', '1,4', '--n-step', '1', '--envs', '2',
                '--capacity', '256', '--compact-replay', '--batch-size', '4', '--warmup', '16',
                '--steps', '128', '--eval-every', '1000', '--max-episode-steps', '12',
                '--train-every', '4', '--target-every', '2', '--mlx-cache-mb', '64',
                '--epsilon-final', '1'])
            self.assertGreater(first['updates'], 0)
            self.assertEqual(first['learned_repeat_stats']['executed_base_actions'], 128)
            self.assertLess(first['learned_repeat_stats']['decisions'], 128)
            self.assertGreater(first['learned_repeat_interrupted'], 0)
            second = run('resumed', ['--resume', str(root/'first/latest'), '--steps', '192'])
            self.assertGreater(second['updates'], first['updates'])
            self.assertEqual(second['learned_repeat_stats']['executed_base_actions'], 64)
            checkpoint = root/'resumed/latest/model.safetensors'
            policy, config = load_policy(checkpoint)
            self.assertEqual(config['learned_repeats'], [1, 4])
            evaluation = evaluate(policy, [10000, 10001, 10002], envs=2)
            self.assertEqual(evaluation['complete_games'], 3)
            for game in evaluation['games']:
                recorded = record_game(policy, game['seed'], tstates=config['tstates'], max_steps=0)
                self.assertEqual(recorded[3], game)
            verified = verify_policy_trace(checkpoint, *recorded[:3], recorded[3])
            self.assertTrue(verified['verified'])
            with self.assertRaises(ValueError):
                load_policy(checkpoint, temperature=.5)


if __name__ == '__main__':
    unittest.main()
