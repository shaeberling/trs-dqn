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

from rl.defense_learning import QUANTILE_ALGORITHM, load_policy, policy_description
from rl.defense_quantile import QuantileLearner, QuantileQ, distortion_weights, quantile_huber
from rl.train import checkpoint
from rl.model import Learner
from tests.test_defense_dqn import configuration


def arrays(tree):
    return {k: np.array(v) for k, v in tree_flatten(tree)}


class DefenseQuantileTests(unittest.TestCase):
    def test_scalar_transfer_preserves_encoder_and_tiles_heads_with_fresh_adam(self):
        scalar = Learner(seed=17, action_count=20)
        obs = np.random.default_rng(4).integers(128, 192, (3, 4, 16, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(scalar, root, dict(config=configuration(), steps=32))
            quantile = QuantileLearner(seed=22, quantiles=4, exploration_power=1.5)
            quantile.initialize_scalar(root/'model.safetensors')
            source = arrays(scalar.online.parameters())
            for key, value in arrays(quantile.online.parameters()).items():
                expected = np.repeat(source[key], 4, axis=0) if key.startswith(('value.', 'advantage.')) else source[key]
                np.testing.assert_array_equal(value, expected)
                np.testing.assert_array_equal(value, arrays(quantile.target.parameters())[key])
            actual = np.array(quantile.online.quantile_values(mx.array(obs)))
            expected = np.repeat(np.array(scalar.online(mx.array(obs)))[:, :, None], 4, axis=2)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
            np.testing.assert_array_equal(quantile.actions(obs), scalar.actions(obs))
            self.assertEqual(quantile.optimizer.state['step'].item(), 0)
            for key, value in arrays(quantile.optimizer.state).items():
                if key.endswith(('.m', '.v')):
                    self.assertTrue(np.all(value == 0))
            batch = (obs, np.zeros(3, np.int32), np.ones(3, np.float32), obs,
                     np.zeros(3, np.float32), np.ones(3, np.float32))
            quantile.train(batch)
            with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
                quantile.initialize_scalar(root/'model.safetensors')

    def test_native_scalar_initialization_and_resumed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p/'parent'
            checkpoint(Learner(seed=17, action_count=20), source, dict(config=configuration(), steps=999))
            base = [sys.executable, '-m', 'rl.defense_dqn']
            args = ['--run', str(p/'run'), '--artifacts', str(p/'artifacts'), '--quantiles', '4',
                    '--init-from-dqn', str(source), '--envs', '2', '--batch-size', '4',
                    '--capacity', '32', '--warmup', '4', '--train-every', '4', '--steps', '16',
                    '--eval-every', '1000', '--max-episode-steps', '3', '--mlx-cache-mb', '64']
            first = subprocess.run(base+args, capture_output=True, text=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
            state = json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['steps'], 16)  # Not inherited 999.
            self.assertEqual(state['config']['quantile_initialization']['steps'], 999)
            resumed = subprocess.run(base+['--run', str(p/'resumed'), '--artifacts', str(p/'resumed-artifacts'),
                                          '--resume', str(p/'run/latest'), '--steps', '32'],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout+resumed.stderr)
            later = json.loads((p/'resumed/latest/state.json').read_text())
            self.assertEqual(later['config']['quantile_initialization'], state['config']['quantile_initialization'])
            self.assertIsNone(later['config']['init_from_dqn'])

    def test_power_bin_masses_and_validation(self):
        np.testing.assert_array_equal(distortion_weights(4), np.full(4, .25, np.float32))
        np.testing.assert_array_equal(distortion_weights(4, 1), [.0625, .1875, .3125, .4375])
        for n in (2, 3, 32, 256):
            w = distortion_weights(n, 1.5)
            self.assertTrue(np.all(w > 0))
            self.assertAlmostEqual(float(w.sum()), 1., places=6)
        for n, power in [(0, 0), (1, 0), (257, 0), (True, 0), (3.5, 0),
                         (4, -1), (4, 5), (4, np.nan), (4, np.inf), (4, True)]:
            with self.subTest(n=n, power=power), self.assertRaises(ValueError):
                distortion_weights(n, power)

    def test_huber_pairwise_sign_midpoints_reduction_and_gradient(self):
        predictions = np.array([[0., 2.], [-1., 3.]], np.float32)
        labels = np.array([[1., 4.], [0., .5]], np.float32)
        delta = labels[:, None, :] - predictions[:, :, None]
        huber = np.where(np.abs(delta) <= 1, .5*delta**2, np.abs(delta)-.5)
        expected = (np.abs(np.array([.25, .75])[None, :, None] - (delta < 0))*huber).mean(2).sum(1)
        actual = quantile_huber(mx.array(predictions), mx.array(labels))
        np.testing.assert_allclose(np.array(actual), expected)
        gradient = mx.grad(lambda p: mx.sum(quantile_huber(p, mx.array(labels))))(mx.array(predictions))
        expected_grad = (-np.abs(np.array([.25, .75])[None, :, None]-(delta < 0))*np.clip(delta, -1, 1)).mean(2)
        np.testing.assert_allclose(np.array(gradient), expected_grad)

    def test_double_q_mean_target_terminal_and_distribution_priority(self):
        class Values:
            def __init__(self, values): self.values = mx.array(values)
            def quantile_values(self, states): return self.values[states]
            def __call__(self, states): return mx.mean(self.quantile_values(states), axis=2)

        agent = QuantileLearner(seed=9, quantiles=2, exploration_power=4)
        # Online next mean selects 1; upper quantile and target mean prefer 0.
        online = Values([[[0., 2.], [0., 0.]], [[-100., 100.], [2., 4.]]])
        agent.target = Values([[[0., 0.], [0., 0.]], [[100., 100.], [4., 6.]]])
        loss, (priority, q) = agent._loss(online, mx.array([0, 0]), mx.array([0, 0]),
                                        mx.array([2., 7.]), mx.array([1, 1]),
                                        mx.array([.5, 0.]), mx.array([1., .25]))
        # First labels 2+.5*[4,6]; second terminal labels independent of target.
        expected = quantile_huber(mx.array([[0., 2.], [0., 2.]]), mx.array([[4., 5.], [7., 7.]]))
        np.testing.assert_allclose(np.array(priority), np.array(expected)/2)
        self.assertAlmostEqual(loss.item(), float(np.mean(np.array(expected)*[1, .25])))
        self.assertEqual(q.item(), 1.)

    def test_training_distortion_is_not_evaluation_or_sorted_predictions(self):
        agent = QuantileLearner(seed=4, quantiles=4, exploration_power=1)
        model = agent.online
        model.value.weight = mx.zeros_like(model.value.weight)
        model.value.bias = mx.zeros_like(model.value.bias)
        model.advantage.weight = mx.zeros_like(model.advantage.weight)
        biases = np.full((20, 4), -20., np.float32)
        biases[0] = [4., 4., 4., 4.]
        biases[1] = [0., 0., 0., 12.]
        biases[2] = [12., 0., 0., 0.]  # Sorting would wrongly make this identical to action 1.
        model.advantage.bias = mx.array(biases.reshape(-1))
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        np.testing.assert_array_equal(agent.actions(obs), [1, 1])
        np.testing.assert_array_equal(np.array(agent.predict(mx.array(obs))), [0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(configuration(), algorithm=QUANTILE_ALGORITHM, quantiles=4,
                          quantile_exploration_power=1.)
            checkpoint(agent, root, dict(config=config, steps=0))
            policy, loaded = load_policy(root/'model.safetensors')
            np.testing.assert_array_equal(policy(obs), [0, 0])
            self.assertEqual(policy_description(loaded), 'learned score-return quantiles, greedy mean Q-values')
            with self.assertRaisesRegex(ValueError, 'Temperature'):
                load_policy(root/'model.safetensors', temperature=.5)

    def test_matching_initialization_updates_and_unchanged_target(self):
        neutral = QuantileLearner(seed=12, quantiles=4)
        risk = QuantileLearner(seed=12, quantiles=4, exploration_power=1.5)
        before = arrays(neutral.target.parameters())
        for k, v in arrays(risk.online.parameters()).items():
            np.testing.assert_array_equal(before[k], v)
        rng = np.random.default_rng(8)
        obs = rng.integers(128, 192, (4, 4, 16, 64), dtype=np.uint8)
        batch = (obs, np.array([19, 18, 17, 16], np.int32), np.ones(4, np.float32),
                 obs, np.full(4, .8, np.float32), np.ones(4, np.float32))
        a, b = neutral.train(batch), risk.train(batch)
        np.testing.assert_array_equal(a[1], b[1]); self.assertEqual(a[0], b[0])
        self.assertTrue(np.isfinite(a[1]).all())
        for k, v in arrays(neutral.online.parameters()).items():
            np.testing.assert_array_equal(v, arrays(risk.online.parameters())[k])
        for k, v in arrays(neutral.target.parameters()).items():
            np.testing.assert_array_equal(v, before[k])
        self.assertTrue(any(not np.array_equal(v, before[k]) for k, v in arrays(neutral.online.parameters()).items()))
        neutral.sync_target()
        for k, v in arrays(neutral.target.parameters()).items():
            np.testing.assert_array_equal(v, arrays(neutral.online.parameters())[k])

    def test_cli_validation_and_cross_algorithm_resume(self):
        from rl import defense_dqn
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            base = ['dqn', '--run', str(p/'run'), '--artifacts', str(p/'artifacts')]
            for extra in (['--quantiles', '1'], ['--quantiles', '-2'], ['--quantiles', '257'],
                          ['--quantile-exploration-power', '1'],
                          ['--quantiles', '4', '--quantile-exploration-power', 'nan'],
                          ['--quantiles', '4', '--quantile-exploration-power', '-1'],
                          ['--quantiles', '4', '--bootstrap-heads', '2'],
                          ['--quantiles', '4', '--exploration-max-repeat', '2']):
                with self.subTest(extra=extra), patch.object(sys, 'argv', base+extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                    defense_dqn.main()
                self.assertEqual(e.exception.code, 2)
            (p/'state.json').write_text(json.dumps(dict(config=configuration())))
            with patch.object(sys, 'argv', base+['--resume', str(p), '--quantiles', '4']), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                defense_dqn.main()
            self.assertEqual(e.exception.code, 2)

    def test_real_emulator_training_and_exact_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            base = [sys.executable, '-m', 'rl.defense_dqn']
            command = base+['--run', str(p/'run'), '--artifacts', str(p/'artifacts'),
                            '--quantiles', '4', '--quantile-exploration-power', '1.5',
                            '--envs', '2', '--batch-size', '4', '--capacity', '32', '--compact-replay',
                            '--warmup', '4', '--n-step', '2', '--train-every', '4', '--target-every', '2',
                            '--steps', '32', '--eval-every', '1000', '--max-episode-steps', '3',
                            '--mlx-cache-mb', '64']
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            state = json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['config']['algorithm'], QUANTILE_ALGORITHM)
            self.assertGreater(state['updates'], 0)
            rows = [json.loads(l) for l in (p/'run/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(not w['mlx_loaded'] for r in rows if r['event']=='workers_started' for w in r['workers']))
            clone = base+['--run', str(p/'clone'), '--artifacts', str(p/'clone-artifacts'),
                         '--resume', str(p/'run/latest'), '--steps', '32']
            result = subprocess.run(clone, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            saved = json.loads((p/'clone/latest/state.json').read_text())
            for key in ('steps', 'updates', 'episodes', 'rng'):
                self.assertEqual(state[key], saved[key])
            for filename in ('model.safetensors', 'target.safetensors', 'optimizer.npz'):
                a, b = mx.load(str(p/'run/latest'/filename)), mx.load(str(p/'clone/latest'/filename))
                self.assertEqual(set(a), set(b))
                for key in a:
                    np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            result = subprocess.run(base+['--run', str(p/'later'), '--artifacts', str(p/'later-artifacts'),
                                         '--resume', str(p/'run/latest'), '--steps', '64'],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            later = json.loads((p/'later/latest/state.json').read_text())
            self.assertGreater(later['updates'], state['updates'])
            self.assertEqual(later['config']['quantiles'], 4)
            self.assertEqual(later['config']['quantile_exploration_power'], 1.5)
