import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.persistent_exploration import PersistentExploration, duration_distribution, start_probability


class PersistentExplorationTests(unittest.TestCase):
    def test_distribution_and_occupancy_calibration(self):
        lengths, probabilities = duration_distribution(64, 1.5)
        self.assertEqual(lengths.tolist(), list(range(1, 65)))
        self.assertAlmostEqual(probabilities.sum(), 1.)
        self.assertAlmostEqual(probabilities[0]/probabilities[3], 8.)
        mean = float(lengths@probabilities)
        for epsilon in (0., .05, .5, 1.):
            probability = start_probability(epsilon, mean)
            self.assertAlmostEqual(probability*mean/(1-probability+probability*mean), epsilon)
        explorer = PersistentExploration(128, 20, 64, 1.5, np.random.default_rng(5))
        greedy = np.zeros(128, np.int32)
        for _ in range(3000): explorer.select(greedy, .05)
        stats = explorer.stats()
        self.assertAlmostEqual(stats['realized_exploratory_fraction'], .05, delta=.006)
        self.assertGreater(sum(stats['sampled_duration_counts'][15:]), 20)
        self.assertEqual(sum(stats['sampled_duration_counts']), stats['starts'])
        self.assertEqual(sum(stats['exploratory_action_counts']), stats['exploratory_steps'])
        self.assertTrue(all(n > 0 for n in stats['exploratory_action_counts']))

    def test_exact_holds_boundary_cancellation_and_zero_epsilon(self):
        explorer = PersistentExploration(2, 20, 64, 1.5, np.random.default_rng(8))
        explorer.remaining[:] = [3, 2]
        explorer.held[:] = [4, 7]
        greedy = np.array([1, 2], np.int32)
        np.testing.assert_array_equal(explorer.select(greedy, 0.), [4, 7])
        np.testing.assert_array_equal(explorer.remaining, [2, 1])
        explorer.reset(np.array([True, False]))
        np.testing.assert_array_equal(explorer.select(greedy, 0.), [1, 7])
        np.testing.assert_array_equal(explorer.select(greedy, 0.), greedy)
        np.testing.assert_array_equal(greedy, [1, 2])
        self.assertEqual(explorer.stats()['cancelled_future_steps'], 2)

    def test_rng_reproducibility_and_resume_starts_without_active_holds(self):
        first = PersistentExploration(4, 20, 64, 1.5, np.random.default_rng(33))
        second = PersistentExploration(4, 20, 64, 1.5, np.random.default_rng(33))
        greedy = np.arange(4)
        for _ in range(100):
            np.testing.assert_array_equal(first.select(greedy, .2), second.select(greedy, .2))
        state = copy.deepcopy(first.rng.bit_generator.state)
        resumed_rng = np.random.default_rng(0); resumed_rng.bit_generator.state = state
        resumed = PersistentExploration(4, 20, 64, 1.5, resumed_rng)
        self.assertFalse(resumed.remaining.any())
        self.assertEqual(resumed.rng.bit_generator.state, state)
        first.reset(np.ones(4, bool))
        for _ in range(100):
            np.testing.assert_array_equal(first.select(greedy, .2), resumed.select(greedy, .2))

    def test_invalid_parameters_and_cli(self):
        for maximum, exponent in [(0, 1.5), (1025, 1.5), (True, 1.5), (2.5, 1.5),
                                  (64, 1), (64, np.nan)]:
            with self.assertRaises(ValueError): duration_distribution(maximum, exponent)
        for epsilon in (-1., 1.1, np.nan):
            with self.assertRaises(ValueError): start_probability(epsilon, 2)
        explorer = PersistentExploration(2, 20, 64, 1.5, np.random.default_rng(1))
        for actions in ([0], [0, 20], [0., 1.]):
            with self.assertRaises(ValueError): explorer.select(actions, .05)
        with self.assertRaises(ValueError): explorer.reset([0, 1])
        from rl import defense_dqn
        for extra in (['--exploration-max-repeat', '0'], ['--exploration-exponent', '1'],
                      ['--exploration-max-repeat', '64', '--bootstrap-heads', '5']):
            with patch.object(sys, 'argv', ['dqn', '--run', 'runs/invalid-test',
                                           '--artifacts', 'runs/invalid-artifact-test']+extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_disabled_identity_enabled_training_and_resume(self):
        import mlx.core as mx
        from rl.defense_learning import load_policy, policy_description
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                command = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/name),
                           '--artifacts', str(root/(name+'-artifacts'))]+extra
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            common = ['--envs', '2', '--batch-size', '4', '--capacity', '32', '--warmup', '4',
                      '--n-step', '2', '--train-every', '4', '--target-every', '2',
                      '--steps', '64', '--eval-every', '1000', '--max-episode-steps', '3',
                      '--mlx-cache-mb', '64']
            default = run('default', common)
            disabled = run('disabled', common+['--exploration-max-repeat', '1'])
            for key in ('steps', 'updates', 'episodes', 'rng'):
                self.assertEqual(default[key], disabled[key])
            for name in ('model.safetensors', 'target.safetensors', 'optimizer.npz'):
                a = mx.load(str(root/'default/latest'/name)); b = mx.load(str(root/'disabled/latest'/name))
                self.assertEqual(set(a), set(b))
                for key in a: np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            enabled = run('enabled', common+['--exploration-max-repeat', '64'])
            self.assertGreater(enabled['persistent_exploration']['exploratory_steps'], 0)
            self.assertGreater(enabled['persistent_exploration']['cancelled_future_steps'], 0)
            # Two-step replay insertion lags collection: six primitive actions
            # are needed before the four-entry warmup threshold is reached.
            self.assertEqual(enabled['persistent_exploration']['decisions'], 58)
            resumed = run('resumed', ['--resume', str(root/'enabled/latest'), '--steps', '96'])
            self.assertEqual(resumed['steps'], 96)
            self.assertEqual(resumed['config']['exploration_max_repeat'], 64)
            self.assertEqual(resumed['persistent_exploration']['decisions'], 26)
            unchanged = run('unchanged', ['--resume', str(root/'enabled/latest'), '--steps', '64'])
            self.assertEqual(unchanged['persistent_exploration_rng'], enabled['persistent_exploration_rng'])
            self.assertEqual(unchanged['persistent_exploration']['decisions'], 0)
            policy, config = load_policy(root/'enabled/latest/model.safetensors')
            self.assertEqual(policy_description(config), 'learned Q-values, greedy')
            self.assertFalse(hasattr(policy, 'remaining'))
            obs = np.full((2, 4, 16, 64), 128, np.uint8)
            np.testing.assert_array_equal(policy(obs), policy(obs))


if __name__ == '__main__':
    unittest.main()
