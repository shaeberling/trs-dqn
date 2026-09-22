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

    def test_persistence_cancels_at_life_and_own_reset_boundaries(self):
        from rl import defense_dqn
        from rl.replay import Replay
        explorers, selections, buffers, supplied = [], [], [], []

        class Explorer(PersistentExploration):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.held[:] = [4, 7]
                self.remaining[:] = 5
                explorers.append(self)

            def select(self, greedy, epsilon):
                actions = super().select(greedy, epsilon)
                selections.append((greedy.copy(), actions.copy(), epsilon))
                return actions

        class RecordingReplay(Replay):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                buffers.append(self)

        class Workers:
            def __init__(self, count, seed, **kwargs):
                self.turn = 0
                self.observations = np.zeros((2, 4, 16, 64), np.uint8)
                self.config = kwargs

            def runtime(self): return []
            def close(self): pass

            def step(self, actions):
                supplied.append(actions.copy())
                self.turn += 1
                rows = []
                for i in range(2):
                    boundary = self.turn == 2
                    terminal = boundary and i == 1
                    info = dict(life_lost=boundary, full_game=False,
                                score=1020, episode_reward=20, highest_stage=1,
                                missions_completed=0, terminated=terminal, truncated=False)
                    rows.append((np.full((4, 16, 64), self.turn, np.uint8), 10,
                                 terminal, False, info,
                                 np.full((4, 16, 64), 99, np.uint8) if terminal else None))
                return rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ['dqn', '--run', str(root/'run'), '--artifacts', str(root/'artifacts'),
                    '--envs', '2', '--capacity', '32', '--warmup', '1', '--n-step', '1',
                    '--train-every', '100', '--steps', '6', '--eval-every', '6',
                    '--epsilon-steps', '1', '--epsilon-final', '0', '--mlx-cache-mb', '64',
                    '--exploration-max-repeat', '64', '--curriculum-probability', '.5',
                    '--curriculum-share', '--curriculum-boot-envs', '1']
            with patch.object(sys, 'argv', args), patch.object(defense_dqn, 'VectorEnv', Workers), \
                    patch.object(defense_dqn, 'Replay', RecordingReplay), \
                    patch('rl.persistent_exploration.PersistentExploration', Explorer), \
                    patch.object(defense_dqn, 'evaluate', return_value=dict(games=[])) as evaluate, \
                    patch.object(defense_dqn, 'publish_best'), \
                    contextlib.redirect_stdout(io.StringIO()):
                defense_dqn.main()
            np.testing.assert_array_equal(supplied[1], [4, 7])
            self.assertEqual([s[2] for s in selections], [0., 0.])
            np.testing.assert_array_equal(supplied[2], selections[1][0])
            self.assertEqual(explorers[0].stats()['cancelled_future_steps'], 8)
            replay = buffers[0]
            np.testing.assert_array_equal(replay.discounts[2:4], 0)
            np.testing.assert_array_equal(replay.next_obs[2:4], 2)
            np.testing.assert_array_equal(replay.obs[5], 99)
            np.testing.assert_allclose(replay.returns[:6], .1)
            self.assertFalse(any('curriculum' in key for key in evaluate.call_args.kwargs))
            state = json.loads((root/'run/latest/state.json').read_text())
            self.assertEqual((state['boot_episodes'], state['restored_segments']), (0, 1))

    def test_native_persistence_with_own_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/'run'),
                       '--artifacts', str(root/'artifacts'), '--envs', '2', '--capacity', '4096',
                       '--compact-replay', '--warmup', '32', '--batch-size', '4',
                       '--train-every', '256', '--steps', '4096', '--eval-every', '10000',
                       '--max-episode-steps', '512', '--mlx-cache-mb', '64',
                       '--exploration-max-repeat', '64', '--epsilon-final', '1',
                       '--curriculum-probability', '1', '--curriculum-share',
                       '--curriculum-boot-envs', '1', '--curriculum-lookback', '4']
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = [json.loads(x) for x in (root/'run/metrics.jsonl').read_text().splitlines()]
            archives = [r for r in rows if r['event'] == 'curriculum_archive']
            self.assertTrue(archives)
            self.assertTrue(all(r['trigger_action']-r['source_action'] == 4 for r in archives))
            episodes = [r for r in rows if r['event'] == 'episode']
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker'] == 0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker'] == 1))
            state = json.loads((root/'run/latest/state.json').read_text())
            self.assertGreater(state['updates'], 0)
            self.assertGreater(state['restored_segments'], 0)
            self.assertGreater(state['persistent_exploration']['cancelled_future_steps'], 0)
            self.assertGreater(state['persistent_exploration']['exploratory_steps'], 0)
            self.assertEqual(state['boot_episodes']+state['restored_segments'], state['episodes'])
            self.assertFalse(state['config']['curriculum_archive_saved'])


if __name__ == '__main__':
    unittest.main()
