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

from rl.defense_option_returns import MultiOptionReturns
from rl.defense_repeat import OptionReturns
from rl.replay import NStep


class Sink:
    def __init__(self):
        self.rows = []

    def add(self, *row):
        self.rows.append(copy.deepcopy(row))


class DefenseOptionReturnsTests(unittest.TestCase):
    def test_actual_duration_discounts_and_boundary_flush(self):
        for terminal, truncated in [(False, False), (True, False), (False, True)]:
            sink = Sink()
            returns = MultiOptionReturns(sink, 3, .5, 1, (1, 4))
            screen = np.zeros((4, 16, 64), np.uint8)
            for i, (option, rewards) in enumerate([(0, [2]), (1, [4, 8, 16, 32]), (0, [64])]):
                screen[:] = i
                returns.begin(screen, option)
                for j, reward in enumerate(rewards):
                    screen[:] = i + 1
                    returns.append(reward, screen, terminal and i == 2, truncated and i == 2)
            self.assertEqual(sink.rows[0][2], 12.)
            self.assertEqual(sink.rows[0][4], 0. if terminal else .5**6)
            self.assertFalse(sink.rows[0][0].any())
            screen[:] = 99
            self.assertTrue((sink.rows[0][3] == 3).all())
            self.assertEqual(returns.completed, 3)
            self.assertEqual(returns.emitted + len(returns.queue), 3)
            if terminal or truncated:
                self.assertEqual([r[2] for r in sink.rows], [12., 20., 64.])
                self.assertEqual([r[4] for r in sink.rows], [0., 0., 0.] if terminal else [.5**6, .5**5, .5])
                self.assertFalse(returns.queue)
                returns.begin(screen, 0)
                returns.append(100., screen, True, False)
                self.assertEqual(sink.rows[-1][2], 100.)

    def test_interrupted_option_flushes_owned_following_without_crossing_lives(self):
        sink = Sink()
        returns = MultiOptionReturns(sink, 5, .5, 1, (1, 4))
        screen = np.zeros((4, 16, 64), np.uint8)
        returns.begin(screen, 0)
        returns.append(1., screen, False, False)
        returns.begin(screen, 1)
        returns.append(2., screen, False, False)
        screen[:] = 7
        returns.append(4., screen, True, False)
        screen[:] = 9
        self.assertEqual([r[2] for r in sink.rows], [3., 4.])
        self.assertEqual([r[4] for r in sink.rows], [0., 0.])
        self.assertTrue(all((r[3] == 7).all() for r in sink.rows))
        self.assertEqual((returns.completed, returns.interrupted, returns.emitted), (2, 1, 2))

    def test_single_option_and_single_duration_equivalence(self):
        for horizon in [1, 5]:
            a, b = Sink(), Sink()
            multi = MultiOptionReturns(a, horizon, .997, 20, (1,))
            ordinary = NStep(b, horizon, .997)
            rng = np.random.default_rng(619)
            for i in range(100):
                obs = rng.integers(256, size=(4, 16, 64), dtype=np.uint8)
                following = rng.integers(256, size=obs.shape, dtype=np.uint8)
                reward = float(rng.integers(100))
                terminal, truncated = i % 13 == 0, i % 19 == 0
                multi.begin(obs, i % 20)
                multi.append(reward, following, terminal, truncated)
                ordinary.append(obs, i % 20, reward, following, terminal, truncated)
            for aa, bb in zip(a.rows, b.rows, strict=True):
                for x, y in zip(aa, bb, strict=True):
                    np.testing.assert_array_equal(x, y)
        a, b = Sink(), Sink()
        multi = MultiOptionReturns(a, 1, .997, 20, (1, 4))
        single = OptionReturns(b, .997, 20, (1, 4))
        screen = np.zeros((4, 16, 64), np.uint8)
        for i in range(20):
            for buffer in [multi, single]:
                buffer.begin(screen, 20 + i)
                for j in range(3):
                    buffer.append(j, screen, j == 2, False)
        for aa, bb in zip(a.rows, b.rows, strict=True):
            for x, y in zip(aa, bb, strict=True):
                np.testing.assert_array_equal(x, y)

    def test_horizon_validation_and_cli(self):
        for invalid in [0, -1, 33, 1.5, True]:
            with self.assertRaises(ValueError):
                MultiOptionReturns(Sink(), invalid)
        from rl import defense_dqn
        for extra in [['--repeat-n-step', '5'], ['--repeat-n-step', '0'],
                      ['--learned-repeats', '1,4', '--n-step', '1', '--repeat-n-step', '33']]:
            with patch.object(sys, 'argv', ['dqn', '--run', '/nonexistent/option-test',
                    '--artifacts', '/nonexistent/option-artifacts'] + extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_learning_resume_and_policy_verification(self):
        from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/name),
                    '--artifacts', str(root/(name+'-artifacts'))] + extra, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            first = run('first', ['--learned-repeats', '1,4', '--n-step', '1', '--repeat-n-step', '5',
                '--envs', '2', '--capacity', '256', '--compact-replay', '--batch-size', '4', '--warmup', '16',
                '--steps', '128', '--eval-every', '1000', '--max-episode-steps', '12', '--train-every', '4',
                '--target-every', '2', '--mlx-cache-mb', '64', '--epsilon-final', '1'])
            second = run('resumed', ['--resume', str(root/'first/latest'), '--steps', '192'])
            self.assertGreater(first['updates'], 0)
            self.assertGreater(second['updates'], first['updates'])
            self.assertEqual(second['config']['repeat_n_step'], 5)
            for state in [first, second]:
                self.assertEqual(state['learned_repeat_returns_emitted'] + state['learned_repeat_returns_queued'],
                                 state['learned_repeat_completed'])
                self.assertGreater(state['learned_repeat_returns_emitted'], 0)
                self.assertGreater(state['learned_repeat_interrupted'], 0)
            checkpoint = root/'resumed/latest/model.safetensors'
            policy, config = load_policy(checkpoint)
            evaluation = evaluate(policy, [10000, 10001], envs=2)
            self.assertEqual(evaluation['complete_games'], 2)
            for game in evaluation['games']:
                recorded = record_game(policy, game['seed'], tstates=config['tstates'], max_steps=0)
                self.assertEqual(recorded[3], game)
            self.assertTrue(verify_policy_trace(checkpoint, *recorded[:3], recorded[3])['verified'])


if __name__ == '__main__':
    unittest.main()
