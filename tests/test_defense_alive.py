import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_alive import visible_alive_bonus
from tests.test_defense import hud


def observation(screen):
    return np.repeat(screen[None], 4, axis=0)


class DefenseAliveTests(unittest.TestCase):
    def test_only_visible_active_hud_steps_pay(self):
        active = observation(hud(b"2620 *"))
        invisible = observation(hud(b"PLAYER 1"))
        zero = observation(hud(b"2620"))
        args = dict(life_lost=False, terminal=False, truncated=False)
        self.assertEqual(visible_alive_bonus(active, .05, **args), .05)
        self.assertEqual(visible_alive_bonus(invisible, .05, **args), 0.)
        self.assertEqual(visible_alive_bonus(zero, .05, **args), 0.)
        for field in args:
            self.assertEqual(visible_alive_bonus(active, .05, **{**args, field: True}), 0.)
        self.assertEqual(visible_alive_bonus(active, 0., **args), 0.)
        for beta in (-1., .51, float('nan'), True):
            with self.assertRaises(ValueError):
                visible_alive_bonus(active, beta, **args)

    def test_native_training_and_reward_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [sys.executable, '-m', 'rl.defense_train', '--envs', '2',
                    '--rollout', '8', '--batch-size', '16', '--epochs', '1',
                    '--max-episode-steps', '3', '--eval-every', '100',
                    '--mlx-cache-mb', '128', '--life-terminal']
            def run(name, extra, success=True):
                result = subprocess.run(base + ['--run', str(root/name),
                                                '--artifacts', str(root/'artifacts'), *extra],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0 if success else 2,
                                 result.stdout+result.stderr)
            run('first', ['--steps', '16', '--alive-beta', '.05'])
            state = json.loads((root/'first/latest/state.json').read_text())
            self.assertEqual(state['config']['alive_beta'], .05)
            self.assertIn('surviving ship', state['config']['reward'])
            self.assertEqual(state['config']['evaluation_reward'],
                             'original displayed score only; no alive bonus or shaping')
            run('resumed', ['--resume', str(root/'first/latest'), '--steps', '32'])
            self.assertEqual(json.loads((root/'resumed/latest/state.json').read_text())['steps'], 32)
            run('changed', ['--resume', str(root/'first/latest'), '--steps', '32',
                            '--alive-beta', '.02'], success=False)
            run('mixed', ['--steps', '16', '--alive-beta', '.05',
                          '--novelty-beta', '.05'], success=False)


if __name__ == '__main__':
    unittest.main()
