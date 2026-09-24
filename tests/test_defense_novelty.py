import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_cells import screen_cell
from rl.defense_novelty import LifeScreenNovelty


def observation(byte):
    frame = np.full((4, 16, 64), 128, dtype=np.uint8)
    frame[:, 5, 5] = byte
    return frame


class DefenseNoveltyTests(unittest.TestCase):
    def test_only_first_visible_cell_visit_per_worker_life_is_rewarded(self):
        first, second = observation(128), observation(191)
        self.assertNotEqual(screen_cell(first), screen_cell(second))
        bonus = LifeScreenNovelty(np.stack([first, first]), .05)
        self.assertEqual(bonus.step(0, first), 0.)
        self.assertEqual(bonus.step(0, second), .05)
        self.assertEqual(bonus.step(0, second), 0.)
        self.assertEqual(bonus.step(1, second), .05)
        self.assertEqual(bonus.step(0, second, life_lost=True), 0.)
        self.assertEqual(bonus.step(0, second), 0.)
        self.assertEqual(bonus.step(0, first), .05)
        self.assertEqual(bonus.step(0, second, reset=first), 0.)
        self.assertEqual(bonus.step(0, first), 0.)
        self.assertEqual(bonus.metrics()['first_visit_hits'], 3)
        self.assertEqual(bonus.metrics()['visible_life_resets'], 2)

    def test_hud_only_change_and_invalid_beta(self):
        first = observation(128)
        changed_hud = first.copy()
        changed_hud[:, 0, 0] = ord('4')
        self.assertEqual(screen_cell(first), screen_cell(changed_hud))
        bonus = LifeScreenNovelty(first[None], .02)
        self.assertEqual(bonus.step(0, changed_hud), 0.)
        for beta in (0, -.1, .51, float('nan'), True):
            with self.assertRaises(ValueError):
                LifeScreenNovelty(first[None], beta)

    def test_native_ppo_training_resume_and_reward_provenance(self):
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
            run('first', ['--steps', '16', '--novelty-beta', '.05'])
            state = json.loads((root/'first/latest/state.json').read_text())
            self.assertEqual(state['steps'], 16)
            self.assertEqual(state['config']['novelty_beta'], .05)
            self.assertIn('first-visit', state['config']['reward'])
            self.assertEqual(state['config']['evaluation_reward'],
                             'original displayed score only; no novelty bonus or shaping')
            run('resumed', ['--resume', str(root/'first/latest'), '--steps', '32'])
            self.assertEqual(json.loads((root/'resumed/latest/state.json').read_text())['steps'], 32)
            run('changed', ['--resume', str(root/'first/latest'), '--steps', '32',
                            '--novelty-beta', '.02'], success=False)


if __name__ == '__main__':
    unittest.main()
