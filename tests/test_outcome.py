import hashlib
import unittest
from pathlib import Path

import numpy as np

from rl.env import screen_info
from rl.evaluate import summary, level_rank, level_target_met
from rl.outcome import OUTCOME_VERSION, screen_outcome, verified_win
from rl.supervise import learner_arguments
from trs.cmd import CMD


class OutcomeTests(unittest.TestCase):
    def game(self, level=8, reserves=1, terminal=True):
        # Synthetic screens only: these fixtures are not gameplay evidence.
        frame = np.full((16, 64), 128, np.uint8)
        frame[0, 6:11] = list(b'00500')
        frame[0, 59:64] = list(f'{level:05d}'.encode())
        if reserves:
            frame[0, 35-reserves:35] = 0x95
        if terminal:
            frame[10, 28:37] = list(b'GAME\x80OVER')
        info = screen_info(frame)
        return dict(**info, **screen_outcome(frame, info), terminated=terminal,
                    truncated=False, steps=100, full_game=True)

    def test_final_level_with_any_visible_reserve_proves_win(self):
        for reserve in range(1, 6):
            game = self.game(reserves=reserve)
            self.assertTrue(verified_win(game))
            self.assertEqual(game['reserve_balls_visible'], reserve)
            self.assertEqual(game['outcome_version'], OUTCOME_VERSION)

    def test_last_ball_final_level_is_unverified_not_a_loss(self):
        game = self.game(reserves=0)
        self.assertFalse(verified_win(game))
        self.assertFalse(game['game_won'])
        self.assertEqual(game['win_status'], 'unverified_final_level')

    def test_reaching_level_eight_and_other_terminal_levels_are_not_wins(self):
        self.assertFalse(verified_win(self.game(terminal=False)))
        for level in [1, 7, 9, 10]:
            self.assertFalse(verified_win(self.game(level=level)))

    def test_practice_truncation_missing_evidence_cannot_count(self):
        game = self.game()
        for override in [dict(full_game=False), dict(truncated=True), dict(terminated=False),
                         dict(outcome_version='unknown'), dict(reserve_balls_visible=0),
                         dict(win_status='unverified_final_level')]:
            self.assertFalse(verified_win({**game, **override}))
        self.assertFalse(verified_win(dict(level=8, score=999, terminated=True)))

    def test_summary_wins_unknowns_legacy_and_selection(self):
        win, unknown = self.game(), self.game(reserves=0)
        result = summary([win, unknown, self.game(level=3)])
        self.assertEqual(result['verified_wins'], 1)
        self.assertEqual(result['unverified_final_level_games'], 1)
        self.assertEqual(result['outcome_assessed_games'], 3)
        self.assertEqual(result['verified_win_rate'], 1/3)
        self.assertTrue(level_target_met(result, 8, 1, game_win=True))
        self.assertFalse(level_target_met(summary([unknown]), 8, 1, game_win=True))
        legacy = summary([dict(level=8, score=999, terminated=True)])
        self.assertNotIn('outcome_version', legacy)
        self.assertFalse(level_target_met(legacy, 8, 1, game_win=True))
        self.assertGreater(level_rank(result, 8, game_win=True),
                           level_rank(legacy, 8, game_win=True))
        mixed = summary([win, dict(level=8, score=999, terminated=True)])
        self.assertEqual(mixed['missing_outcome_games'], 1)
        incomplete = summary([win, {**win, 'full_game': False}])
        self.assertEqual(incomplete['verified_wins'], 1)
        self.assertFalse(level_target_met(incomplete, 8, 1, game_win=True))

    def test_supervised_winner_keeps_learning_until_full_audit(self):
        args = learner_arguments('run', 'source', 4, game_win=True)
        self.assertIn('--target-game-win', args)
        self.assertIn('--no-stop-on-target', args)
        self.assertEqual(args[args.index('--target-level')+1], 8)
        self.assertEqual(args[args.index('--steps')+1], 0)

    def test_original_binary_victory_and_loss_contract_readonly(self):
        # Never instantiate or mutate the emulator: decode a separate buffer.
        class Image:
            def __init__(self):
                self.data = bytearray(65536)

            def poke(self, address, value):
                self.data[address] = value

        path = Path('var/breakdown.cmd')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '0eada4c36fbdb135ca14f390cdb3e01a3a3da121a1328a531fcc6c1e2e931ee6')
        image = Image()
        self.assertEqual(CMD(image).load(str(path)), 0x5200)
        spans = {
            # Both loss paths: load lives, decrement, store, compare zero,
            # GAME OVER if zero; otherwise redraw reserve HUD.
            0x5c98: '3a 57 65 3d 32 57 65 fe 00 ca 17 5d cd ab 57',
            0x5c4b: '3a 57 65 3d 32 57 65 fe 00 ca 17 5d cd ab 57',
            # Reserve renderer uses lives-1 and columns30..34 (0x3c1e).
            0x57ab: 'c5 e5 3a 57 65 3d 4f 06 05 21 1e 3c',
            0x57c3: '36 95 18 f6',
            # Final-level victory jumps straight to GAME OVER, no decrement.
            0x5869: '3a 16 63 3c 32 16 63 fe 09 ca 17 5d cd 3b 57',
        }
        for address, expected in spans.items():
            expected = bytes.fromhex(expected)
            self.assertEqual(image.data[address:address+len(expected)], expected)


if __name__ == '__main__':
    unittest.main()
