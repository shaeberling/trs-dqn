"""Conservative, screen-only victory evidence for the original eight-level game.

Both loss branches decrement lives to zero before GAME OVER. On the last
life the reserve HUD was already empty; victory leaves reserves unchanged.
Thus GAME OVER at level 8 with a visible reserve proves a win. A last-ball
level-8 ending is ambiguous: never report it as a loss or a verified win.
This module reads only the same video bytes available to the policy.
"""

import numpy as np


OUTCOME_VERSION = "original-eight-reserve-proof-v1"


def screen_outcome(screen, info):
    reserves = int(np.count_nonzero(np.asarray(screen)[0, 30:35] == 0x95))
    terminal = bool(info['game_over'])
    final_level = info['level'] == 8
    won = terminal and final_level and reserves > 0
    status = ('verified_win' if won else 'unverified_final_level'
              if terminal and final_level else 'loss' if terminal else 'ongoing')
    return dict(outcome_version=OUTCOME_VERSION, reserve_balls_visible=reserves,
                game_won=won, win_status=status)


def verified_win(game):
    """Only a complete from-boot game can count toward the user's goal."""
    return bool(game.get('game_won') is True
                and game.get('outcome_version') == OUTCOME_VERSION
                and game.get('win_status') == 'verified_win'
                and 1 <= game.get('reserve_balls_visible', 0) <= 5
                and game.get('level') == 8 and game.get('game_over') is True
                and game.get('terminated') is True and not game.get('truncated', False)
                and game.get('full_game', True))
