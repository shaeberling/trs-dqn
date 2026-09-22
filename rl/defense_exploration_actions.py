"""Optional fixed training exploration distribution; never an acting policy.

Stage 1 compares the whole movement byte: Space+arrow aliases forward fire.
Balance twelve command groups, retaining all twenty individual actions with
positive probability. This is fixed at setup, never conditioned on a screen,
stage, score, position or collision. Later stages need not share these aliases.
"""

import numpy as np


def action_probabilities(mode, action_count):
    if mode == 'uniform':
        return None  # Preserve original integer draws and exact RNG behavior.
    if mode != 'stage1-balanced' or action_count != 20:
        raise ValueError('stage1-balanced exploration requires the standard 20 actions')
    weights = np.ones(20, dtype=np.float64)
    weights[9:18] = 1/9
    return weights/weights.sum()
