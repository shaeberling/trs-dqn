"""Optional training-only reward for a visibly live gameplay frame.

The displayed HUD is already used to establish lives and score. This adds
no hidden emulator state, action target or stage route, but it *is* reward
shaping and must be disclosed separately from score-only training.
"""

import numpy as np

from .defense import screen_info


def visible_alive_bonus(observation, beta, *, life_lost, terminal, truncated):
    if isinstance(beta, bool) or not np.isfinite(beta) or not 0 <= beta <= .5:
        raise ValueError("alive beta must be finite and in [0, 0.5]")
    if not beta or life_lost or terminal or truncated:
        return 0.
    observation = np.asarray(observation)
    if observation.shape != (4, 16, 64) or observation.dtype != np.uint8:
        raise ValueError("expected four raw Defense video frames")
    visible = screen_info(observation[-1])
    return float(beta) if visible["score"] is not None and visible["lives"] else 0.
