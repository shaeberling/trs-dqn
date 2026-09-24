"""Training-only novelty from this worker's own visible gameplay screen.

The cell omits the HUD and is reset at a visible life boundary. This is an
explicit auxiliary reward, never a policy input, action label or game oracle.
Complete-game evaluation still uses only the original displayed score/stage.
"""

import numpy as np

from .defense_cells import CELL_ENCODING, screen_cell


class LifeScreenNovelty:
    def __init__(self, observations, beta):
        if isinstance(beta, bool) or not np.isfinite(beta) or not 0 < beta <= .5:
            raise ValueError("novelty beta must be finite and in (0, 0.5]")
        observations = np.asarray(observations)
        if observations.ndim != 4 or observations.shape[1:] != (4, 16, 64):
            raise ValueError("expected a batch of four-frame Defense observations")
        self.beta = float(beta)
        self.seen = [{screen_cell(obs)} for obs in observations]
        self.hits = 0
        self.bonus_sum = 0.
        self.life_resets = 0

    def step(self, worker, observation, *, life_lost=False, reset=None):
        if not 0 <= worker < len(self.seen):
            raise ValueError("invalid worker")
        if reset is not None or life_lost:
            self.seen[worker] = {screen_cell(reset if reset is not None else observation)}
            self.life_resets += 1
            return 0.
        key = screen_cell(observation)
        if key in self.seen[worker]:
            return 0.
        self.seen[worker].add(key)
        self.hits += 1
        self.bonus_sum += self.beta
        return self.beta

    def metrics(self):
        return dict(encoding=CELL_ENCODING, beta=self.beta, first_visit_hits=self.hits,
                    bonus_sum=round(self.bonus_sum, 6), visible_life_resets=self.life_resets,
                    current_life_cells=[len(cells) for cells in self.seen])
