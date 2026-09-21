"""Persistent bootstrap memberships for own n-step experience; no MLX import."""

import numpy as np

from .replay import Replay


class BootstrapReplay(Replay):
    def __init__(self, capacity, heads, probability, rng):
        if (isinstance(heads, bool) or not isinstance(heads, int) or heads < 2
                or not np.isfinite(probability) or not 0 < probability <= 1):
            raise ValueError("invalid bootstrap membership configuration")
        super().__init__(capacity)
        self.masks = np.empty((capacity, heads), np.float32)
        self._probability, self._rng = probability, rng

    def add(self, obs, action, reward, next_obs, discount):
        # Draw once at insertion, never at sampling. Empty memberships are legal.
        self.masks[self.pos] = self._rng.random(self.masks.shape[1]) < self._probability
        super().add(obs, action, reward, next_obs, discount)

    def sample(self, batch_size, rng, beta):
        indices, batch = super().sample(batch_size, rng, beta)
        return indices, (*batch, self.masks[indices])
