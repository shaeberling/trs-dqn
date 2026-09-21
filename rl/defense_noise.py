"""Training-only Gaussian perturbations of the learned actor's output bias.

One independent draw per worker, retained until a visible life/episode boundary.
No game data, action labels, backend initialization or controller is involved.
"""

import operator

import numpy as np


class PolicyBiasNoise:
    def __init__(self, envs, actions, std, rng):
        envs, actions = operator.index(envs), operator.index(actions)
        if min(envs, actions) < 1 or not np.isfinite(std) or std < 0:
            raise ValueError("invalid policy bias noise configuration")
        self.std, self.rng = float(std), rng
        self.bias = np.zeros((envs, actions), np.float32)
        self.draws = 0
        self.redraw(np.ones(envs, dtype=bool))

    def redraw(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (len(self.bias),) or boundaries.dtype != np.bool_:
            raise ValueError("noise boundaries must be one boolean per worker")
        count = int(boundaries.sum())
        if count and self.std:
            self.bias[boundaries] = self.rng.normal(0, self.std, (count, self.bias.shape[1]))
            self.draws += count
