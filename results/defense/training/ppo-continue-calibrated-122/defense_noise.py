"""Training-only Gaussian perturbations of the learned actor's output layer.

One independent draw per worker, retained until a visible life/episode boundary.
No game data, action labels, backend initialization or controller is involved.
"""

import operator

import numpy as np


class PolicyParameterNoise:
    def __init__(self, envs, shape, std, rng):
        envs = operator.index(envs)
        shape = tuple(operator.index(n) for n in shape)
        if not shape or min(envs, *shape) < 1 or not np.isfinite(std) or std < 0:
            raise ValueError("invalid policy parameter noise configuration")
        self.std, self.rng = float(std), rng
        self.values = np.zeros((envs, *shape), np.float32)
        self.draws = 0
        self.redraw(np.ones(envs, dtype=bool))

    def redraw(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (len(self.values),) or boundaries.dtype != np.bool_:
            raise ValueError("noise boundaries must be one boolean per worker")
        count = int(boundaries.sum())
        if count and self.std:
            self.values[boundaries] = self.rng.normal(0, self.std, (count, *self.values.shape[1:]))
            self.draws += count


class PolicyBiasNoise(PolicyParameterNoise):
    def __init__(self, envs, actions, std, rng):
        super().__init__(envs, (actions,), std, rng)

    @property
    def bias(self):
        return self.values


class PolicyWeightNoise(PolicyParameterNoise):
    def __init__(self, envs, actions, features, std, rng):
        super().__init__(envs, (actions, features), std, rng)


class NoiseRollout:
    """Store each life draw once, with time-major worker IDs for PPO samples."""

    def __init__(self, noise):
        self.noise = noise
        self.bank = [row.copy() for row in noise.values]
        self.current = np.arange(len(noise.values), dtype=np.int32)
        self.records = []

    def record(self):
        self.records.append(self.current.copy())
        return self.noise.values

    def redraw(self, boundaries):
        self.noise.redraw(boundaries)
        for worker in np.flatnonzero(boundaries):
            self.current[worker] = len(self.bank)
            self.bank.append(self.noise.values[worker].copy())

    def arrays(self):
        if not self.records:
            raise ValueError("cannot finalize an empty noise rollout")
        return np.stack(self.bank), np.concatenate(self.records)
