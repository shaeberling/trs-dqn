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


class PolicyKeyNoise:
    """Symmetric per-life key-factor bias over the ordinary action profile.

    One Gaussian factor per physical key, plus separate NOOP/CONTINUE factors.
    A composite command receives the sum of its factors normalized to unit
    variance, so related commands explore coherently without favoring a
    particular key, screen position or route. Only the resulting action-logit
    bias is retained by PPO's existing rollout likelihood machinery.
    """

    FACTORS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "CONTINUE_PREVIOUS")

    def __init__(self, envs, actions, std, rng):
        from .defense import action_names
        from .defense_repeat_previous import POLICY_ACTION_NAMES

        if (isinstance(envs, bool) or isinstance(actions, bool)
                or not isinstance(envs, int) or envs < 1
                or actions not in (len(action_names(False)), len(POLICY_ACTION_NAMES))
                or isinstance(std, bool) or not np.isfinite(std) or std < 0):
            raise ValueError("invalid factorized-key noise configuration")
        names = action_names(False) if actions == 20 else POLICY_ACTION_NAMES
        matrix = np.zeros((actions, len(self.FACTORS)), np.float32)
        for index, name in enumerate(names):
            keys = (name,) if name in ("NOOP", "CONTINUE_PREVIOUS") else name.split("+")
            for key in keys:
                matrix[index, self.FACTORS.index(key)] = 1.
            matrix[index] /= np.sqrt(len(keys))
        self.matrix, self.std, self.rng = matrix, float(std), rng
        self.values = np.zeros((envs, actions), np.float32)
        self.draws = 0
        self.redraw(np.ones(envs, dtype=bool))

    def redraw(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (len(self.values),) or boundaries.dtype != np.bool_:
            raise ValueError("noise boundaries must be one boolean per worker")
        count = int(boundaries.sum())
        if count and self.std:
            factors = self.rng.normal(0, self.std, (count, len(self.FACTORS)))
            self.values[boundaries] = factors @ self.matrix.T
            self.draws += count


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
