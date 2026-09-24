"""Training-only Gaussian perturbations of the learned actor's output layer.

Draws are independent across workers. Optional fixed or randomly renewed
action-count intervals can redraw symmetric key factors within a life; they
read no game state and never select or override an action.
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
        return boundaries


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
    """Symmetric key-factor bias over the ordinary action profile.

    One Gaussian factor per physical key, plus separate NOOP/CONTINUE factors.
    A composite command receives the sum of its factors normalized to unit
    variance, so related commands explore coherently without favoring a
    particular key, screen position or route. Optional random renewal periods
    are worker-local and independent of game state. Only the resulting
    action-logit bias is retained by PPO's rollout likelihood machinery.
    """

    FACTORS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "CONTINUE_PREVIOUS")

    def __init__(self, envs, actions, std, rng, interval=0, interval_range=None):
        from .defense import action_names
        from .defense_repeat_previous import POLICY_ACTION_NAMES

        if (isinstance(envs, bool) or isinstance(actions, bool)
                or not isinstance(envs, int) or envs < 1
                or actions not in (len(action_names(False)), len(POLICY_ACTION_NAMES))
                or isinstance(std, bool) or not np.isfinite(std) or std < 0
                or isinstance(interval, bool) or not isinstance(interval, int) or interval < 0):
            raise ValueError("invalid factorized-key noise configuration")
        if interval_range is not None:
            if (not isinstance(interval_range, tuple) or len(interval_range) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in interval_range)
                    or not 1 <= interval_range[0] <= interval_range[1] or interval):
                raise ValueError("random key-noise interval requires a positive ordered pair and no fixed interval")
        names = action_names(False) if actions == 20 else POLICY_ACTION_NAMES
        matrix = np.zeros((actions, len(self.FACTORS)), np.float32)
        for index, name in enumerate(names):
            keys = (name,) if name in ("NOOP", "CONTINUE_PREVIOUS") else name.split("+")
            for key in keys:
                matrix[index, self.FACTORS.index(key)] = 1.
            matrix[index] /= np.sqrt(len(keys))
        self.matrix, self.std, self.rng = matrix, float(std), rng
        self.interval, self.interval_range = interval, interval_range
        self.values = np.zeros((envs, actions), np.float32)
        self.elapsed = np.zeros(envs, np.int32)
        self.periods = np.zeros(envs, np.int32)
        self.draws = 0
        self.redraw(np.ones(envs, dtype=bool))

    def redraw(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (len(self.values),) or boundaries.dtype != np.bool_:
            raise ValueError("noise boundaries must be one boolean per worker")
        self.elapsed += 1
        if self.interval_range is None:
            effective = boundaries | (self.interval > 0 and self.elapsed >= self.interval)
        else:
            effective = boundaries | (self.elapsed >= self.periods)
        count = int(effective.sum())
        if count and self.std:
            factors = self.rng.normal(0, self.std, (count, len(self.FACTORS)))
            self.values[effective] = factors @ self.matrix.T
            self.draws += count
        if count and self.interval_range is not None:
            low, high = self.interval_range
            self.periods[effective] = (self.rng.integers(low, high+1, size=count)
                                       if self.std else low)
        self.elapsed[effective] = 0
        return effective


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
        changed = self.noise.redraw(boundaries)
        for worker in np.flatnonzero(changed):
            self.current[worker] = len(self.bank)
            self.bank.append(self.noise.values[worker].copy())

    def arrays(self):
        if not self.records:
            raise ValueError("cannot finalize an empty noise rollout")
        return np.stack(self.bank), np.concatenate(self.records)
