"""Training-only random action persistence, with no screen or game-state input.

Adapted from temporally extended epsilon-greedy (Dabney et al., 2021).
Durations use a bounded power law. Epsilon specifies the nominal fraction of
exploratory *steps*, not option starts; life/episode cuts can reduce that fraction.
"""

import operator

import numpy as np


def duration_distribution(max_repeat, exponent):
    if isinstance(max_repeat, bool):
        raise ValueError('max repeat must be an integer')
    try:
        maximum = operator.index(max_repeat)
    except TypeError as error:
        raise ValueError('max repeat must be an integer') from error
    if not 1 <= maximum <= 1024 or not np.isfinite(exponent) or exponent <= 1:
        raise ValueError('requires max repeat in 1..1024 and finite exponent > 1')
    lengths = np.arange(1, maximum+1, dtype=np.int32)
    probabilities = lengths.astype(np.float64)**-exponent
    probabilities /= probabilities.sum()
    return lengths, probabilities


def start_probability(epsilon, mean_duration):
    """Renewal occupancy: e = p*E[L] / (1-p+p*E[L])."""
    if (not np.isfinite(epsilon) or not 0 <= epsilon <= 1
            or not np.isfinite(mean_duration) or mean_duration < 1):
        raise ValueError('invalid epsilon or mean duration')
    return float(epsilon/(mean_duration*(1-epsilon)+epsilon))


def worker_epsilons(epsilon, envs, boot_envs, boot_epsilon=None):
    """Optional fixed worker-role rates; never consult observations or scores.

    The disabled path returns the original scalar without changing RNG use.
    Warmup is handled separately by the trainer and remains uniform random.
    """
    if boot_epsilon is None:
        return epsilon
    if (not 0 < boot_envs < envs or not np.isfinite(boot_epsilon)
            or not 0 <= boot_epsilon <= 1 or not np.isfinite(epsilon)
            or not 0 <= epsilon <= 1):
        raise ValueError('worker epsilon override requires valid rates and reserved boot workers')
    rates = np.full(envs, epsilon, dtype=np.float64)
    rates[:boot_envs] = boot_epsilon
    return rates


class PersistentExploration:
    def __init__(self, envs, action_count, max_repeat, exponent, rng, action_probabilities=None):
        self.lengths, self.probabilities = duration_distribution(max_repeat, exponent)
        self.envs, self.action_count = operator.index(envs), operator.index(action_count)
        if self.envs < 1 or self.action_count < 1:
            raise ValueError('positive worker and action counts required')
        self.action_probabilities = None
        if action_probabilities is not None:
            probabilities = np.asarray(action_probabilities, dtype=np.float64)
            if (probabilities.shape != (self.action_count,) or not np.isfinite(probabilities).all()
                    or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0)):
                raise ValueError('requires positive normalized probabilities for every action')
            self.action_probabilities = probabilities.copy()
        self.rng = rng
        self.mean_duration = float(self.lengths@self.probabilities)
        self.remaining = np.zeros(self.envs, np.int32)
        self.held = np.zeros(self.envs, np.int32)
        self.starts = self.exploratory_steps = self.decisions = self.cancelled_steps = 0
        self.duration_counts = np.zeros(len(self.lengths), np.int64)
        self.action_counts = np.zeros(self.action_count, np.int64)
        self.worker_exploratory_steps = np.zeros(self.envs, np.int64)

    def select(self, greedy, epsilon):
        greedy = np.asarray(greedy)
        if (greedy.shape != (self.envs,) or not np.issubdtype(greedy.dtype, np.integer)
                or np.any(greedy < 0) or np.any(greedy >= self.action_count)):
            raise ValueError('requires one valid learned greedy action per worker')
        idle = np.flatnonzero(self.remaining == 0)
        rates = np.asarray(epsilon)
        if rates.ndim == 0:
            probability = start_probability(epsilon, self.mean_duration)
        else:
            if (rates.shape != (self.envs,) or not np.all(np.isfinite(rates))
                    or np.any(rates < 0) or np.any(rates > 1)):
                raise ValueError('requires one valid epsilon per worker')
            probability = rates[idle]/(self.mean_duration*(1-rates[idle])+rates[idle])
        starting = idle[self.rng.random(len(idle)) < probability]
        if len(starting):
            duration = self.rng.choice(self.lengths, size=len(starting), p=self.probabilities)
            self.held[starting] = (self.rng.integers(self.action_count, size=len(starting))
                                  if self.action_probabilities is None else
                                  self.rng.choice(self.action_count, size=len(starting), p=self.action_probabilities))
            self.remaining[starting] = duration
            self.duration_counts += np.bincount(duration-1, minlength=len(self.lengths))
            self.starts += len(starting)
        exploring = self.remaining > 0
        selected = greedy.copy()
        selected[exploring] = self.held[exploring]
        self.action_counts += np.bincount(selected[exploring], minlength=self.action_count)
        self.exploratory_steps += int(exploring.sum())
        self.worker_exploratory_steps += exploring
        self.decisions += self.envs
        self.remaining[exploring] -= 1
        return selected

    def reset(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != (self.envs,) or boundaries.dtype != np.bool_:
            raise ValueError('requires one boolean boundary per worker')
        self.cancelled_steps += int(self.remaining[boundaries].sum())
        self.remaining[boundaries] = 0

    def stats(self):
        return dict(decisions=self.decisions, starts=self.starts,
                    exploratory_steps=self.exploratory_steps,
                    worker_exploratory_steps=self.worker_exploratory_steps.tolist(),
                    realized_exploratory_fraction=self.exploratory_steps/self.decisions if self.decisions else 0.,
                    expected_uninterrupted_duration=self.mean_duration,
                    cancelled_future_steps=self.cancelled_steps,
                    sampled_duration_counts=self.duration_counts.tolist(),
                    exploratory_action_counts=self.action_counts.tolist())
