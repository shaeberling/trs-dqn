"""Learned action-duration execution and own-experience semi-Markov returns.

The network chooses a joint (ordinary action, duration) option. Execution only
counts down that choice; it never inspects the screen, emulator or a route.
All observations remain at the original base-action cadence. Boundary signals
are the same visible life/episode boundaries used by the ordinary learner.
"""

import operator

import numpy as np


DEFAULT_DURATIONS = (1, 4, 16, 64)


def validate_spec(action_count, durations):
    if isinstance(action_count, bool):
        raise ValueError('positive integer action count required')
    try:
        action_count = operator.index(action_count)
        durations = tuple(durations)
        if any(isinstance(d, bool) for d in durations):
            raise ValueError('durations must be integers')
        durations = tuple(operator.index(d) for d in durations)
    except TypeError as error:
        raise ValueError('integer action count and durations required') from error
    if (action_count < 1 or not durations or durations[0] != 1
            or tuple(sorted(set(durations))) != durations or durations[-1] > 256):
        raise ValueError('durations must be unique increasing integers starting at 1, at most 256')
    return action_count, durations


class RepeatedActions:
    """One learned choice per option; no exploration or environment dependency."""
    def __init__(self, envs, action_count=20, durations=DEFAULT_DURATIONS):
        self.action_count, self.durations = validate_spec(action_count, durations)
        if isinstance(envs, bool) or not isinstance(envs, int) or envs < 1:
            raise ValueError('positive integer worker count required')
        self.remaining = np.zeros(envs, np.int32)
        self.options = np.zeros(envs, np.int32)
        self.starts = np.zeros(action_count * len(self.durations), np.int64)
        self.executed = np.zeros_like(self.starts)
        self.cancelled_steps = 0

    def select(self, choices):
        choices = np.asarray(choices)
        if (choices.shape != self.remaining.shape or not np.issubdtype(choices.dtype, np.integer)
                or np.any(choices < 0) or np.any(choices >= len(self.starts))):
            raise ValueError('one valid option choice required per worker')
        idle = self.remaining == 0
        self.options[idle] = choices[idle]
        self.remaining[idle] = np.asarray(self.durations)[choices[idle] // self.action_count]
        self.starts += np.bincount(choices[idle], minlength=len(self.starts))
        self.executed += np.bincount(self.options, minlength=len(self.starts))
        self.remaining -= 1
        return self.options % self.action_count

    def reset(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != self.remaining.shape or boundaries.dtype != np.bool_:
            raise ValueError('one boolean boundary required per worker')
        self.cancelled_steps += int(self.remaining[boundaries].sum())
        self.remaining[boundaries] = 0

    def stats(self):
        return dict(option_starts=self.starts.tolist(), base_actions=self.executed.tolist(),
                    cancelled_steps=self.cancelled_steps,
                    decisions=int(self.starts.sum()), executed_base_actions=int(self.executed.sum()))


class OptionReturns:
    """Only actually executed own transitions; never imagined action outcomes."""
    def __init__(self, replay, gamma=.997, action_count=20, durations=DEFAULT_DURATIONS):
        self.action_count, self.durations = validate_spec(action_count, durations)
        if not np.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError('discount must be in (0,1)')
        self.replay, self.gamma = replay, float(gamma)
        self.pending = None
        self.completed = self.interrupted = 0

    def begin(self, observation, option):
        if self.pending is not None:
            raise ValueError('cannot replace a pending option')
        if (isinstance(option, bool) or not isinstance(option, (int, np.integer))
                or not 0 <= option < self.action_count * len(self.durations)):
            raise ValueError('invalid option')
        observation = np.asarray(observation)
        if observation.shape != (4, 16, 64) or observation.dtype != np.uint8:
            raise ValueError('requires the original four raw video frames')
        self.pending = (observation.copy(), int(option))
        self.steps = 0
        self.reward = 0.
        self.discount = 1.

    def append(self, reward, following, terminated, truncated):
        if self.pending is None:
            raise ValueError('a learned option must precede its experience')
        following = np.asarray(following)
        if (following.shape != (4, 16, 64) or following.dtype != np.uint8
                or not np.isfinite(reward)):
            raise ValueError('invalid observed transition')
        self.reward += self.discount * float(reward)
        self.discount *= self.gamma
        self.steps += 1
        first, option = self.pending
        duration = self.durations[option // self.action_count]
        if self.steps == duration or terminated or truncated:
            self.replay.add(first, option, self.reward, following, 0. if terminated else self.discount)
            self.completed += 1
            self.interrupted += int(self.steps < duration)
            self.pending = None
            return True
        return False


class LearnedRepeatPolicy:
    """Greedy joint values with independent counters per complete evaluation game.

State contains only the agent's previously chosen option and its countdown.
Game identity uses evaluation RNG objects, as in the existing recurrent policy;
no random numbers are consumed and no screen features are hand-selected.
"""
    def __init__(self, infer, action_count=20, durations=DEFAULT_DURATIONS):
        self.action_count, self.durations = validate_spec(action_count, durations)
        self.infer = infer
        self.reset_seed(0)

    def reset_seed(self, seed):
        self.memories = {}
        self.serial_keys = None

    def _choose(self, obs, keys):
        if len(obs) != len(keys) or len(set(keys)) != len(keys):
            raise ValueError('one distinct game identity required per observation')
        idle = [i for i, key in enumerate(keys) if self.memories.get(key, (0, 0))[1] == 0]
        if idle:
            values = np.asarray(self.infer(np.asarray(obs)[idle]))
            if (values.shape != (len(idle), self.action_count * len(self.durations))
                    or not np.isfinite(values).all()):
                raise ValueError('invalid learned action-duration values')
            # Shortest duration wins exact initial ties; no directional preference.
            for index, option in zip(idle, values.argmax(axis=1), strict=True):
                self.memories[keys[index]] = (int(option), self.durations[option // self.action_count])
        actions = []
        for key in keys:
            option, remaining = self.memories[key]
            actions.append(option % self.action_count)
            self.memories[key] = (option, remaining - 1)
        return np.asarray(actions, np.int32)

    def __call__(self, obs):
        if self.serial_keys is None:
            self.serial_keys = [object() for _ in obs]
        if len(self.serial_keys) != len(obs):
            raise ValueError('reset_seed before changing serial stream count')
        return self._choose(obs, self.serial_keys)

    def sample_with_rngs(self, obs, rngs):
        return self._choose(obs, rngs)

    def observe_boundaries(self, boundaries, rngs=None):
        keys = self.serial_keys if rngs is None else rngs
        boundaries = np.asarray(boundaries)
        if (keys is None or boundaries.shape != (len(keys),) or boundaries.dtype != np.bool_
                or len(set(keys)) != len(keys)):
            raise ValueError('one boolean boundary required per game')
        for key, ended in zip(keys, boundaries, strict=True):
            if ended:
                self.memories.pop(key, None)
