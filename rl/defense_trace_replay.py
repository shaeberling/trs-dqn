"""Own visible trajectories for recomputing greedy trace cuts at sampling time.

No collection-time greedy labels, behavior probabilities or native snapshots
enter replay. Each worker retains the existing life/episode queue boundaries.
"""

import operator

import numpy as np

from .frame_storage import FrameTable
from .replay import NStep, Replay


class TraceStorage:
    def __init__(self, base, capacity, horizon):
        self.base, self.pool = base, base.pool
        self.obs, self.next_obs = base.obs, base.next_obs
        self.following = FrameTable(capacity*horizon, self.pool)

    def stats(self):
        result = self.base.stats()
        result['screen_index_bytes'] += self.following.indices.nbytes
        result['dense_equivalent_screen_bytes'] += len(self.following)*4*16*64
        return result


class TraceReplay(Replay):
    def __init__(self, capacity, horizon, action_count=20):
        if any(isinstance(v, (bool, np.bool_)) for v in (capacity, horizon, action_count)):
            raise ValueError('trace counts must be positive integers')
        capacity, horizon, action_count = map(operator.index, (capacity, horizon, action_count))
        if (not 1 <= horizon <= 32 or action_count < 1 or capacity < 1
                or capacity*4*(horizon+2) > np.iinfo(np.int32).max-128):
            raise ValueError('invalid trace capacity, action count or horizon (1..32)')
        super().__init__(capacity, compact=True)
        self.horizon, self.action_count = horizon, action_count
        self.frame_storage = TraceStorage(self.frame_storage, capacity, horizon)
        self.sequence_actions = np.full((capacity, horizon), -1, np.int32)
        self.sequence_rewards = np.zeros((capacity, horizon), np.float32)
        self.sequence_discounts = np.zeros((capacity, horizon), np.float32)

    def add_sequence(self, sequence, gamma):
        if not 1 <= len(sequence) <= self.horizon or not np.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError('invalid trace length or gamma')
        # Validate before changing ring references or priorities. Padding is
        # recognizable by action -1 and uses only the last actual observation.
        for j, (obs, action, reward, following, terminal) in enumerate(sequence):
            if (obs.shape != (4, 16, 64) or obs.dtype != np.uint8
                    or following.shape != (4, 16, 64) or following.dtype != np.uint8
                    or isinstance(action, (bool, np.bool_))
                    or not isinstance(action, (int, np.integer))
                    or not 0 <= action < self.action_count or not np.isfinite(reward)
                    or not isinstance(terminal, (bool, np.bool_))
                    or terminal and j != len(sequence)-1):
                raise ValueError('invalid own trace transition or crossed terminal')
        index = self.pos
        self.sequence_actions[index].fill(-1)
        self.sequence_rewards[index].fill(0)
        self.sequence_discounts[index].fill(0)
        total, discount = 0., 1.
        for j in range(self.horizon):
            following = sequence[min(j, len(sequence)-1)][3]
            self.frame_storage.following[index*self.horizon+j] = following
            if j < len(sequence):
                _, action, reward, _, terminal = sequence[j]
                self.sequence_actions[index, j] = action
                self.sequence_rewards[index, j] = reward
                self.sequence_discounts[index, j] = 0. if terminal else gamma
                total += discount*reward
                discount *= 0. if terminal else gamma
        # Preserve the existing prioritized ring and sampling implementation.
        super().add(sequence[0][0], sequence[0][1], total, sequence[-1][3], discount)

    def sample(self, batch_size, rng, beta):
        indices, base = super().sample(batch_size, rng, beta)
        following = self.frame_storage.following[
            indices[:, None]*self.horizon+np.arange(self.horizon)]
        return indices, (base[0], self.sequence_actions[indices].copy(),
                         self.sequence_rewards[indices].copy(), following,
                         self.sequence_discounts[indices].copy(), base[-1])


class TraceNStep(NStep):
    def _emit(self):
        self.replay.add_sequence(list(self.queue), self.gamma)
        self.queue.popleft()
