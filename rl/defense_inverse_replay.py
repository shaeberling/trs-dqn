"""Exact original DQN fields plus the root action's actual next screen stack."""

from .defense_trace_replay import TraceReplay
from .replay import Replay


class InverseReplay(TraceReplay):
    def sample(self, batch_size, rng, beta):
        indices, base = Replay.sample(self, batch_size, rng, beta)
        # Not the n-step endpoint: inverse labels refer to the very first action.
        immediate = self.frame_storage.following[indices*self.horizon]
        return indices, (*base, immediate)
