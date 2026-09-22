"""Own visible paths plus the exact existing scalar DQN replay fields."""

import numpy as np

from .defense_trace_replay import TraceReplay
from .replay import Replay


class SprReplay(TraceReplay):
    def sample(self, batch_size, rng, beta):
        indices, base = Replay.sample(self, batch_size, rng, beta)
        following = self.frame_storage.following[
            indices[:, None]*self.horizon+np.arange(self.horizon)]
        # Preserve stored float32 n-step returns/discounts exactly; do not
        # rebuild them with different arithmetic inside the auxiliary learner.
        return indices, (*base, self.sequence_actions[indices].copy(), following)
