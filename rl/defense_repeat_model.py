"""Joint action-duration DQN with optional transfer from its own scalar Q model."""

import mlx.core as mx

from .defense_repeat import validate_spec
from .model import Learner, QNetwork


class RepeatLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, durations=(1, 4, 16, 64)):
        self.base_actions, self.durations = validate_spec(action_count, durations)
        super().__init__(learning_rate, seed, self.base_actions * len(self.durations))

    def initialize_scalar(self, checkpoint):
        """Copy own encoder/value; tile advantages, with fresh Adam and replay.

        This is value-based initialization, not a claim that different-duration
        options truly have identical returns. Their values subsequently learn
        from actually executed discounted score increments.
        """
        if int(self.optimizer.state['step'].item()) != 0:
            raise ValueError('scalar initialization requires fresh Adam')
        parent = QNetwork(self.base_actions)
        parent.load_weights(str(checkpoint))
        self.online.conv = parent.conv
        self.online.hidden = parent.hidden
        self.online.value = parent.value
        n = len(self.durations)
        self.online.advantage.weight = mx.concatenate([parent.advantage.weight] * n, axis=0)
        self.online.advantage.bias = mx.concatenate([parent.advantage.bias] * n, axis=0)
        self.sync_target()
        self.state = [self.online.state, self.target.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1), inputs=self.online.state)
