"""Screen-only bootstrapped Double DQN with optional fixed random priors.

Shared convolutional features, independent hidden/dueling heads. Priors are
random networks, never game knowledge, demonstrations or reward bonuses.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .model import Learner, render_table


class MultiHeadQ(nn.Module):
    def __init__(self, action_count, heads):
        super().__init__()
        self.conv = [nn.Conv2d(8, 16, 5, 2, 2), nn.Conv2d(16, 32, 3, 2, 1),
                     nn.Conv2d(32, 32, 3, 2, 1)]
        self.hidden = [nn.Linear(32*6*16, 256) for _ in range(heads)]
        self.value = [nn.Linear(256, 1) for _ in range(heads)]
        self.advantage = [nn.Linear(256, action_count) for _ in range(heads)]
        self._table = mx.array(render_table())

    def __call__(self, screen):
        x = self._table[screen.astype(mx.int32)]
        x = x.transpose(0, 2, 4, 3, 5, 1, 6).reshape(-1, 48, 128, 8)
        for layer in self.conv:
            x = nn.relu(layer(x))
        x = x.reshape(x.shape[0], -1)
        values = []
        for hidden, value, advantage in zip(self.hidden, self.value, self.advantage, strict=True):
            h = nn.relu(hidden(x))
            a = advantage(h)
            values.append(value(h)+a-mx.mean(a, axis=1, keepdims=True))
        return mx.stack(values, axis=1)  # batch, head, action


class BootstrapQ(nn.Module):
    def __init__(self, action_count=20, heads=5, prior_scale=1.):
        super().__init__()
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 2:
            raise ValueError("bootstrap requires at least two heads")
        if not np.isfinite(prior_scale) or prior_scale < 0:
            raise ValueError("prior scale must be finite and nonnegative")
        self.learned = MultiHeadQ(action_count, heads)
        self.prior = MultiHeadQ(action_count, heads)
        self.prior.freeze()
        self._prior_scale = prior_scale

    def head_values(self, screen):
        return self.learned(screen) + self._prior_scale*mx.stop_gradient(self.prior(screen))

    def __call__(self, screen):
        # Fixed evaluation policy: greedy mean of the saved ensemble + priors.
        return mx.mean(self.head_values(screen), axis=1)


class BootstrapLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, heads=5, prior_scale=1.):
        mx.random.seed(seed)
        self.online = BootstrapQ(action_count, heads, prior_scale)
        self.target = BootstrapQ(action_count, heads, prior_scale)
        self.target.update(tree_map(lambda x: mx.array(x), self.online.parameters()))
        self.optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.optimizer.init(self.online.trainable_parameters())
        self.state = [self.online.state, self.target.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1),
                                  inputs=self.online.state)
        self.predict_heads = mx.compile(
            lambda x: mx.argmax(self.online.head_values(x), axis=2), inputs=self.online.state)

    def _loss(self, model, obs, actions, returns, next_obs, discounts, weights, masks):
        next_actions = mx.argmax(model.head_values(next_obs), axis=2, keepdims=True)
        target_q = mx.take_along_axis(self.target.head_values(next_obs), next_actions, axis=2)[:, :, 0]
        labels = mx.stop_gradient(returns[:, None]+discounts[:, None]*target_q)
        q = mx.take_along_axis(model.head_values(obs), actions[:, None, None], axis=2)[:, :, 0]
        absolute = mx.abs(q-labels)
        huber = mx.where(absolute < 1, .5*absolute*absolute, absolute-.5)
        loss = mx.sum(weights[:, None]*masks*huber)/mx.maximum(mx.sum(masks), 1.)
        # One shared priority per transition, covering disagreement in all heads.
        return loss, (mx.mean(absolute, axis=1), mx.mean(q))

    def _update(self, obs, actions, returns, next_obs, discounts, weights, masks):
        (loss, (errors, q)), grads = nn.value_and_grad(self.online, self._loss)(
            self.online, obs, actions, returns, next_obs, discounts, weights, masks)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.)
        self.optimizer.update(self.online, grads)
        return loss, errors, q, norm

    def actions(self, observations, heads=None):
        if heads is None:
            return super().actions(observations)
        choices = np.array(self.predict_heads(mx.array(observations)))
        heads = np.asarray(heads)
        if (heads.shape != (len(observations),) or heads.dtype.kind not in "iu"
                or np.any(heads < 0) or np.any(heads >= choices.shape[1])):
            raise ValueError("one valid bootstrap head required per observation")
        return choices[np.arange(len(choices)), heads]
