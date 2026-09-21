"""Compiled MLX dueling Double DQN. Every network input is screen video RAM."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map


def render_table():
    # Exact graphics raster plus a second channel encoding visible ASCII cells.
    # This is a fixed screen encoding, not object detection or game-state input.
    table = np.zeros((256, 3, 2, 2), np.float32)
    for c in range(256):
        if 128 <= c <= 191:
            for bit in range(6):
                table[c, bit//2, bit % 2, 0] = (c >> bit) & 1
        elif 32 <= c < 127:
            table[c, :, :, 1] = c / 127.0
    return table


class QNetwork(nn.Module):
    def __init__(self, action_count=6):
        super().__init__()
        self.conv = [nn.Conv2d(8, 16, 5, 2, 2),
                     nn.Conv2d(16, 32, 3, 2, 1),
                     nn.Conv2d(32, 32, 3, 2, 1)]
        self.hidden = nn.Linear(32*6*16, 256)
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_count)
        # A private attribute is not a model parameter in MLX.
        self._table = mx.array(render_table())

    def features(self, screen):
        # B,T,Y,X -> B,Y,dy,X,dx,T,C -> B,48,128,8
        x = self._table[screen.astype(mx.int32)]
        x = x.transpose(0, 2, 4, 3, 5, 1, 6).reshape(-1, 48, 128, 8)
        for layer in self.conv:
            x = nn.relu(layer(x))
        return nn.relu(self.hidden(x.reshape(x.shape[0], -1)))

    def policy_value(self, screen):
        x = self.features(screen)
        return self.advantage(x), self.value(x)[:, 0]

    def __call__(self, screen):
        x = self.features(screen)
        a = self.advantage(x)
        return self.value(x) + a - mx.mean(a, axis=1, keepdims=True)


class Learner:
    def __init__(self, learning_rate=1e-4, seed=0):
        mx.random.seed(seed)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.update(tree_map(lambda x: mx.array(x), self.online.parameters()))
        self.optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.optimizer.init(self.online.trainable_parameters())
        self.state = [self.online.state, self.target.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1),
                                  inputs=self.online.state)

    def _loss(self, model, obs, actions, returns, next_obs, discounts, weights):
        next_actions = mx.argmax(model(next_obs), axis=1, keepdims=True)
        target_q = mx.take_along_axis(self.target(next_obs), next_actions, axis=1)[:, 0]
        labels = mx.stop_gradient(returns + discounts * target_q)
        q = mx.take_along_axis(model(obs), actions[:, None], axis=1)[:, 0]
        td = q - labels
        absolute = mx.abs(td)
        huber = mx.where(absolute < 1, 0.5*td*td, absolute-0.5)
        return mx.mean(weights*huber), (absolute, mx.mean(q))

    def _update(self, obs, actions, returns, next_obs, discounts, weights):
        (loss, (errors, q)), grads = nn.value_and_grad(self.online, self._loss)(
            self.online, obs, actions, returns, next_obs, discounts, weights)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.0)
        self.optimizer.update(self.online, grads)
        return loss, errors, q, norm

    def train(self, batch):
        result = self.update(*(mx.array(x) for x in batch))
        mx.eval(result, self.state)
        return float(result[0].item()), np.array(result[1]), float(result[2].item())

    def sync_target(self):
        self.target.update(tree_map(lambda x: mx.array(x), self.online.parameters()))
        mx.eval(self.target.state)

    def actions(self, observations):
        return np.array(self.predict(mx.array(observations)))
