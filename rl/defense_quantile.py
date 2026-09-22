"""Fixed-quantile Double DQN; screen-only score-return distributions.

QR-DQN's pairwise quantile Huber loss (Dabney et al., arXiv:1710.10044).
Optional training-only power distortion follows the positive-eta formula in
arXiv:1806.06923, integrated over fixed quantile bins. This is not IQN:
Bellman action selection and all evaluation remain risk-neutral mean greedy.
Return dispersion is not an epistemic confidence interval or novelty reward.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .model import Learner, QNetwork


def distortion_weights(quantiles, power=0.):
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or not 2 <= quantiles <= 256:
        raise ValueError("quantiles must be an integer from 2 to 256")
    if isinstance(power, bool) or not np.isfinite(power) or not 0 <= power <= 4:
        raise ValueError("quantile exploration power must be finite and between 0 and 4")
    # If tau = U**(1/(1+eta)), its CDF is tau**(1+eta). Quantile
    # locations keep their trained indices; never sort predictions at action time.
    edges = np.linspace(0., 1., quantiles+1)
    return np.diff(edges**(1.+power)).astype(np.float32)


class QuantileQ(QNetwork):
    def __init__(self, action_count=20, quantiles=32):
        distortion_weights(quantiles)
        super().__init__(action_count=action_count)
        self.value = nn.Linear(256, quantiles)
        self.advantage = nn.Linear(256, action_count*quantiles)
        self._action_count = action_count
        self._quantiles = quantiles

    def quantile_values(self, screen):
        x = self.features(screen)
        a = self.advantage(x).reshape(-1, self._action_count, self._quantiles)
        return self.value(x)[:, None, :] + a - mx.mean(a, axis=1, keepdims=True)

    def __call__(self, screen):
        return mx.mean(self.quantile_values(screen), axis=2)


def quantile_huber(predicted, labels):
    """Per-transition sum_i mean_j rho_tau_i(label_j - prediction_i), kappa=1."""
    n = predicted.shape[-1]
    tau = (mx.arange(n, dtype=mx.float32)+.5)/n
    delta = labels[:, None, :] - predicted[:, :, None]
    absolute = mx.abs(delta)
    huber = mx.where(absolute <= 1., .5*delta*delta, absolute-.5)
    asymmetric = mx.abs(tau[None, :, None] - mx.stop_gradient((delta < 0).astype(mx.float32)))
    return mx.sum(mx.mean(asymmetric*huber, axis=2), axis=1)


class QuantileLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, quantiles=32,
                 exploration_power=0.):
        weights = distortion_weights(quantiles, exploration_power)
        mx.random.seed(seed)
        self.online = QuantileQ(action_count, quantiles)
        self.target = QuantileQ(action_count, quantiles)
        self.target.update(tree_map(lambda x: mx.array(x), self.online.parameters()))
        self.optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.optimizer.init(self.online.trainable_parameters())
        self.state = [self.online.state, self.target.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1),
                                  inputs=self.online.state)
        self._distortion = mx.array(weights)
        self._exploration_power = exploration_power
        self.predict_exploration = mx.compile(
            lambda x: mx.argmax(mx.sum(self.online.quantile_values(x)*self._distortion, axis=2), axis=1),
            inputs=self.online.state)

    def _loss(self, model, obs, actions, returns, next_obs, discounts, weights):
        # Double-Q uses ONLINE risk-neutral means to select the next action,
        # then TARGET quantiles for that action. Exploration power is absent.
        next_actions = mx.argmax(model(next_obs), axis=1)
        target = mx.take_along_axis(self.target.quantile_values(next_obs),
                                    next_actions[:, None, None], axis=1)[:, 0, :]
        labels = mx.stop_gradient(returns[:, None] + discounts[:, None]*target)
        predicted = mx.take_along_axis(model.quantile_values(obs),
                                       actions[:, None, None], axis=1)[:, 0, :]
        per_transition = quantile_huber(predicted, labels)
        # Nonnegative distributional loss gives PER a signal even if signed
        # mean residuals cancel. Divide by N for priority scale, not training loss.
        priority = per_transition / predicted.shape[-1]
        return mx.mean(weights*per_transition), (priority, mx.mean(predicted))

    def actions(self, observations):
        if self._exploration_power == 0:
            return super().actions(observations)
        return np.array(self.predict_exploration(mx.array(observations)))

    def initialize_scalar(self, checkpoint):
        """Transfer own scalar Q weights to initially coincident quantiles.

        Fresh initialization only: online and target both receive parent online
        weights; Adam, counters and replay are new. No assumed return spread.
        Environment/provenance compatibility is checked by the trainer.
        """
        if int(self.optimizer.state['step'].item()) != 0:
            raise ValueError('scalar initialization requires a fresh optimizer')
        parent = QNetwork(self.online._action_count)
        parent.load_weights(str(checkpoint))
        self.online.conv = parent.conv
        self.online.hidden = parent.hidden
        n = self.online._quantiles
        for name in ('value', 'advantage'):
            source, target = getattr(parent, name), getattr(self.online, name)
            target.weight = mx.repeat(source.weight, n, axis=0)
            target.bias = mx.repeat(source.bias, n, axis=0)
        self.sync_target()
        self.state = [self.online.state, self.target.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        # Rebind compiled inference after replacing encoder modules.
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1), inputs=self.online.state)
        self.predict_exploration = mx.compile(
            lambda x: mx.argmax(mx.sum(self.online.quantile_values(x)*self._distortion, axis=2), axis=1),
            inputs=self.online.state)
