"""Own-experience temporal consistency, adapted from Pohlen et al. (2018).

Only the next-value regularizer is used, not their demonstration/imitation
system or value transform. Our n-step Double-Q bootstrap action is selected by
the online model; its saved-target value is fixed. Learning-terminal rows are
masked because they have no bootstrap. No reward or acting changes are made.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import Learner


def consistency_loss(online_next, target_next, discounts, weights):
    # The choice is discrete; gradients flow through its selected value only.
    selected = mx.stop_gradient(mx.argmax(online_next, axis=1, keepdims=True))
    current = mx.take_along_axis(online_next, selected, axis=1)[:, 0]
    reference = mx.stop_gradient(mx.take_along_axis(target_next, selected, axis=1)[:, 0])
    error = current - reference
    absolute = mx.abs(error)
    huber = mx.where(absolute < 1., .5*error*error, absolute-.5)
    active = (discounts > 0).astype(online_next.dtype)
    return mx.mean(weights*active*huber), mx.mean(active*absolute), mx.mean(active)


class ConsistencyLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, weight=1.):
        if not np.isfinite(weight) or not 0 < weight <= 10:
            raise ValueError('consistency weight must be finite and in (0,10]')
        self.weight = float(weight)
        self.local_updates = 0
        self.last_td = self.last_consistency = self.last_gap = self.last_active = 0.
        super().__init__(learning_rate, seed, action_count)

    def _joint_loss(self, model, obs, actions, returns, following, discounts, weights):
        td, (errors, q) = Learner._loss(self, model, obs, actions, returns, following, discounts, weights)
        consistency, gap, active = consistency_loss(model(following), self.target(following), discounts, weights)
        return td+self.weight*consistency, (errors, q, td, consistency, gap, active)

    def _update(self, *batch):
        (loss, (errors, q, td, consistency, gap, active)), grads = nn.value_and_grad(
            self.online, self._joint_loss)(self.online, *batch)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.)
        self.optimizer.update(self.online, grads)
        return loss, errors, q, norm, td, consistency, gap, active

    def train(self, batch):
        result = self.update(*(mx.array(x) for x in batch))
        mx.eval(result, self.state)
        self.local_updates += 1
        self.last_td, self.last_consistency, self.last_gap, self.last_active = (
            float(result[i].item()) for i in range(4, 8))
        # Priorities remain absolute TD errors only, not auxiliary residuals.
        return float(result[0].item()), np.array(result[1]), float(result[2].item())

    def consistency_stats(self):
        return dict(local_updates=self.local_updates, weight=self.weight,
                    last_weighted_td_loss=self.last_td,
                    last_weighted_consistency_loss=self.last_consistency,
                    last_mean_masked_absolute_next_gap=self.last_gap,
                    last_bootstrapping_fraction=self.last_active)
