"""Finite-horizon greedy-cut Double DQN, inspired by Watkins' trace cutting.

Greediness is recomputed using the CURRENT online network at sampled states.
This is a deep, lagged-target adaptation, not a tabular convergence claim.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import Learner


def trace_targets(online_next, target_next, actions, rewards, discounts):
    """Root action is unrestricted; stop before the first later deviation.

    next[:, i] is S_(i+1), hence its greedy action is compared with A_(i+1),
    never A_i. A padded action of -1 ends a partial/truncated path. Zero
    discount ends a real learning terminal without bootstrap.
    """
    greedy = mx.argmax(online_next, axis=-1)
    bootstrap = mx.take_along_axis(target_next, greedy[..., None], axis=-1)[..., 0]
    batch, horizon = actions.shape
    deviations = mx.concatenate([(actions[:, 1:] >= 0) &
                                  (actions[:, 1:] != greedy[:, :-1]),
                                  mx.zeros((batch, 1), mx.bool_)], axis=1)
    cuts = mx.concatenate([actions[:, 1:] != greedy[:, :-1],
                           mx.ones((batch, 1), mx.bool_)], axis=1)
    lengths = mx.argmax(cuts | (discounts == 0), axis=1)+1
    policy_cuts = (mx.take_along_axis(deviations, (lengths-1)[:, None], axis=1)[:, 0]
                   & (mx.take_along_axis(discounts, (lengths-1)[:, None], axis=1)[:, 0] != 0))
    labels = rewards[:, -1]+discounts[:, -1]*bootstrap[:, -1]
    for i in range(horizon-2, -1, -1):
        labels = rewards[:, i]+discounts[:, i]*mx.where(cuts[:, i], bootstrap[:, i], labels)
    return mx.stop_gradient(labels), lengths, policy_cuts


class TraceLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, horizon=5):
        self.backup_counts = np.zeros(horizon, np.int64)
        self.policy_cut_count = 0
        super().__init__(learning_rate, seed, action_count)

    def _loss(self, model, obs, actions, rewards, following, discounts, weights):
        batch, horizon = actions.shape
        flat = following.reshape(batch*horizon, *obs.shape[1:])
        online_next = model(flat).reshape(batch, horizon, -1)
        target_next = self.target(flat).reshape(batch, horizon, -1)
        labels, lengths, cuts = trace_targets(online_next, target_next, actions, rewards, discounts)
        q = mx.take_along_axis(model(obs), actions[:, :1], axis=1)[:, 0]
        absolute = mx.abs(q-labels)
        huber = mx.where(absolute < 1, .5*absolute*absolute, absolute-.5)
        return mx.mean(weights*huber), (absolute, mx.mean(q), lengths, cuts)

    def _update(self, obs, actions, rewards, following, discounts, weights):
        (loss, (errors, q, lengths, cuts)), grads = nn.value_and_grad(self.online, self._loss)(
            self.online, obs, actions, rewards, following, discounts, weights)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.)
        self.optimizer.update(self.online, grads)
        return loss, errors, q, norm, lengths, cuts

    def train(self, batch):
        result = self.update(*(mx.array(x) for x in batch))
        mx.eval(result, self.state)
        self.backup_counts += np.bincount(np.array(result[4])-1, minlength=len(self.backup_counts))
        self.policy_cut_count += int(np.array(result[5]).sum())
        return float(result[0].item()), np.array(result[1]), float(result[2].item())

    def trace_stats(self):
        total = int(self.backup_counts.sum())
        return dict(sampled_backups=total, backup_action_histogram=self.backup_counts.tolist(),
                    mean_backup_actions=float(self.backup_counts@np.arange(1, len(self.backup_counts)+1)/total)
                    if total else 0., policy_cut_backups=self.policy_cut_count,
                    policy_cut_fraction=self.policy_cut_count/total if total else 0.)
