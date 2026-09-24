"""Residual screen-history GRU and sequence PPO; imported only by GPU parent."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import QNetwork
from .ppo import PPO, _load_backend
from .recurrent_policy import OwnActionRecurrentPolicy, RecurrentPolicy


class ResidualRecurrentNetwork(nn.Module):
    def __init__(self, action_count=20, hidden_size=128, memory_scale=1.,
                 own_action_input=False):
        super().__init__()
        if (isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1
                or not np.isfinite(memory_scale) or not 0 <= memory_scale <= 1
                or not isinstance(own_action_input, bool)):
            raise ValueError("invalid recurrent network configuration")
        self.base = QNetwork(action_count=action_count)
        self.memory = nn.GRU(256+int(own_action_input)*(action_count+1), hidden_size)
        self.memory_actor = nn.Linear(hidden_size, action_count)
        self.memory_value = nn.Linear(hidden_size, 1)
        self.memory_actor.weight = mx.zeros_like(self.memory_actor.weight)
        self.memory_actor.bias = mx.zeros_like(self.memory_actor.bias)
        self.memory_value.weight = mx.zeros_like(self.memory_value.weight)
        self.memory_value.bias = mx.zeros_like(self.memory_value.bias)
        self._hidden_size, self._memory_scale = hidden_size, memory_scale
        self._own_action_input, self._action_count = own_action_input, action_count
        if own_action_input:
            self._action_table = mx.eye(action_count+1, dtype=mx.float32)

    def memory_input(self, features, previous_actions=None):
        if not self._own_action_input:
            if previous_actions is not None:
                raise ValueError("screen-only recurrent network rejects own actions")
            return features
        if (previous_actions is None
                or previous_actions.shape != features.shape[:-1]):
            raise ValueError("own-action input must align with screen features")
        return mx.concatenate((features,
                               self._action_table[previous_actions.astype(mx.int32)]), axis=-1)

    def heads(self, features, hidden):
        logits = self.base.advantage(features) + self._memory_scale * self.memory_actor(hidden)
        values = (self.base.value(features) + self._memory_scale * self.memory_value(hidden))[..., 0]
        return logits, values

    def step(self, obs, hidden, previous_actions=None):
        features = self.base.features(obs)
        memory_input = self.memory_input(features, previous_actions)
        hidden = self.memory(memory_input[:, None, :], hidden)[:, 0, :]
        logits, values = self.heads(features, hidden)
        return logits, values, hidden

    def sequence(self, obs, hidden, starts, previous_actions=None):
        batch, length = obs.shape[:2]
        features = self.base.features(obs.reshape(-1, *obs.shape[2:])).reshape(batch, length, 256)
        memory_input = self.memory_input(features, previous_actions)
        states = []
        for t in range(length):
            # Only actual reset boundaries, not life-loss labels, clear memory.
            hidden = hidden * (1 - starts[:, t, None].astype(hidden.dtype))
            hidden = self.memory(memory_input[:, t:t+1, :], hidden)[:, 0, :]
            states.append(hidden)
        all_hidden = mx.stack(states, axis=1)
        logits, values = self.heads(features, all_hidden)
        return logits, values, hidden


class RecurrentPPO(PPO):
    def __init__(self, seed=41, learning_rate=2.5e-4, entropy=.002, action_count=20,
                 value_coefficient=.5, hidden_size=128, memory_scale=1., freeze_base=False,
                 own_action_input=False):
        if (isinstance(value_coefficient, (bool, np.bool_))
                or not np.isfinite(value_coefficient) or value_coefficient < 0):
            raise ValueError("value coefficient must be finite and nonnegative")
        if not isinstance(freeze_base, bool):
            raise ValueError("freeze_base must be boolean")
        _load_backend()  # inherited save() uses the ordinary PPO backend globals.
        mx.random.seed(seed)
        self.model = ResidualRecurrentNetwork(action_count, hidden_size, memory_scale,
                                              own_action_input=own_action_input)
        if freeze_base:
            # Freeze before creating Adam: no base gradients or optimizer slots.
            # parameters()/save_weights() still preserve the complete base.
            self.model.base.freeze()
        self.optimizer = optim.Adam(learning_rate, eps=1e-5)
        self.optimizer.init(self.model.trainable_parameters())
        self.entropy, self.value_coefficient = entropy, value_coefficient
        self.hidden_size, self.memory = hidden_size, None
        self.own_action_input, self.action_count = own_action_input, action_count
        self.previous_actions = None
        self.compile()

    def compile(self):
        self.state = [self.model.state, self.optimizer.state]
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(self.model.step, inputs=self.model.state)
        mx.eval(self.state)

    def reset_memory(self, count):
        self.memory = np.zeros((count, self.hidden_size), np.float32)
        if self.own_action_input:
            self.previous_actions = np.full(count, self.action_count, np.int32)

    def reset_done(self, mask):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (len(self.memory),):
            raise ValueError("one reset flag per recurrent stream required")
        self.memory[mask] = 0
        if self.own_action_input:
            self.previous_actions[mask] = self.action_count

    def act(self, obs, rng):
        if self.memory is None or len(self.memory) != len(obs):
            raise ValueError("reset_memory before acting")
        if self.own_action_input:
            logits, values, hidden = self.predict(
                mx.array(obs), mx.array(self.memory), mx.array(self.previous_actions))
        else:
            logits, values, hidden = self.predict(mx.array(obs), mx.array(self.memory))
        logits, values, self.memory = np.array(logits), np.array(values), np.array(hidden)
        logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        actions = (rng.random(len(obs))[:, None] > np.cumsum(np.exp(logp), axis=1)).sum(axis=1).clip(0, logits.shape[1]-1)
        if self.own_action_input:
            self.previous_actions = actions.astype(np.int32).copy()
        return actions.astype(np.int32), logp[np.arange(len(obs)), actions], values

    def bootstrap_value(self, obs, indices=None):
        memory = self.memory if indices is None else self.memory[indices]
        # Prediction must not consume a screen or advance carried neural memory.
        if self.own_action_input:
            previous = self.previous_actions if indices is None else self.previous_actions[indices]
            return self.predict(mx.array(obs), mx.array(memory), mx.array(previous))[1]
        return self.predict(mx.array(obs), mx.array(memory))[1]

    def _loss(self, model, obs, actions, old_logp, advantages, returns, initial_hidden,
              starts, previous_actions=None):
        logits, values, _ = model.sequence(obs, mx.stop_gradient(initial_hidden), starts,
                                           previous_actions)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logp = mx.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
        logratio = logp - old_logp
        ratio = mx.exp(logratio)
        actor = -mx.mean(mx.minimum(ratio * advantages, mx.clip(ratio, .8, 1.2) * advantages))
        critic = .5 * mx.mean(mx.square(values - returns))
        entropy = -mx.mean(mx.sum(mx.exp(log_probs) * log_probs, axis=-1))
        kl = mx.mean((ratio - 1) - logratio)
        return actor + self.value_coefficient * critic - self.entropy * entropy, (actor, critic, entropy, kl)

    def _update(self, *args):
        (loss, metrics), grads = nn.value_and_grad(self.model, self._loss)(self.model, *args)
        grads, _ = optim.clip_grad_norm(grads, .5)
        self.optimizer.update(self.model, grads)
        return loss, metrics

    def policy(self, seed=0):
        if self.own_action_input:
            def infer(obs, hidden, previous):
                logits, _, updated = self.predict(
                    mx.array(obs), mx.array(hidden), mx.array(previous))
                return np.array(logits), np.array(updated)
            return OwnActionRecurrentPolicy(infer, self.hidden_size, self.action_count,
                                            seed=seed)
        def infer(obs, hidden):
            logits, _, updated = self.predict(mx.array(obs), mx.array(hidden))
            return np.array(logits), np.array(updated)
        return RecurrentPolicy(infer, self.hidden_size, seed=seed)
