"""Score-only actor/critic learning in the learned Defense world model.

Gaussian RSSM from the local preflight, REINFORCE behavior learning inspired
by DreamerV2 (arxiv:2010.02193). This is an adaptation, not a reproduction.
The emulator is never queried during imagination; no recorded action is an
imitation target. Model errors can invalidate predicted returns.
"""

from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .defense_world_model import WorldModel
from .recurrent_policy import RecurrentPolicy


ALGORITHM = 'gaussian-rssm-imagined-reinforce'
ARCHITECTURE = 'screen-rssm-imagined-actor-v1'


def head(outputs):
    return nn.Sequential(nn.Linear(160, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                         nn.Linear(256, outputs))


def lambda_returns(rewards, discounts, values, trace=.95):
    """r[t] is earned arriving at state[t+1]; values includes the final state."""
    following = values[:, -1]
    returns = []
    for t in reversed(range(rewards.shape[1])):
        following = rewards[:, t]+discounts[:, t]*((1-trace)*values[:, t+1]+trace*following)
        returns.append(following)
    return mx.stack(returns[::-1], axis=1)


def survival_weights(discounts):
    return mx.cumprod(mx.concatenate([mx.ones_like(discounts[:, :1]), discounts[:, :-1]], axis=1), axis=1)


class WorldActor(nn.Module):
    """Self-contained acting bundle; decoder/reward heads retained for provenance."""
    def __init__(self):
        super().__init__()
        self.world = WorldModel()
        self.actor = head(20)

    def infer(self, observations, memory):
        # The memory contains only prior neural state and our own previous action.
        h, z = memory[:, :128], memory[:, 128:160]
        previous = (memory[:, 160]-1).astype(mx.int32)
        state = (h, z, mx.zeros_like(z), mx.ones_like(z))
        updated, _ = self.world.step(state, previous, mx.random.key(0),
            embedding=self.world.encode(observations[:, -1]), sample=False)
        features = self.world.features(updated)
        return self.actor(features), mx.concatenate([features, mx.zeros_like(memory[:, :1])], axis=1)


class WorldPolicy(RecurrentPolicy):
    def __init__(self, infer, seed=0, temperature=1.):
        super().__init__(infer, 161, seed, temperature)

    def choose(self, obs, hidden, uniforms):
        actions, updated = super().choose(obs, hidden, uniforms)
        updated[:, 160] = actions+1
        return actions, updated


def acting_policy(model, temperature=1.):
    predict = mx.compile(model.infer, inputs=model.state)
    def infer(obs, memory):
        logits, updated = predict(mx.array(obs), mx.array(memory))
        return np.array(logits), np.array(updated)
    return WorldPolicy(infer, temperature=temperature)


class ImaginationLearner:
    def __init__(self, world=None, seed=0, horizon=15, gamma=.999, trace=.95, entropy=.001):
        if (isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 64
                or not np.isfinite([gamma, trace, entropy]).all()
                or not 0 < gamma <= 1 or not 0 <= trace <= 1 or not 0 <= entropy <= 1):
            raise ValueError('invalid imagination configuration')
        mx.random.seed(seed)
        self.model = WorldActor()
        if world is not None:
            self.model.world = world
        self.critic, self.target = head(1), head(1)
        self.target.update(tree_map(lambda x: mx.array(x), self.critic.parameters()))
        self.actor_optimizer = optim.Adam(learning_rate=4e-5, eps=1e-5)
        self.value_optimizer = optim.Adam(learning_rate=1e-4, eps=1e-5)
        self.actor_optimizer.init(self.model.actor.trainable_parameters())
        self.value_optimizer.init(self.critic.trainable_parameters())
        self.random = {'key': mx.random.key(seed+305)}
        self.horizon, self.gamma, self.trace, self.entropy = horizon, gamma, trace, entropy
        self.updates = 0
        self.rebind()

    def rebind(self):
        self.state = [self.model.state, self.critic.state, self.target.state,
                      self.actor_optimizer.state, self.value_optimizer.state, self.random]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def imagine(self, start, key):
        world = self.model.world
        state = tuple(mx.stop_gradient(s) for s in start)
        features, actions, rewards, discounts = [world.features(state)], [], [], []
        keys = mx.random.split(key, self.horizon*2)
        for t in range(self.horizon):
            logits = self.model.actor(features[-1])
            action = mx.stop_gradient(mx.random.categorical(logits, key=keys[2*t]))
            state, _ = world.step(state, action, keys[2*t+1], sample=True)
            feature = world.features(state)
            actions.append(action); features.append(feature)
            rewards.append(world.reward(feature)[:, 0])
            discounts.append(self.gamma*mx.sigmoid(world.continue_logit(feature)[:, 0]))
        # REINFORCE: gradients do not backpropagate through sampled trajectories.
        return tuple(mx.stop_gradient(mx.stack(values, axis=1)) for values in
                     (features, actions, rewards, discounts))

    def _actor_loss(self, actor, features, actions, advantages, weights):
        logits = actor(mx.stop_gradient(features))
        log_probs = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
        entropy = -mx.sum(mx.exp(log_probs)*log_probs, axis=-1)
        loss = -mx.mean(mx.stop_gradient(weights)*(selected*mx.stop_gradient(advantages)+self.entropy*entropy))
        return loss, mx.mean(entropy)

    @staticmethod
    def _value_loss(critic, features, returns, weights):
        values = critic(mx.stop_gradient(features))[..., 0]
        return .5*mx.mean(mx.stop_gradient(weights)*(values-mx.stop_gradient(returns))**2)

    def _update(self, start, initial_alive):
        key, following = mx.random.split(self.random['key'])
        features, actions, rewards, discounts = self.imagine(start, key)
        values = self.target(features)[..., 0]
        returns = mx.stop_gradient(lambda_returns(rewards, discounts, values, self.trace))
        weights = mx.stop_gradient(initial_alive[:, None]*survival_weights(discounts))
        current = features[:, :-1]
        advantages = mx.stop_gradient(returns-self.critic(current)[..., 0])
        (actor_loss, entropy), actor_grad = nn.value_and_grad(self.model.actor, self._actor_loss)(
            self.model.actor, current, actions, advantages, weights)
        value_loss, value_grad = nn.value_and_grad(self.critic, self._value_loss)(
            self.critic, current, returns, weights)
        actor_grad, actor_norm = optim.clip_grad_norm(actor_grad, max_norm=100.)
        value_grad, value_norm = optim.clip_grad_norm(value_grad, max_norm=100.)
        self.actor_optimizer.update(self.model.actor, actor_grad)
        self.value_optimizer.update(self.critic, value_grad)
        self.random['key'] = following
        return mx.stack([actor_loss, value_loss, entropy, mx.mean(returns), mx.mean(rewards),
                         mx.mean(discounts), actor_norm, value_norm])

    def train(self, start, alive):
        result = self.update(tuple(mx.array(s) for s in start), mx.array(alive))
        mx.eval(result, self.state)
        values = np.array(result)
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite imagination update')
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.update(tree_map(lambda x: mx.array(x), self.critic.parameters()))
            mx.eval(self.target.state)
        names = ('actor_loss', 'value_loss', 'entropy', 'imagined_return', 'imagined_reward',
                 'imagined_discount', 'actor_gradient_norm', 'value_gradient_norm')
        return dict(zip(names, map(float, values), strict=True))

    def starts(self, frames, actions, continuation, burn=8):
        # Deterministic posterior states match real acting; rollout latents sample.
        states, _ = self.model.world.observe(mx.array(frames), mx.array(actions),
                                            mx.random.key(0), sample=False)
        start = tuple(mx.stop_gradient(s[:, burn:-1].reshape(-1, s.shape[-1])) for s in states)
        # A state immediately after a visible boundary is not an alive start.
        alive = mx.array(continuation[:, burn-1:-1].reshape(-1))
        mx.eval(start, alive)
        return start, alive

    def save(self, path, config, sampling_rng):
        from .defense_learning import sha256, write_json
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        self.model.save_weights(str(path/'model.safetensors'))
        self.critic.save_weights(str(path/'critic.safetensors'))
        self.target.save_weights(str(path/'critic-target.safetensors'))
        for name, state in (('actor-optimizer.npz', self.actor_optimizer.state),
                            ('value-optimizer.npz', self.value_optimizer.state)):
            mx.savez(str(path/name), **dict(tree_flatten(state)))
        mx.savez(str(path/'random.npz'), **self.random)
        write_json(path/'state.json', dict(config=config, steps=self.updates, actor_updates=self.updates,
            step_unit='imagined actor/critic updates, not emulator actions',
            sampling_rng=sampling_rng.bit_generator.state,
            behavior=dict(horizon=self.horizon, gamma=self.gamma, trace=self.trace, entropy=self.entropy),
            hashes={p.name: sha256(p) for p in path.iterdir() if p.is_file()}))

    def restore(self, path, sampling_rng):
        from .defense_learning import sha256
        path = Path(path); saved = json.loads((path/'state.json').read_text())
        if saved['behavior'] != dict(horizon=self.horizon, gamma=self.gamma, trace=self.trace, entropy=self.entropy):
            raise ValueError('imagination settings differ')
        names = ('model.safetensors', 'critic.safetensors', 'critic-target.safetensors',
                 'actor-optimizer.npz', 'value-optimizer.npz', 'random.npz')
        for name in names:
            if sha256(path/name) != saved['hashes'][name]:
                raise ValueError('imagination checkpoint checksum mismatch')
        for model, name in ((self.model, names[0]), (self.critic, names[1]), (self.target, names[2])):
            model.load_weights(str(path/name))
        self.actor_optimizer.state = tree_unflatten(list(mx.load(str(path/names[3])).items()))
        self.value_optimizer.state = tree_unflatten(list(mx.load(str(path/names[4])).items()))
        self.random = dict(mx.load(str(path/names[5])))
        self.updates = saved['actor_updates']
        sampling_rng.bit_generator.state = saved['sampling_rng']
        self.rebind()
        return saved['config']
