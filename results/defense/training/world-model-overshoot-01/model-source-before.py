"""Small Dreamer-inspired RSSM for own screen/action trajectories.

Dynamics preflight only: no planner, acting policy, emulator query, or reward
bonus. Hafner et al., https://arxiv.org/abs/1912.01603. Architecture and image
encoding are adapted to this repository, not a paper reproduction.
"""

from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .model import render_table


def normal_kl(mean, std, prior_mean, prior_std):
    return mx.sum(mx.log(prior_std/std) +
                  (std**2+(mean-prior_mean)**2)/(2*prior_std**2)-.5, axis=-1)


class WorldModel(nn.Module):
    def __init__(self, hidden=128, stochastic=32, actions=20):
        super().__init__()
        self._hidden, self._stochastic, self._actions = hidden, stochastic, actions
        self._table = mx.array(render_table())
        self.encoder = [nn.Conv2d(2, 16, 4, 2, 1), nn.Conv2d(16, 32, 4, 2, 1),
                        nn.Conv2d(32, 32, 4, 2, 1)]
        self.embedding = nn.Linear(6*16*32, 256)
        self.action_input = nn.Linear(stochastic+actions, hidden)
        self.memory = nn.GRU(hidden, hidden)
        self.prior = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                   nn.Linear(hidden, stochastic*2))
        self.posterior = nn.Sequential(nn.Linear(hidden+256, hidden), nn.ELU(),
                                       nn.Linear(hidden, stochastic*2))
        self.decode_input = nn.Linear(hidden+stochastic, 6*16*32)
        self.decoder = [nn.ConvTranspose2d(32, 32, 4, 2, 1),
                        nn.ConvTranspose2d(32, 16, 4, 2, 1),
                        nn.ConvTranspose2d(16, 2, 4, 2, 1)]
        self.reward = nn.Sequential(nn.Linear(hidden+stochastic, hidden), nn.ELU(),
                                    nn.Linear(hidden, 1))
        self.continue_logit = nn.Sequential(nn.Linear(hidden+stochastic, hidden), nn.ELU(),
                                           nn.Linear(hidden, 1))

    def pixels(self, frames):
        shape = frames.shape[:-2]
        cells = self._table[frames.reshape(-1, 16, 64).astype(mx.int32)]
        return cells.transpose(0, 1, 3, 2, 4, 5).reshape(*shape, 48, 128, 2)-.5

    def encode(self, frames):
        x = self.pixels(frames).reshape(-1, 48, 128, 2)
        for layer in self.encoder:
            x = nn.elu(layer(x))
        return nn.elu(self.embedding(x.reshape(len(x), -1))).reshape(*frames.shape[:-2], 256)

    def initial(self, batch):
        return (mx.zeros((batch, self._hidden)), mx.zeros((batch, self._stochastic)),
                mx.zeros((batch, self._stochastic)), mx.ones((batch, self._stochastic)))

    def step(self, state, action, key, embedding=None, sample=True):
        # action is the PREVIOUS command; no current/future screen in the prior.
        h, z, _, _ = state
        one_hot = (action[:, None] == mx.arange(self._actions)[None]).astype(mx.float32)
        x = nn.elu(self.action_input(mx.concatenate([z, one_hot], axis=-1)))
        h = self.memory(x[:, None], h)[:, 0]
        pm, ps = mx.split(self.prior(h), 2, axis=-1)
        ps = nn.softplus(ps)+.1
        if embedding is None:
            mean, std = pm, ps
        else:
            mean, std = mx.split(self.posterior(mx.concatenate([h, embedding], axis=-1)), 2, axis=-1)
            std = nn.softplus(std)+.1
        z = mean+std*mx.random.normal(mean.shape, key=key) if sample else mean
        return (h, z, mean, std), (pm, ps)

    @staticmethod
    def features(state):
        return mx.concatenate(state[:2], axis=-1)

    def observe(self, frames, actions, key, sample=True):
        embedded = self.encode(frames)
        state = self.initial(len(frames))
        keys = mx.random.split(key, frames.shape[1])
        states, divergences = [], []
        for t in range(frames.shape[1]):
            # A zero vector, not NOOP one-hot, at the beginning of a chunk.
            previous = actions[:, t-1] if t else mx.full((len(frames),), -1, mx.int32)
            state, (pm, ps) = self.step(state, previous, keys[t], embedded[:, t], sample)
            states.append(state)
            divergences.append(normal_kl(state[2], state[3], pm, ps))
        return tuple(mx.stack([s[k] for s in states], axis=1) for k in range(4)), mx.stack(divergences, axis=1)

    def decode(self, features):
        x = nn.elu(self.decode_input(features)).reshape(-1, 6, 16, 32)
        for layer in self.decoder[:-1]:
            x = nn.elu(layer(x))
        return self.decoder[-1](x).reshape(*features.shape[:-1], 48, 128, 2)

    def loss(self, frames, actions, rewards, continuation, key, burn=8):
        states, kl = self.observe(frames, actions, key)
        features = self.features(states)[:, burn+1:]
        # Reward[i] belongs to action[i] and the resulting frame[i+1].
        target = self.pixels(frames[:, burn+1:])
        reconstruction = .5*mx.mean(mx.sum((self.decode(features)-target)**2, axis=(-3, -2, -1)))
        reward_loss = .5*mx.mean((self.reward(features)[..., 0]-rewards[:, burn:])**2)
        boundary_loss = mx.mean(nn.losses.binary_cross_entropy(
            self.continue_logit(features)[..., 0], continuation[:, burn:], with_logits=True))
        divergence = mx.mean(mx.maximum(kl[:, burn+1:], 3.))
        return reconstruction+reward_loss+boundary_loss+divergence, mx.stack(
            [reconstruction, reward_loss, boundary_loss, divergence])


class WorldLearner:
    def __init__(self, seed=0, learning_rate=6e-4, burn=8):
        mx.random.seed(seed)
        self.model = WorldModel()
        self.optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.optimizer.init(self.model.trainable_parameters())
        self.random = {'key': mx.random.key(seed+193)}
        self.burn, self.updates = burn, 0
        self.rebind()

    def rebind(self):
        self.state = [self.model.state, self.optimizer.state, self.random]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def _update(self, frames, actions, rewards, continuation):
        key, next_key = mx.random.split(self.random['key'])
        def objective(model):
            return model.loss(frames, actions, rewards, continuation, key, self.burn)
        (loss, terms), grads = nn.value_and_grad(self.model, objective)(self.model)
        grads, norm = optim.clip_grad_norm(grads, max_norm=100.)
        self.optimizer.update(self.model, grads)
        self.random['key'] = next_key
        return loss, terms, norm

    def train(self, batch):
        result = self.update(*(mx.array(v) for v in batch))
        mx.eval(result, self.state)
        if not all(np.isfinite(np.array(v)).all() for v in result):
            raise FloatingPointError('nonfinite world-model update')
        self.updates += 1
        return dict(loss=float(result[0]), reconstruction=float(result[1][0]),
                    reward=float(result[1][1]), continuation=float(result[1][2]),
                    kl=float(result[1][3]), gradient_norm=float(result[2]))

    def save(self, directory, sampling_rng, metadata):
        from .defense_learning import sha256, write_json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_weights(str(directory/'world.safetensors'))
        mx.savez(str(directory/'optimizer.npz'), **dict(tree_flatten(self.optimizer.state)))
        mx.savez(str(directory/'random.npz'), **self.random)
        write_json(directory/'state.json', dict(updates=self.updates, burn=self.burn,
            sampling_rng=sampling_rng.bit_generator.state, metadata=metadata,
            hashes={n: sha256(directory/n) for n in ('world.safetensors', 'optimizer.npz', 'random.npz')}))

    def restore(self, directory, sampling_rng):
        from .defense_learning import sha256
        directory = Path(directory)
        saved = json.loads((directory/'state.json').read_text())
        if saved['burn'] != self.burn:
            raise ValueError('incompatible burn-in')
        for name in ('world.safetensors', 'optimizer.npz', 'random.npz'):
            if sha256(directory/name) != saved['hashes'][name]:
                raise ValueError('world checkpoint checksum mismatch')
        self.model.load_weights(str(directory/'world.safetensors'))
        self.optimizer.state = tree_unflatten(list(mx.load(str(directory/'optimizer.npz')).items()))
        self.random = dict(mx.load(str(directory/'random.npz')))
        sampling_rng.bit_generator.state = saved['sampling_rng']
        self.updates = saved['updates']
        self.rebind()
        return saved['metadata']
