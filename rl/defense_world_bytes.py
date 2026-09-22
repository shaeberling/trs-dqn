"""Optional categorical reconstruction of all visible screen bytes.

Auxiliary representation loss only. Acting uses the unchanged core RSSM;
there are no hidden-memory targets, object labels, demonstrations or rewards.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .defense_world_model import WorldLearner


BYTE_ARCHITECTURE = 'visible-cell-deconvolution-256-v1'


class ByteDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(160, 4*16*32)
        self.expand = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.output = nn.ConvTranspose2d(32, 256, 4, 2, 1)

    def __call__(self, features):
        x = nn.elu(self.input(features)).reshape(-1, 4, 16, 32)
        x = self.output(nn.elu(self.expand(x)))
        return x.reshape(*features.shape[:-1], 16, 64, 256)


class ByteWorld(nn.Module):
    def __init__(self, world, decoder):
        super().__init__()
        self.world, self.decoder = world, decoder

    def loss(self, frames, actions, rewards, continuation, key, burn, weight):
        filtered = self.world.observe(frames, actions, key)
        core, terms = self.world.loss(frames, actions, rewards, continuation, key, burn,
                                      filtered=filtered)
        features = self.world.features(filtered[0])[:, burn+1:]
        logits = self.decoder(features)
        byte_loss = nn.losses.cross_entropy(logits, frames[:, burn+1:].astype(mx.int32), reduction='mean')
        return core+weight*byte_loss, mx.concatenate([terms, byte_loss[None]])


class ByteWorldLearner:
    def __init__(self, seed=0, learning_rate=6e-4, burn=8, byte_weight=1.):
        if not np.isfinite(byte_weight) or byte_weight <= 0:
            raise ValueError('positive finite byte reconstruction weight required')
        self.base = WorldLearner(seed=seed, learning_rate=learning_rate, burn=burn)
        self.model, self.optimizer, self.random = self.base.model, self.base.optimizer, self.base.random
        self.decoder = ByteDecoder()
        self.system = ByteWorld(self.model, self.decoder)
        self.byte_optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.byte_optimizer.init(self.decoder.trainable_parameters())
        self.byte_reconstruction = dict(weight=float(byte_weight), architecture=BYTE_ARCHITECTURE)
        self.overshooting = self.base.overshooting
        self.burn, self.updates, self.byte_updates = burn, 0, 0
        self.rebind()

    def rebind(self):
        self.state = [self.model.state, self.decoder.state, self.optimizer.state,
                      self.byte_optimizer.state, self.random]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def _update(self, frames, actions, rewards, continuation):
        key, following = mx.random.split(self.random['key'])
        def objective(system):
            return system.loss(frames, actions, rewards, continuation, key,
                               self.burn, self.byte_reconstruction['weight'])
        (loss, terms), grads = nn.value_and_grad(self.system, objective)(self.system)
        grads, norm = optim.clip_grad_norm(grads, max_norm=100.)
        self.optimizer.update(self.model, grads['world'])
        self.byte_optimizer.update(self.decoder, grads['decoder'])
        self.random['key'] = following
        return loss, terms, norm

    def train(self, batch):
        result = self.update(*(mx.array(v) for v in batch))
        mx.eval(result, self.state)
        if not all(np.isfinite(np.array(v)).all() for v in result):
            raise FloatingPointError('nonfinite byte-world update')
        self.updates += 1; self.byte_updates += 1
        return dict(loss=float(result[0]), reconstruction=float(result[1][0]),
            reward=float(result[1][1]), continuation=float(result[1][2]), kl=float(result[1][3]),
            byte_cross_entropy=float(result[1][4]), gradient_norm=float(result[2]))

    def save(self, directory, sampling_rng, metadata):
        from .defense_learning import sha256, write_json
        directory = Path(directory)
        self.base.updates, self.base.random = self.updates, self.random
        self.base.save(directory, sampling_rng, metadata)
        self.decoder.save_weights(str(directory/'byte-decoder.safetensors'))
        mx.savez(str(directory/'byte-optimizer.npz'), **dict(tree_flatten(self.byte_optimizer.state)))
        saved = json.loads((directory/'state.json').read_text())
        saved.update(byte_reconstruction=self.byte_reconstruction, byte_updates=self.byte_updates)
        saved['hashes'].update({n: sha256(directory/n) for n in ('byte-decoder.safetensors', 'byte-optimizer.npz')})
        write_json(directory/'state.json', saved)

    def restore(self, directory, sampling_rng, allow_objective_change=False):
        from .defense_learning import sha256
        directory = Path(directory); saved = json.loads((directory/'state.json').read_text())
        previous = saved.get('byte_reconstruction')
        if previous != self.byte_reconstruction and not allow_objective_change:
            raise ValueError('byte objective differs; explicit change required')
        if saved.get('overshooting', {}).get('weight', 0.):
            raise ValueError('combined byte/overshoot continuation is not supported')
        if saved.get('prior_outputs', {}).get('weight', 0.):
            raise ValueError('combined byte/prior-output continuation is not supported')
        if previous:
            if previous['architecture'] != BYTE_ARCHITECTURE:
                raise ValueError('incompatible byte decoder')
            for name in ('byte-decoder.safetensors', 'byte-optimizer.npz'):
                if sha256(directory/name) != saved['hashes'][name]:
                    raise ValueError('byte checkpoint checksum mismatch')
        metadata = self.base.restore(directory, sampling_rng, allow_objective_change=True)
        self.optimizer, self.random, self.updates = self.base.optimizer, self.base.random, self.base.updates
        if previous:
            self.decoder.load_weights(str(directory/'byte-decoder.safetensors'))
            self.byte_optimizer.state = tree_unflatten(list(mx.load(str(directory/'byte-optimizer.npz')).items()))
            self.byte_updates = saved['byte_updates']
        self.rebind()
        return metadata
