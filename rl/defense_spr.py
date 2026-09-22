"""Optional SPR-inspired auxiliary loss on the learner's own visible paths.

Schwarzer et al., arXiv:2007.05929. This scalar-DQN adaptation uses the existing
32-channel encoder, location-wise LayerNorm, and mean valid-step cosine loss.
No prediction model, dropout or future observation is used to select actions.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .model import Learner, QNetwork


AUX_FILES = ('spr-aux.safetensors', 'spr-ema.safetensors',
             'spr-optimizer.npz', 'spr-rng.npz')


def normalize_spatial(x):
    minimum = mx.min(x, axis=(1, 2, 3), keepdims=True)
    maximum = mx.max(x, axis=(1, 2, 3), keepdims=True)
    return (x-minimum)/mx.maximum(maximum-minimum, 1e-5)


def spatial_features(network, screen, key, dropout):
    # Same fixed visible encoding and learned convolutions as QNetwork.
    x = network._table[screen.astype(mx.int32)]
    x = x.transpose(0, 2, 4, 3, 5, 1, 6).reshape(-1, 48, 128, 8)
    keys = mx.random.split(key, len(network.conv))
    for layer, subkey in zip(network.conv, keys):
        x = nn.relu(layer(x))
        if dropout:
            x = x*mx.random.bernoulli(1-dropout, x.shape, key=subkey)/(1-dropout)
    return normalize_spatial(x)


def cosine_distance(predicted, target):
    target = mx.stop_gradient(target)
    def unit(x):
        return x*mx.rsqrt(mx.maximum(mx.sum(x*x, axis=-1, keepdims=True), 1e-8))
    return 1-mx.clip(mx.sum(unit(predicted)*unit(target), axis=-1), -1, 1)


class SprAuxiliary(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.conv1 = nn.Conv2d(32+action_count, 32, 3, padding=1)
        self.norm = nn.LayerNorm(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.predictor = nn.Linear(256, 256)
        self._action_count = action_count

    def transition(self, latent, actions):
        one_hot = (actions[:, None] == mx.arange(self._action_count)[None]).astype(latent.dtype)
        tiled = mx.broadcast_to(one_hot[:, None, None, :],
                                (*latent.shape[:3], self._action_count))
        x = nn.relu(self.norm(self.conv1(mx.concatenate([latent, tiled], axis=-1))))
        return normalize_spatial(nn.relu(self.conv2(x)))


class SprTrainingModel(nn.Module):
    def __init__(self, q, auxiliary):
        super().__init__()
        self.q, self.aux = q, auxiliary


class SprLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, horizon=5,
                 weight=.1, ema=.99, dropout=.5):
        if (not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 5
                or not np.isfinite([weight, ema, dropout]).all()
                or not 0 < weight <= 10 or not 0 <= ema < 1 or not 0 <= dropout < 1):
            raise ValueError('invalid SPR horizon, weight, EMA or dropout')
        super().__init__(learning_rate, seed, action_count)
        self.weight, self.ema_rate, self.dropout = weight, ema, dropout
        self.horizon = horizon
        self.aux = SprAuxiliary(action_count)
        self.ema = QNetwork(action_count)
        self.aux_optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.aux_optimizer.init(self.aux.trainable_parameters())
        self.aux_rng = {'key': mx.random.key(seed+4089)}
        self.training_model = SprTrainingModel(self.online, self.aux)
        self.local_updates = self.predictions = 0
        self.last_loss = 0.
        self.restore_auxiliary(None)

    def rebind(self):
        self.state = [self.online.state, self.target.state, self.optimizer.state,
                      self.aux.state, self.ema.state, self.aux_optimizer.state, self.aux_rng]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1),
                                  inputs=self.online.state)

    def restore_auxiliary(self, directory, saved=None):
        if directory is None:
            # Conversion happens AFTER the own scalar online weights are loaded.
            self.ema.update(tree_map(lambda x: mx.array(x), self.online.parameters()))
        else:
            from .defense_learning import sha256
            directory = Path(directory)
            hashes = (saved or {}).get('spr_auxiliary_hashes', {})
            if set(hashes) != set(AUX_FILES):
                raise ValueError('SPR resume requires complete auxiliary provenance')
            for name in AUX_FILES:
                if not (directory/name).is_file() or sha256(directory/name) != hashes[name]:
                    raise ValueError('SPR auxiliary checkpoint missing or checksum mismatch: '+name)
            self.aux.load_weights(str(directory/AUX_FILES[0]))
            self.ema.load_weights(str(directory/AUX_FILES[1]))
            self.aux_optimizer.state = tree_unflatten(list(mx.load(str(directory/AUX_FILES[2])).items()))
            self.aux_optimizer.learning_rate = self.optimizer.learning_rate
            random_state = mx.load(str(directory/AUX_FILES[3]))
            if (set(random_state) != {'key'} or random_state['key'].shape != (2,)
                    or random_state['key'].dtype != mx.uint32):
                raise ValueError('invalid SPR random key')
            self.aux_rng.update(random_state)
        self.rebind()

    def save_auxiliary(self, directory):
        from .defense_learning import sha256
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in zip(AUX_FILES[:2], (self.aux, self.ema)):
            temp = directory/name.replace('.safetensors', '.tmp.safetensors')
            model.save_weights(str(temp)); temp.replace(directory/name)
        for name, state in ((AUX_FILES[2], dict(tree_flatten(self.aux_optimizer.state))),
                            (AUX_FILES[3], self.aux_rng)):
            temp = directory/name.replace('.npz', '.tmp.npz')
            mx.savez(str(temp), **state); temp.replace(directory/name)
        return {name: sha256(directory/name) for name in AUX_FILES}

    def auxiliary_loss(self, model, obs, actions, following, key):
        batch, horizon = actions.shape
        root_key, target_key = mx.random.split(key)
        latent = spatial_features(model.q, obs, root_key, self.dropout)
        future = spatial_features(self.ema, following.reshape(batch*horizon, *obs.shape[1:]),
                                  target_key, self.dropout)
        targets = mx.stop_gradient(nn.relu(self.ema.hidden(future.reshape(batch*horizon, -1))))
        targets = targets.reshape(batch, horizon, -1)
        losses = []
        for i in range(horizon):
            latent = model.aux.transition(latent, mx.maximum(actions[:, i], 0))
            projected = nn.relu(model.q.hidden(latent.reshape(batch, -1)))
            losses.append(cosine_distance(model.aux.predictor(projected), targets[:, i]))
        valid = actions >= 0
        per_path = mx.sum(mx.stack(losses, axis=1)*valid, axis=1)/mx.maximum(valid.sum(axis=1), 1)
        return per_path, valid.sum()

    def _joint_loss(self, model, obs, actions, returns, following, discounts, weights,
                    path_actions, path_following, key):
        td_loss, (errors, q) = Learner._loss(self, model.q, obs, actions, returns,
                                           following, discounts, weights)
        auxiliary, count = self.auxiliary_loss(model, obs, path_actions, path_following, key)
        spr_loss = mx.mean(weights*auxiliary)
        return td_loss+self.weight*spr_loss, (errors, q, spr_loss, count)

    def _update(self, *batch):
        key, following_key = mx.random.split(self.aux_rng['key'])
        (loss, (errors, q, spr_loss, count)), grads = nn.value_and_grad(
            self.training_model, self._joint_loss)(self.training_model, *batch, key)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.)
        self.optimizer.update(self.online, grads['q'])
        self.aux_optimizer.update(self.aux, grads['aux'])
        self.ema.update(tree_map(lambda target, online: self.ema_rate*target+(1-self.ema_rate)*online,
                                 self.ema.parameters(), self.online.parameters()))
        self.aux_rng['key'] = following_key
        return loss, errors, q, norm, spr_loss, count

    def train(self, batch):
        result = self.update(*(mx.array(x) for x in batch))
        mx.eval(result, self.state)
        self.local_updates += 1
        self.predictions += int(result[5].item())
        self.last_loss = float(result[4].item())
        return float(result[0].item()), np.array(result[1]), float(result[2].item())

    def spr_stats(self):
        return dict(local_updates=self.local_updates, valid_future_predictions=self.predictions,
                    last_weighted_cosine_loss=self.last_loss, weight=self.weight,
                    ema=self.ema_rate, dropout=self.dropout)
