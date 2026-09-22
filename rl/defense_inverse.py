"""Inverse-action auxiliary representation learning, with no intrinsic reward.

Uses only the inverse-classification idea from Pathak et al. (1705.05363),
not ICM's forward prediction bonus or its exploration policy.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .model import Learner


AUX_FILES = ('inverse-head.safetensors', 'inverse-optimizer.npz')
ARCHITECTURE = 'own-adjacent-hidden-pair-v1'


class InverseHead(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.hidden = nn.Linear(512, 256)
        self.output = nn.Linear(256, action_count)

    def __call__(self, before, after):
        return self.output(nn.relu(self.hidden(mx.concatenate((before, after), axis=1))))


class InverseTrainingModel(nn.Module):
    def __init__(self, q, auxiliary):
        super().__init__()
        self.q, self.aux = q, auxiliary


class InverseLearner(Learner):
    def __init__(self, learning_rate=1e-4, seed=0, action_count=20, weight=.01):
        if not np.isfinite(weight) or not 0 < weight <= 10:
            raise ValueError('invalid inverse weight')
        super().__init__(learning_rate, seed, action_count)
        self.weight = weight
        self.aux = InverseHead(action_count)
        self.aux_optimizer = optim.Adam(learning_rate=learning_rate, eps=1e-5)
        self.aux_optimizer.init(self.aux.trainable_parameters())
        self.training_model = InverseTrainingModel(self.online, self.aux)
        self.local_updates = self.examples = 0
        self.last_loss = self.last_accuracy = 0.
        self.rebind()

    def rebind(self):
        self.state = [self.online.state, self.target.state, self.optimizer.state,
                      self.aux.state, self.aux_optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(lambda x: mx.argmax(self.online(x), axis=1), inputs=self.online.state)

    def restore_auxiliary(self, directory, saved=None):
        if directory is not None:
            from .defense_learning import sha256
            directory = Path(directory)
            hashes = (saved or {}).get('inverse_auxiliary_hashes', {})
            if ((saved or {}).get('config', {}).get('inverse_architecture') != ARCHITECTURE
                    or set(hashes) != set(AUX_FILES)):
                raise ValueError('inverse resume requires complete compatible auxiliary provenance')
            for name in AUX_FILES:
                if not (directory/name).is_file() or sha256(directory/name) != hashes[name]:
                    raise ValueError('inverse auxiliary checkpoint missing or checksum mismatch: '+name)
            self.aux.load_weights(str(directory/AUX_FILES[0]))
            self.aux_optimizer.state = tree_unflatten(list(mx.load(str(directory/AUX_FILES[1])).items()))
        self.aux_optimizer.learning_rate = self.optimizer.learning_rate
        self.rebind()

    def save_auxiliary(self, directory):
        from .defense_learning import sha256
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory/'inverse-head.tmp.safetensors'
        self.aux.save_weights(str(temporary)); temporary.replace(directory/AUX_FILES[0])
        temporary = directory/'inverse-optimizer.tmp.npz'
        mx.savez(str(temporary), **dict(tree_flatten(self.aux_optimizer.state)))
        temporary.replace(directory/AUX_FILES[1])
        return {name: sha256(directory/name) for name in AUX_FILES}

    def auxiliary_loss(self, model, obs, actions, immediate):
        logits = model.aux(model.q.features(obs), model.q.features(immediate))
        return nn.losses.cross_entropy(logits, actions, reduction='none'), mx.mean(mx.argmax(logits, axis=1) == actions)

    def _joint_loss(self, model, obs, actions, returns, following, discounts, weights, immediate):
        td_loss, (errors, q) = Learner._loss(self, model.q, obs, actions, returns, following, discounts, weights)
        inverse, accuracy = self.auxiliary_loss(model, obs, actions, immediate)
        loss = mx.mean(weights*inverse)
        return td_loss+self.weight*loss, (errors, q, loss, accuracy)

    def _update(self, *batch):
        (loss, (errors, q, inverse, accuracy)), grads = nn.value_and_grad(
            self.training_model, self._joint_loss)(self.training_model, *batch)
        grads, norm = optim.clip_grad_norm(grads, max_norm=10.)
        self.optimizer.update(self.online, grads['q'])
        self.aux_optimizer.update(self.aux, grads['aux'])
        return loss, errors, q, norm, inverse, accuracy

    def train(self, batch):
        result = self.update(*(mx.array(x) for x in batch))
        mx.eval(result, self.state)
        self.local_updates += 1
        self.examples += len(batch[0])
        self.last_loss = float(result[4].item())
        self.last_accuracy = float(result[5].item())
        return float(result[0].item()), np.array(result[1]), float(result[2].item())

    def inverse_stats(self):
        return dict(local_updates=self.local_updates, own_pairs=self.examples, weight=self.weight,
                    last_weighted_cross_entropy=self.last_loss, last_sampled_accuracy=self.last_accuracy)
