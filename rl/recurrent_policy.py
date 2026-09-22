"""Screen-history memory bookkeeping; no emulator state or MLX imports."""

import numpy as np

RECURRENT_ARCHITECTURE = "screen-cnn-residual-gru-v1"


def sequence_batches(array, length):
    """Time,worker,... -> chunk*worker,time,... without crossing worker streams."""
    array = np.asarray(array)
    if (isinstance(length, bool) or not isinstance(length, int) or length < 1
            or array.ndim < 2 or array.shape[0] % length):
        raise ValueError("sequence length must divide the time dimension")
    chunks, workers = array.shape[0] // length, array.shape[1]
    shaped = array.reshape(chunks, length, workers, *array.shape[2:])
    return shaped.swapaxes(1, 2).reshape(chunks * workers, length, *array.shape[2:])


class RecurrentPolicy:
    """Independent neural memory per evaluation RNG/game, never per worker slot."""
    def __init__(self, infer, hidden_size, seed=0, temperature=1.):
        if (isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1
                or isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("invalid recurrent policy configuration")
        self.infer, self.hidden_size, self.temperature = infer, hidden_size, temperature
        self.reset_seed(seed)

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.memories = {}  # Generator keys retain identity; no recycled integer IDs.
        self.serial_memory = None

    def choose(self, obs, hidden, uniforms):
        logits, next_hidden = self.infer(obs, hidden)
        logits, next_hidden = np.asarray(logits), np.asarray(next_hidden)
        if (logits.ndim != 2 or logits.shape[0] != len(obs) or not logits.shape[1]
                or next_hidden.shape != (len(obs), self.hidden_size)
                or not np.isfinite(logits).all() or not np.isfinite(next_hidden).all()):
            raise ValueError("invalid recurrent policy output")
        logits = logits / self.temperature
        probs = np.exp(logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True))
        actions = (uniforms[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, logits.shape[1]-1)
        return actions, next_hidden.copy()

    def __call__(self, obs):
        if self.serial_memory is None:
            self.serial_memory = np.zeros((len(obs), self.hidden_size), np.float32)
        if len(self.serial_memory) != len(obs):
            raise ValueError("reset_seed before changing serial stream count")
        actions, self.serial_memory = self.choose(obs, self.serial_memory, self.rng.random(len(obs)))
        return actions

    def sample_with_rngs(self, obs, rngs):
        if len(obs) != len(rngs) or len(set(rngs)) != len(rngs):
            raise ValueError("one distinct policy RNG required per game")
        hidden = np.stack([self.memories.get(rng, np.zeros(self.hidden_size, np.float32)) for rng in rngs])
        actions, updated = self.choose(obs, hidden, np.array([rng.random() for rng in rngs]))
        for rng, memory in zip(rngs, updated, strict=True):
            self.memories[rng] = memory.copy()
        return actions
