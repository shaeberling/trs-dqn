"""Fixed grouped categorical action head for Defense PPO.

The model still sees only rendered screens and emits twenty learned logits.
Stage-one fire aliases 9..17 are summed as one categorical choice and sent
through canonical Space. Commands 0..8 and 18..19 stay distinct. The mapping
is fixed for every screen; it never examines stage, score or native memory.
"""

import numpy as np


COMMAND_MAP = np.array((*range(10), 18, 19), dtype=np.int32)


def group_logits_numpy(logits):
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("canonical fire requires finite twenty-command logits")
    fire = np.logaddexp.reduce(values[:, 9:18], axis=1, keepdims=True)
    return np.concatenate((values[:, :9], fire, values[:, 18:20]), axis=1)


def group_logits_mlx(logits):
    import mlx.core as mx
    return mx.concatenate((logits[:, :9], mx.logsumexp(logits[:, 9:18], axis=1, keepdims=True),
                           logits[:, 18:20]), axis=1)


def learning_indices(actions):
    """Map emitted keyboard IDs back to grouped likelihood indices."""
    import mlx.core as mx
    return mx.where(actions >= 18, actions-8, actions)


class CanonicalFirePolicy:
    """Evaluation sampler for the grouped learned categorical distribution."""

    def __init__(self, infer_logits, temperature=1.0):
        from .temperature_probe import temperature_policy
        self.inner = temperature_policy(lambda obs: group_logits_numpy(infer_logits(obs)), temperature)

    def __call__(self, obs):
        return COMMAND_MAP[self.inner(obs)]

    def sample_with_rngs(self, obs, rngs):
        return COMMAND_MAP[self.inner.sample_with_rngs(obs, rngs)]

    def reset_seed(self, seed):
        self.inner.reset_seed(seed)
