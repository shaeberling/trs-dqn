"""Direction-neutral initialization for a fresh Defense key-duration actor.

Stage-one fire aliases have the same physical effect.  Offset their initial
logits so each of the twelve distinct physical commands gets equal mass for
every duration.  This is an action prior, not a screen-dependent controller.
"""

import numpy as np

from .defense_canonical_fire import COMMAND_MAP
from .defense_repeat import validate_spec


def balanced_duration_initial_bias(durations, duration_weights):
    """Return one actor bias per original key-duration option.

    ``duration_weights`` is the initial categorical mass over hold lengths;
    it must not select a direction.  The global logit constant is immaterial.
    """
    _, durations = validate_spec(20, durations)
    weights = np.array(duration_weights, dtype=np.float64, copy=True)
    if (weights.shape != (len(durations),) or not np.all(np.isfinite(weights))
            or not np.all(weights > 0)):
        raise ValueError("one positive finite weight required per duration")
    with np.errstate(over="ignore"):
        total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("duration weights must have a finite positive total")
    weights /= total
    if not np.all(weights > 0):
        raise ValueError("duration weights must remain positive after normalization")
    physical = np.zeros(20, np.float64)
    physical[9:18] = -np.log(9.)
    return np.concatenate([physical + np.log(weight) for weight in weights]).astype(np.float32)


def group_duration_logits_numpy(logits, duration_count):
    """Group nine stage-one fire aliases within each hold-length row."""
    values = np.asarray(logits)
    if (values.ndim != 2 or not isinstance(duration_count, int)
            or isinstance(duration_count, bool) or duration_count < 1
            or values.shape[1] != 20 * duration_count
            or not np.all(np.isfinite(values))):
        raise ValueError("finite twenty-command logits required for each duration")
    rows = values.reshape(len(values), duration_count, 20)
    fire = np.logaddexp.reduce(rows[:, :, 9:18], axis=2, keepdims=True)
    grouped = np.concatenate((rows[:, :, :9], fire, rows[:, :, 18:]), axis=2)
    return grouped.reshape(len(values), duration_count * 12)


def group_duration_logits_mlx(logits, duration_count):
    """Differentiable grouped logits for score-only option PPO."""
    import mlx.core as mx

    if (len(logits.shape) != 2 or not isinstance(duration_count, int)
            or isinstance(duration_count, bool) or duration_count < 1
            or logits.shape[1] != 20 * duration_count):
        raise ValueError("twenty-command logits required for each duration")
    rows = logits.reshape(logits.shape[0], duration_count, 20)
    fire = mx.logsumexp(rows[:, :, 9:18], axis=2, keepdims=True)
    grouped = mx.concatenate((rows[:, :, :9], fire, rows[:, :, 18:]), axis=2)
    return grouped.reshape(logits.shape[0], duration_count * 12)


def grouped_duration_physical_actions(grouped_actions):
    """Map executor outputs (0..11) to unchanged original keyboard actions."""
    actions = np.asarray(grouped_actions)
    if (not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0)
            or np.any(actions >= len(COMMAND_MAP))):
        raise ValueError("grouped command IDs must be in 0..11")
    return COMMAND_MAP[actions]


def grouped_duration_log_probs_numpy(grouped_logits, duration_count, mix):
    """Exact behavior likelihood with direction-neutral duration exploration."""
    logits = np.asarray(grouped_logits)
    if (logits.ndim != 2 or not isinstance(duration_count, int)
            or isinstance(duration_count, bool) or duration_count < 1
            or logits.shape[1] != 12 * duration_count
            or not np.all(np.isfinite(logits)) or not np.isfinite(mix)
            or not 0 <= mix < 1):
        raise ValueError("invalid grouped option logits or duration mixture")
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    if mix == 0:
        return log_probs
    joint = log_probs.reshape(len(logits), duration_count, 12)
    key = np.logaddexp.reduce(joint, axis=1, keepdims=True)
    return np.logaddexp(joint + np.log1p(-mix),
                        key - np.log(duration_count) + np.log(mix)).reshape(logits.shape)


def grouped_duration_log_probs_mlx(grouped_logits, duration_count, mix):
    """Differentiable copy of the grouped behavior distribution."""
    import mlx.core as mx

    if (len(grouped_logits.shape) != 2 or not isinstance(duration_count, int)
            or isinstance(duration_count, bool) or duration_count < 1
            or grouped_logits.shape[1] != 12 * duration_count
            or not np.isfinite(mix) or not 0 <= mix < 1):
        raise ValueError("invalid grouped option logits or duration mixture")
    log_probs = grouped_logits - mx.logsumexp(grouped_logits, axis=-1, keepdims=True)
    if mix == 0:
        return log_probs
    joint = log_probs.reshape(grouped_logits.shape[0], duration_count, 12)
    key = mx.logsumexp(joint, axis=1, keepdims=True)
    return mx.logaddexp(joint + np.log1p(-mix),
                        key - np.log(duration_count) + np.log(mix)).reshape(grouped_logits.shape)
