"""Screen-only PPO with learned physical key and hold duration.

Only option starts are policy decisions. Forced continuation steps still
provide real visible-score rewards and critic targets, but are masked out of
the actor/entropy/KL objectives. The executor reads only its own chosen option
and visible life boundaries; it never detects obstacles or steers movement.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from .defense_repeat import validate_spec
from .ppo import PPO


def duration_action_names(durations, action_count=20):
    count, durations = validate_spec(action_count, durations)
    names = action_names(False)
    if count != len(names):
        raise ValueError("duration policy requires original twenty physical commands")
    return tuple(f"{name}@{duration}" for duration in durations for name in names)


class DurationPPO(PPO):
    """PPO whose actor loss is evaluated only at real option decisions."""

    def _loss(self, model, obs, actions, old_logp, advantages, returns, actor_mask,
              reference_log_probs=None, logit_bias=None, head_weight_noise=None):
        import mlx.core as mx

        if head_weight_noise is None:
            logits, values = model.policy_value(obs)
        else:
            logits, values = model.policy_value(obs, head_weight_noise=head_weight_noise)
        if logit_bias is not None:
            logits = logits + mx.stop_gradient(logit_bias)
        log_probs = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        logp = mx.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
        mask = actor_mask.astype(mx.float32)
        decision_count = mx.maximum(mx.sum(mask), 1.)
        logratio = mx.where(mask > 0, logp-old_logp, 0.)
        ratio = mx.exp(logratio)
        actor = -mx.sum(mask*mx.minimum(ratio*advantages,
                                        mx.clip(ratio, .8, 1.2)*advantages))/decision_count
        critic = .5*mx.mean(mx.square(values-returns))
        per_sample_entropy = -mx.sum(mx.exp(log_probs)*log_probs, axis=-1)
        entropy = mx.sum(mask*per_sample_entropy)/decision_count
        kl = mx.sum(mask*((ratio-1)-logratio))/decision_count
        if reference_log_probs is not None or self.reference_kl_weight or self.canonical_fire:
            raise ValueError("duration PPO does not combine with reference or grouped actions")
        return actor+self.value_coefficient*critic-self.entropy*entropy, (actor, critic, entropy, kl)

    def _update(self, obs, actions, old_logp, advantages, returns, actor_mask,
                reference_log_probs=None, logit_bias=None, head_weight_noise=None):
        import mlx.nn as nn
        import mlx.optimizers as optim

        (loss, metrics), grads = nn.value_and_grad(self.model, self._loss)(
            self.model, obs, actions, old_logp, advantages, returns, actor_mask,
            reference_log_probs, logit_bias, head_weight_noise)
        grads, _ = optim.clip_grad_norm(grads, .5)
        self.optimizer.update(self.model, grads)
        return loss, metrics

    def act(self, obs, rng, *, actor_mask, logit_bias=None, head_weight_noise=None):
        mask = np.asarray(actor_mask)
        if mask.shape != (len(obs),) or mask.dtype != np.bool_:
            raise ValueError("one boolean real-decision mask required per worker")
        # Draws on forced steps are discarded; they are never executed or
        # assigned actor likelihood. This retains the standard vectorized
        # inference path and keeps all value predictions at base cadence.
        actions, logp, values = super().act(obs, rng, logit_bias=logit_bias,
                                            head_weight_noise=head_weight_noise)
        actions[~mask] = 0
        logp[~mask] = 0.
        return actions, logp, values


class DurationCategoricalPolicy:
    """Sample a joint option only when its own previous hold expires."""

    def __init__(self, policy, durations, action_count=20):
        self.action_count, self.durations = validate_spec(action_count, durations)
        self.policy = policy
        self.reset_seed(0)

    def reset_seed(self, seed):
        self.memories = {}
        self.serial_keys = None
        if hasattr(self.policy, "reset_seed"):
            self.policy.reset_seed(seed)

    def _execute(self, obs, keys, *, rngs=None):
        if len(obs) != len(keys) or len(set(keys)) != len(keys):
            raise ValueError("one distinct game identity required per observation")
        idle = [i for i, key in enumerate(keys) if self.memories.get(key, (0, 0))[1] == 0]
        if idle:
            subset = np.asarray(obs)[idle]
            choices = (self.policy(subset) if rngs is None else
                       self.policy.sample_with_rngs(subset, [rngs[i] for i in idle]))
            choices = np.asarray(choices)
            if (choices.shape != (len(idle),) or
                    not np.issubdtype(choices.dtype, np.integer) or
                    np.any(choices < 0) or
                    np.any(choices >= self.action_count*len(self.durations))):
                raise ValueError("invalid learned joint option")
            for index, option in zip(idle, choices, strict=True):
                option = int(option)
                self.memories[keys[index]] = (option, self.durations[option//self.action_count])
        physical = []
        for key in keys:
            option, remaining = self.memories[key]
            physical.append(option % self.action_count)
            self.memories[key] = (option, remaining-1)
        return np.asarray(physical, np.int32)

    def __call__(self, obs):
        if self.serial_keys is None:
            self.serial_keys = [object() for _ in obs]
        if len(self.serial_keys) != len(obs):
            raise ValueError("reset_seed before changing serial stream count")
        return self._execute(obs, self.serial_keys)

    def sample_with_rngs(self, obs, rngs):
        return self._execute(obs, rngs, rngs=rngs)

    def observe_boundaries(self, boundaries, rngs=None):
        keys = self.serial_keys if rngs is None else rngs
        boundaries = np.asarray(boundaries)
        if (keys is None or boundaries.shape != (len(keys),) or
                boundaries.dtype != np.bool_ or len(set(keys)) != len(keys)):
            raise ValueError("one boolean visible boundary required per game")
        if hasattr(self.policy, "observe_boundaries"):
            self.policy.observe_boundaries(boundaries, rngs)
        for key, ended in zip(keys, boundaries, strict=True):
            if ended:
                self.memories.pop(key, None)


def initialize_duration_policy(model, checkpoint, durations, *, tstates, observation_stride,
                               logit_spacing=2.):
    """Copy all own PPO features; tile physical actor rows with neutral hold priors."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    count, durations = validate_spec(20, durations)
    if (isinstance(logit_spacing, bool) or not np.isfinite(logit_spacing)
            or not 0 <= logit_spacing <= 20):
        raise ValueError("duration logit spacing must be finite and in 0..20")
    checkpoint = Path(checkpoint)
    state_path, weights_path = checkpoint/"state.json", checkpoint/"model.safetensors"
    state_bytes, weights_bytes = state_path.read_bytes(), weights_path.read_bytes()
    state = json.loads(state_bytes)
    config = state.get("config", {})
    if (config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("algorithm") != "ppo" or config.get("architecture") is not None
            or config.get("canonical_fire") or config.get("repeat_previous_action")
            or config.get("action_names") != list(action_names(False))
            or config.get("tstates") != tstates
            or config.get("observation_stride", 1) != observation_stride):
        raise ValueError("duration initialization requires matching own ordinary Defense PPO")
    source = mx.load(str(weights_path))
    target = dict(tree_flatten(model.parameters()))
    if source.keys() != target.keys():
        raise ValueError("source and target parameter names differ")
    offsets = tuple(-float(logit_spacing)*i for i in range(len(durations)))
    copied = []
    for name, wanted in target.items():
        value = source[name]
        if name in ("advantage.weight", "advantage.bias"):
            if (value.shape[0] != count or wanted.shape[0] != count*len(durations)
                    or value.shape[1:] != wanted.shape[1:]):
                raise ValueError("actor row shape differs")
            value = mx.concatenate([value+offset if name == "advantage.bias" else value
                                    for offset in offsets], axis=0)
        elif value.shape != wanted.shape:
            raise ValueError("base parameter shape differs: " + name)
        if value.dtype != wanted.dtype or not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError("invalid copied parameter: " + name)
        copied.append((name, value))
    if state_path.read_bytes() != state_bytes or weights_path.read_bytes() != weights_bytes:
        raise ValueError("initialization source changed while loading")
    model.update(tree_unflatten(copied))
    mx.eval(model.state)
    return dict(method="own ordinary Defense PPO full policy; copied physical actor rows with direction-neutral duration priors; fresh optimizer/RNG",
                durations=list(durations), duration_logit_spacing=float(logit_spacing),
                duration_logit_offsets=list(offsets),
                source_checkpoint=str(checkpoint),
                source_model_sha256=hashlib.sha256(weights_bytes).hexdigest(),
                source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                source_training_steps=state.get("steps"), trajectories_loaded=False,
                native_states_loaded=False)
