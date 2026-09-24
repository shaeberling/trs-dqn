"""Learned continue-previous-key option; memory contains only the policy's own key.

The network samples one of the twenty ordinary commands or a twenty-first
CONTINUE choice on every base action. CONTINUE repeats its own last executed
command, initially NOOP, with no inspection of the game or route. The PPO
likelihood is always for the sampled choice, not the resulting physical key.
"""

import numpy as np

from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256


CONTINUE = 20
POLICY_ACTION_NAMES = action_names(False) + ("CONTINUE_PREVIOUS",)


class RepeatPreviousActions:
    """Independent own-action memory for one or more complete game streams."""

    def __init__(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive stream count required")
        self.previous = np.zeros(count, np.int32)

    def execute(self, choices):
        choices = np.asarray(choices)
        if (choices.shape != self.previous.shape
                or not np.issubdtype(choices.dtype, np.integer)
                or np.any(choices < 0) or np.any(choices > CONTINUE)):
            raise ValueError("one valid sampled choice required per stream")
        physical = np.where(choices == CONTINUE, self.previous, choices).astype(np.int32)
        self.previous[:] = physical
        return physical

    def reset(self, boundaries):
        boundaries = np.asarray(boundaries)
        if boundaries.shape != self.previous.shape or boundaries.dtype != np.bool_:
            raise ValueError("one boolean boundary required per stream")
        self.previous[boundaries] = 0


class RepeatPreviousPolicy:
    """Wrap a screen-only categorical policy for serial and parallel evaluation."""

    def __init__(self, policy):
        self.policy = policy
        self.reset_seed(0)

    def reset_seed(self, seed):
        self.previous = {}
        self.serial_keys = None
        if hasattr(self.policy, "reset_seed"):
            self.policy.reset_seed(seed)

    def _execute(self, choices, keys):
        choices = np.asarray(choices)
        if (choices.shape != (len(keys),) or len(set(keys)) != len(keys)
                or not np.issubdtype(choices.dtype, np.integer)
                or np.any(choices < 0) or np.any(choices > CONTINUE)):
            raise ValueError("one valid sampled choice and identity required per game")
        physical = []
        for choice, key in zip(choices, keys, strict=True):
            action = self.previous.get(key, 0) if choice == CONTINUE else int(choice)
            self.previous[key] = action
            physical.append(action)
        return np.asarray(physical, np.int32)

    def __call__(self, obs):
        if self.serial_keys is None:
            self.serial_keys = [object() for _ in obs]
        if len(self.serial_keys) != len(obs):
            raise ValueError("reset_seed before changing serial stream count")
        return self._execute(self.policy(obs), self.serial_keys)

    def sample_with_rngs(self, obs, rngs):
        if len(obs) != len(rngs):
            raise ValueError("one RNG per game required")
        return self._execute(self.policy.sample_with_rngs(obs, rngs), rngs)

    def observe_boundaries(self, boundaries, rngs=None):
        keys = self.serial_keys if rngs is None else rngs
        boundaries = np.asarray(boundaries)
        if (keys is None or boundaries.shape != (len(keys),)
                or boundaries.dtype != np.bool_ or len(set(keys)) != len(keys)):
            raise ValueError("one boolean boundary required per game")
        if hasattr(self.policy, "observe_boundaries"):
            self.policy.observe_boundaries(boundaries, rngs)
        for key, ended in zip(keys, boundaries, strict=True):
            if ended:
                self.previous.pop(key, None)


def initialize_repeat_policy(model, checkpoint, *, tstates, observation_stride,
                             bias_offset=0.):
    """Copy an own PPO; seed CONTINUE from the mean actor row plus a neutral bias."""
    import hashlib
    import json
    from pathlib import Path

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    if (isinstance(bias_offset, bool) or not np.isfinite(bias_offset)
            or not 0 <= bias_offset <= 10):
        raise ValueError("Continue bias offset must be finite and in 0..10")

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
        raise ValueError("Continue-action initialization requires matching own ordinary Defense PPO")
    source = mx.load(str(weights_path))
    target = dict(tree_flatten(model.parameters()))
    if source.keys() != target.keys():
        raise ValueError("Source and target parameter names differ")
    copied = []
    for name, wanted in target.items():
        value = source[name]
        if name in ("advantage.weight", "advantage.bias"):
            if (value.shape[0] != CONTINUE or wanted.shape[0] != CONTINUE+1
                    or value.shape[1:] != wanted.shape[1:]):
                raise ValueError("Actor row shape differs")
            continuation = mx.mean(value, axis=0, keepdims=True)
            if name == "advantage.bias":
                continuation = continuation + bias_offset
            value = mx.concatenate((value, continuation), axis=0)
        elif value.shape != wanted.shape:
            raise ValueError("Base parameter shape differs: " + name)
        if value.dtype != wanted.dtype or not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError("Invalid copied parameter: " + name)
        copied.append((name, value))
    if state_path.read_bytes() != state_bytes or weights_path.read_bytes() != weights_bytes:
        raise ValueError("Initialization source changed while loading")
    model.update(tree_unflatten(copied))
    mx.eval(model.state)
    return dict(method="own ordinary PPO full policy; mean-actor CONTINUE row with direction-neutral bias; fresh optimizer/RNG",
                continue_initial_bias_offset=float(bias_offset),
                source_checkpoint=str(checkpoint),
                source_model_sha256=hashlib.sha256(weights_bytes).hexdigest(),
                source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                source_training_steps=state.get("steps"),
                trajectories_loaded=False, native_states_loaded=False)
