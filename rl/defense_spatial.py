"""Zero-initialized learned spatial residual over an own screen-only PPO.

The residual mixes the existing CNN's 6x16 feature map before its dense
layer. No game object, route, reward, key choice or hidden RAM is encoded.
At initialization the gate is exactly zero, preserving the copied policy.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from .defense_duration_ppo import DurationPPO, duration_action_names
from .defense_spatial_spec import SPATIAL_ARCHITECTURE
from .model import QNetwork
from .ppo import _load_backend


class SpatialDurationNetwork(nn.Module):
    def __init__(self, action_count=80):
        super().__init__()
        if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 1:
            raise ValueError("positive spatial actor action count required")
        self.base = QNetwork(action_count=action_count)
        self.mix1 = nn.Conv2d(32, 32, 5, 1, 2)
        self.mix2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.spatial_scale = mx.array(0., dtype=mx.float32)

    @property
    def advantage(self):
        return self.base.advantage

    def features(self, screen):
        # This is exactly the original QNetwork screen raster and CNN path.
        x = self.base._table[screen.astype(mx.int32)]
        x = x.transpose(0, 2, 4, 3, 5, 1, 6).reshape(-1, 48, 128, 8)
        for layer in self.base.conv:
            x = nn.relu(layer(x))
        mixed = self.mix2(nn.relu(self.mix1(x)))
        x = x + self.spatial_scale*mixed
        return nn.relu(self.base.hidden(x.reshape(x.shape[0], -1)))

    def policy_value(self, screen, head_weight_noise=None):
        x = self.features(screen)
        logits = self.base.advantage(x)
        if head_weight_noise is not None:
            logits = logits + mx.matmul(mx.stop_gradient(head_weight_noise), x[:, :, None])[:, :, 0]
        return logits, self.base.value(x)[:, 0]


class SpatialDurationPPO(DurationPPO):
    def __init__(self, seed=41, learning_rate=2.5e-5, entropy=.002,
                 action_count=80, value_coefficient=.5, canonical_fire=False):
        if canonical_fire:
            raise ValueError("spatial duration PPO cannot use grouped canonical-fire actions")
        _load_backend()  # inherited save() uses the ordinary PPO backend globals
        mx.random.seed(seed)
        self.model = SpatialDurationNetwork(action_count)
        self.optimizer = optim.Adam(learning_rate, eps=1e-5)
        self.optimizer.init(self.model.trainable_parameters())
        self.entropy, self.value_coefficient = entropy, value_coefficient
        self.reference_kl_weight, self.canonical_fire = 0., False
        self.compile()


def initialize_duration_checkpoint(model, checkpoint, durations, *, tstates,
                                   observation_stride, spatial, extend_longest=False,
                                   appended_logit_offset=-2.):
    """Copy an own trained duration policy, optionally adding one neutral long hold."""
    checkpoint = Path(checkpoint)
    state_path, weights_path = checkpoint/"state.json", checkpoint/"model.safetensors"
    state_bytes, weights_bytes = state_path.read_bytes(), weights_path.read_bytes()
    config = json.loads(state_bytes)["config"]
    names = action_names(False)
    source_durations = config.get("learned_durations")
    if not isinstance(extend_longest, bool) or (extend_longest and spatial):
        raise ValueError("longest-duration extension requires an ordinary duration target")
    if (isinstance(appended_logit_offset, bool)
            or not np.isfinite(appended_logit_offset) or not -4. <= appended_logit_offset <= 4.
            or (not extend_longest and appended_logit_offset != -2.)):
        raise ValueError("long-hold logit offset must be finite, in -4..4, and require extension")
    if extend_longest:
        if (not isinstance(source_durations, list) or len(durations) != len(source_durations)+1
                or list(durations[:-1]) != source_durations
                or durations[-1] <= source_durations[-1]):
            raise ValueError("extended durations must append one longer hold to the own source")
    elif source_durations != list(durations):
        raise ValueError("own trained duration checkpoint has incompatible hold options")
    if (config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("algorithm") != "ppo" or config.get("architecture") is not None
            or config.get("allow_enter") or config.get("repeat_previous_action")
            or config.get("canonical_fire") or not config.get("life_terminal")
            or config.get("action_names") != list(names)
            or config.get("policy_action_names") != list(duration_action_names(source_durations))
            or config.get("tstates") != tstates
            or config.get("observation_stride", 1) != observation_stride):
        raise ValueError("own trained duration checkpoint has incompatible screen/action protocol")
    if spatial != isinstance(model, SpatialDurationNetwork):
        raise ValueError("spatial initialization target does not match model architecture")
    target = model.base if spatial else model
    source = mx.load(str(weights_path))
    expected = dict(tree_flatten(target.parameters()))
    if source.keys() != expected.keys():
        raise ValueError("own trained duration parameter names differ")
    copied = []
    for name, wanted in expected.items():
        value = source[name]
        if extend_longest and name in ("advantage.weight", "advantage.bias"):
            if (value.shape[0] != len(names)*len(source_durations)
                    or wanted.shape[0] != len(names)*len(durations)
                    or value.shape[1:] != wanted.shape[1:]):
                raise ValueError("extended actor rows differ from own source")
            final_rows = value[-len(names):]
            if name == "advantage.bias":
                final_rows = final_rows+float(appended_logit_offset)
            value = mx.concatenate((value, final_rows), axis=0)
        if (value.shape != wanted.shape or value.dtype != wanted.dtype
                or not bool(mx.all(mx.isfinite(value)).item())):
            raise ValueError("invalid own duration parameter: " + name)
        copied.append((name, value))
    if state_path.read_bytes() != state_bytes or weights_path.read_bytes() != weights_bytes:
        raise RuntimeError("source checkpoint changed during duration initialization")
    target.update(tree_unflatten(copied))
    mx.eval(model.state)
    return dict(method="own full trained key-duration policy; fresh optimizer and RNG; "
                       + ("zero-output learned spatial residual" if spatial else
                          "uniformly penalized extra long option" if extend_longest else
                          "unchanged CNN control"),
                source_checkpoint=str(checkpoint),
                source_durations=source_durations, target_durations=list(durations),
                appended_logit_offset=float(appended_logit_offset) if extend_longest else None,
                source_model_sha256=hashlib.sha256(weights_bytes).hexdigest(),
                source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                source_training_steps=json.loads(state_bytes)["steps"],
                copied_parameters=[name for name, _ in copied],
                trajectories_loaded=False, native_states_loaded=False)
