"""Optional own-encoder transfer for a fresh Defense learner, not a resume.

Only learned screen-encoder parameters are copied. Heads, optimizer, counters,
RNGs and emulator episodes remain fresh. No trajectories or native states load.
"""

import hashlib
import json
from pathlib import Path

from .defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names


def initialize_encoder(model, checkpoint, *, allow_enter=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    checkpoint = Path(checkpoint)
    state_path, weights_path = checkpoint / "state.json", checkpoint / "model.safetensors"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    config = state.get("config", {})
    if (config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("allow_enter", False) != allow_enter
            or config.get("action_names") != list(action_names(allow_enter))):
        raise ValueError("Encoder initialization requires a compatible own Defense checkpoint")
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    source = mx.load(str(weights_path))
    expected = dict(tree_flatten(model.parameters()))
    if source.keys() != expected.keys():
        raise ValueError("Encoder initialization parameter names differ")
    copied = []
    for name, target in expected.items():
        value = source[name]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Encoder initialization parameter shape/dtype differs: {name}")
        if name.startswith(("conv.", "hidden.")):
            if not bool(mx.all(mx.isfinite(value)).item()):
                raise ValueError(f"Encoder initialization contains non-finite weights: {name}")
            copied.append((name, value))
    if (state_path.read_bytes() != state_bytes
            or hashlib.sha256(weights_path.read_bytes()).hexdigest() != digest):
        raise ValueError("Encoder initialization source changed while loading; use an immutable checkpoint")
    model.update(tree_unflatten(copied))
    mx.eval(model.state)
    return dict(method="own screen encoder only; fresh actor, value head and optimizer",
                source_checkpoint=str(checkpoint), source_model_sha256=digest,
                source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                source_training_steps=state.get("steps"),
                copied_parameters=[name for name, _ in copied],
                frozen_parameters=False, trajectories_loaded=False, native_states_loaded=False)
