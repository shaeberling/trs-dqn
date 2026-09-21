"""Uniformly average compatible PPO checkpoints into an evaluation-only model."""

import argparse
import hashlib
import json
from pathlib import Path


def mean_parameters(parameters):
    """Incremental FP32 mean; no policy outputs, actions, or game data are mixed."""
    import mlx.core as mx

    if not parameters:
        raise ValueError("provide at least one parameter set")
    first = parameters[0]
    if not first:
        raise ValueError("empty parameter set")
    means = {}
    for index, current in enumerate(parameters):
        if current.keys() != first.keys():
            raise ValueError("checkpoint parameter keys differ")
        for key, value in current.items():
            if value.shape != first[key].shape or value.dtype != mx.float32:
                raise ValueError(f"incompatible FP32 parameter: {key}")
            if not bool(mx.all(mx.isfinite(value)).item()):
                raise ValueError(f"non-finite parameter: {key}")
            means[key] = value if index == 0 else means[key]+(value-means[key])/(index+1)
        mx.eval(means)
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in means.values()):
        raise ValueError("non-finite average")
    return means


def average_checkpoints(checkpoints, output):
    paths = [Path(path).resolve(strict=True) for path in checkpoints]
    output = Path(output)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct checkpoint files")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    states = [json.loads((path.parent/"state.json").read_text()) for path in paths]
    config = states[0]["config"]
    if config.get("algorithm") != "ppo" or any(state["config"] != config for state in states):
        raise ValueError("average checkpoints from one identical PPO run configuration")

    import mlx.core as mx
    from .model import QNetwork
    from .train import write_json

    sources, parameters = [], []
    for path, state in zip(paths, states, strict=True):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parameters.append(mx.load(str(path)))
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source changed while loading: {path}")
        sources.append(dict(checkpoint=str(path), sha256=digest, steps=state["steps"]))
    weights = mean_parameters(parameters)
    model = QNetwork()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.state)
    state = dict(steps=max(item["steps"] for item in states),
                 config={**config, "run": str(output), "checkpoint_kind": "uniform-parameter-average"},
                 best_mean=-1.0, best_level_rank=None,
                 averaging=dict(method="incremental uniform FP32 parameter mean",
                                sources=sources, weights=[1/len(paths)]*len(paths),
                                additional_training_actions=0),
                 evaluation_only=True, resume_supported=False)
    output.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(output/"model.tmp.safetensors"))
    (output/"model.tmp.safetensors").replace(output/"model.safetensors")
    write_json(output/"state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(average_checkpoints(args.checkpoints, args.output), indent=2))


if __name__ == "__main__":
    main()
