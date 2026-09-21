"""Restricted evaluation-only sampling diagnostic; never trains or promotes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import categorical_policy, checkpoint_config, evaluate


def temperature_policy(infer_logits, temperature):
    if isinstance(temperature, (bool, np.bool_)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if temperature == 1:
        return categorical_policy(infer_logits)
    return categorical_policy(lambda obs: np.asarray(infer_logits(obs))/temperature)


def load_temperature_policy(checkpoint, temperature):
    import mlx.core as mx
    from .model import QNetwork

    model = QNetwork()
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    predict = mx.compile(model.policy_value, inputs=model.state)
    return temperature_policy(lambda obs: np.array(predict(mx.array(obs))[0]), temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--temperature", type=float, choices=(0.8, 1.0), required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--envs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.games <= 20 or not 10000 <= args.seed <= 10020-args.games:
        parser.error("diagnostic restricted to reused primary seeds10000-10019")
    if args.envs < 1:
        parser.error("--envs must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    config = checkpoint_config(args.checkpoint)
    if config.get("algorithm") != "ppo":
        parser.error("diagnostic requires a categorical PPO checkpoint")
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy = load_temperature_policy(args.checkpoint, args.temperature)
    tstates = config.get("tstates", 100000)
    result = evaluate(policy, range(args.seed, args.seed+args.games),
                      envs=args.envs, tstates=tstates, max_steps=10000000000//tstates,
                      observation_stride=config.get("observation_stride", 1), verbose=True)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != before:
        raise RuntimeError("checkpoint changed during diagnostic")
    result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=before,
                  policy="learned categorical logits, temperature-scaled sampling",
                  temperature=args.temperature, deterministic_override=False,
                  evaluation_only=True, promotion_eligible=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "games"}), flush=True)
    if result["incomplete_games"]:
        raise SystemExit("Diagnostic incomplete; the entire suite is disqualified")


if __name__ == "__main__":
    main()
