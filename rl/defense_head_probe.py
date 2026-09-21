"""Frozen bootstrap-head diagnostic; never trains or promotes a replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_learning import BOOTSTRAP_ALGORITHM, evaluate, greedy_policy, load_policy, sha256


def head_policy(infer_heads, head, heads):
    """Select one fixed learned head for an entire game, retaining its prior."""
    if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < heads:
        raise ValueError("invalid fixed head")

    def values(obs):
        q = np.asarray(infer_heads(obs))
        if q.ndim != 3 or q.shape[:2] != (len(obs), heads) or not np.isfinite(q).all():
            raise ValueError("invalid head values")
        return q[:, head, :]

    return greedy_policy(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.envs < 1:
        parser.error("--envs must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    before = sha256(args.checkpoint)
    state_before = sha256(args.checkpoint.parent / "state.json")
    ensemble, config = load_policy(args.checkpoint)
    if config.get("algorithm") != BOOTSTRAP_ALGORITHM:
        parser.error("requires a bootstrap DQN checkpoint")

    import mlx.core as mx
    from .defense_bootstrap import BootstrapQ
    mx.set_cache_limit(256 * 1024 * 1024)
    model = BootstrapQ(len(config["action_names"]), config["bootstrap_heads"],
                       config["bootstrap_prior_scale"])
    model.load_weights(str(args.checkpoint))
    mx.eval(model.state)
    predict = mx.compile(model.head_values, inputs=model.state)
    infer = lambda obs: np.array(predict(mx.array(obs)))
    results = {}
    for name, policy in [("ensemble", ensemble)] + [
            (str(h), head_policy(infer, h, config["bootstrap_heads"]))
            for h in range(config["bootstrap_heads"])]:
        print(json.dumps(dict(event="probe_start", head=name)), flush=True)
        result = evaluate(policy, range(10000, 10010), tstates=config["tstates"],
                          observation_stride=config.get("observation_stride", 1),
                          allow_enter=config.get("allow_enter", False), max_steps=0,
                          envs=args.envs,
                          log=lambda row: print(json.dumps(row), flush=True))
        if result["incomplete_games"]:
            raise RuntimeError("incomplete diagnostic suite; no result published")
        results[name] = result
        print(json.dumps(dict(head=name, **{k: v for k, v in result.items() if k != "games"})),
              flush=True)
    if (sha256(args.checkpoint) != before
            or sha256(args.checkpoint.parent / "state.json") != state_before):
        raise RuntimeError("checkpoint changed during diagnostic")
    saved = args.checkpoint.parent / "evaluation.json"
    if saved.exists() and results["ensemble"]["games"] != json.loads(saved.read_text())["games"]:
        raise RuntimeError("ensemble did not reproduce saved validation games")
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=before,
                  state_sha256=state_before, game_sha256=config["game_sha256"],
                  evaluation_only=True, promotion_eligible=False, parameter_updates=0,
                  seeds=list(range(10000, 10010)), eval_max_steps=0,
                  tstates=config["tstates"], observation_stride=config.get("observation_stride", 1),
                  policy="greedy fixed learned head plus saved prior; no per-state head selection",
                  ensemble_reproduces_saved_games=True if saved.exists() else None,
                  probe_source_sha256=sha256(Path(__file__)),
                  model_source_sha256=sha256(Path(__file__).with_name("defense_bootstrap.py")),
                  environment_source_sha256=sha256(Path(__file__).with_name("defense.py")),
                  results=results,
                  limitations=["Reused validation seeds, not a fresh success-rate estimate.",
                               "Head policies are diagnostics, not the checkpoint's ensemble policy.",
                               "No diagnostic action or game record is used for training."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
