"""Frozen Defense action/history timing diagnostic, excluded from promotion."""

import argparse
import json
from pathlib import Path

from .defense_learning import evaluate, load_policy, policy_description, sha256
from .temporal_probe import TemporalPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stride", type=int, choices=(1, 2), required=True)
    parser.add_argument("--tstates", type=int, choices=(50_000, 100_000), default=50_000)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.games <= 10 or not 10_000 <= args.seed <= 10_010-args.games:
        parser.error("this diagnostic is restricted to reused primary seeds 10000-10009")
    if args.envs < 1:
        parser.error("--envs must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    config = json.loads((args.checkpoint.parent/"state.json").read_text())["config"]
    if config.get("algorithm", "ppo") != "ppo":
        parser.error("this diagnostic requires an original categorical PPO checkpoint")
    before = sha256(args.checkpoint)
    policy, checked_config = load_policy(args.checkpoint)
    if checked_config != config:
        raise RuntimeError("checkpoint configuration changed while loading")
    wrapped = TemporalPolicy(policy, stride=args.stride)
    result = evaluate(wrapped, range(args.seed, args.seed+args.games),
                      tstates=args.tstates, max_steps=0, envs=args.envs,
                      allow_enter=config.get("allow_enter", False),
                      log=lambda row: print(json.dumps(row), flush=True))
    if sha256(args.checkpoint) != before:
        raise RuntimeError("checkpoint changed during evaluation")
    result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=before,
                  game_sha256=config["game_sha256"], environment_version=config["environment_version"],
                  policy=policy_description(config), deterministic_override=False,
                  evaluation_only=True, promotion_eligible=False, parameter_updates=0,
                  tstates=args.tstates, observation_stride=args.stride,
                  nominal_history_span_tstates=3*args.stride*args.tstates,
                  checkpoint_tstates=config["tstates"], eval_max_steps=0,
                  probe_source_sha256=sha256(Path(__file__)),
                  history_wrapper_source_sha256=sha256(Path(__file__).with_name("temporal_probe.py")),
                  limitations=["Nominal history span excludes variable HUD/terminal settling.",
                               "Frozen-policy timing probe, not learning at a new control rate.",
                               "Reused validation seeds; not a fresh success-rate test."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "games"}), flush=True)
    if result["incomplete_games"]:
        raise SystemExit("Diagnostic incomplete; the entire suite is disqualified")


if __name__ == "__main__":
    main()
