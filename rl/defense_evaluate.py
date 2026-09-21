"""Evaluate a frozen screen-only Defense checkpoint on complete fresh games."""

import argparse
from pathlib import Path

from .defense_learning import evaluate, load_policy, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--tstates", type=int,
                        help="explicit temporal probe; default uses checkpoint action duration")
    args = parser.parse_args()
    if min(args.games, args.envs) < 1 or args.max_steps < 0:
        parser.error("games/envs must be positive; max-steps must be nonnegative")
    if args.tstates is not None and not 1 <= args.tstates <= 1_000_000:
        parser.error("tstates must be between 1 and 1,000,000")
    if args.output.exists():
        parser.error("output exists; use a fresh path")
    policy, config = load_policy(args.checkpoint)
    tstates = config["tstates"] if args.tstates is None else args.tstates
    result = evaluate(policy, range(args.seed, args.seed+args.games), tstates=tstates,
                      max_steps=args.max_steps, envs=args.envs, log=lambda row: print(row, flush=True),
                      allow_enter=config.get("allow_enter", False))
    result.update(checkpoint_sha256=sha256(args.checkpoint), config=config,
                  policy="learned categorical, sampled", evaluation_seed=args.seed,
                  evaluation_tstates=tstates, training_tstates=config["tstates"],
                  temporal_override=tstates != config["tstates"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k not in ("games", "config")})
    if result["incomplete_games"]:
        raise SystemExit("Incomplete evaluation; truncated games do not count as complete results")


if __name__ == "__main__":
    main()
