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
    args = parser.parse_args()
    if min(args.games, args.envs) < 1 or args.max_steps < 0:
        parser.error("games/envs must be positive; max-steps must be nonnegative")
    if args.output.exists():
        parser.error("output exists; use a fresh path")
    policy, config = load_policy(args.checkpoint)
    result = evaluate(policy, range(args.seed, args.seed+args.games), tstates=config["tstates"],
                      max_steps=args.max_steps, envs=args.envs, log=lambda row: print(row, flush=True))
    result.update(checkpoint_sha256=sha256(args.checkpoint), config=config,
                  policy="learned categorical, sampled", evaluation_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k not in ("games", "config")})
    if result["incomplete_games"]:
        raise SystemExit("Incomplete evaluation; truncated games do not count as complete results")


if __name__ == "__main__":
    main()
