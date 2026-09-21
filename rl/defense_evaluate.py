"""Evaluate a frozen screen-only Defense checkpoint on complete fresh games."""

import argparse
from pathlib import Path
import math
import json
import shutil

import numpy as np

from .defense import ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import (evaluate, game_rank, load_policy, record_game, sha256,
                              verify_policy_trace, write_json, policy_description)
from .defense_smoke import write_replay


def record_probe(checkpoint, evaluation, output):
    """Preserve an explicit sampling diagnostic; never replace the trained best."""
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError("probe output already exists")
    if evaluation["temporal_override"] or evaluation["incomplete_games"]:
        raise ValueError("probe recording requires complete games at checkpoint timing")
    if sha256(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("probe weights differ from evaluation")
    temperature = evaluation["temperature"]
    policy, config = load_policy(checkpoint, temperature=temperature)
    candidate = max(evaluation["games"], key=game_rank)
    frames, actions, rewards, result, events = record_game(
        policy, candidate["seed"], tstates=config["tstates"], max_steps=0,
        allow_enter=config.get("allow_enter", False),
        observation_stride=config.get("observation_stride", 1))
    if result != candidate:
        raise RuntimeError("probe replay differs from evaluation")
    verification = verify_policy_trace(checkpoint, frames, actions, rewards, result,
                                      temperature=temperature)
    if sha256(checkpoint) != evaluation["checkpoint_sha256"]:
        raise RuntimeError("probe weights changed during verification")
    metadata = dict(game="Obstacle Run / Missile Defense", game_sha256=GAME_SHA256,
                    environment_version=ENVIRONMENT_VERSION, action_names=config["action_names"],
                    policy=evaluation["policy"], temperature=temperature, trained_model=True,
                    evaluation_only=True, promotion_eligible=False,
                    checkpoint_sha256=evaluation["checkpoint_sha256"],
                    verified_actions=len(actions), tstates=config["tstates"], max_steps=0,
                    result=result, events=events, observation_stride=config.get("observation_stride", 1))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("model.safetensors", "state.json"):
        shutil.copy2(checkpoint.parent/name, output/name)
    write_json(output/"evaluation.json", evaluation)
    write_json(output/"verification.json", verification)
    np.savez_compressed(output/"trace.npz", frames=frames, actions=actions, rewards=rewards,
                        metadata=json.dumps(metadata))
    write_replay(output/"replay.html", frames, actions, metadata)
    write_json(output/"manifest.json", dict(metadata=metadata,
               hashes={p.name: sha256(p) for p in sorted(output.iterdir())}))


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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="evaluation-only sampling probe; 1 preserves the trained policy")
    parser.add_argument("--replay-output", type=Path,
                        help="new separate verified probe bundle; never promotes to trained best")
    args = parser.parse_args()
    if min(args.games, args.envs) < 1 or args.max_steps < 0:
        parser.error("games/envs must be positive; max-steps must be nonnegative")
    if args.tstates is not None and not 1 <= args.tstates <= 1_000_000:
        parser.error("tstates must be between 1 and 1,000,000")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if args.output.exists():
        parser.error("output exists; use a fresh path")
    if args.replay_output and args.replay_output.exists():
        parser.error("replay output exists; use a fresh path")
    before = sha256(args.checkpoint)
    policy, config = load_policy(args.checkpoint, temperature=args.temperature)
    tstates = config["tstates"] if args.tstates is None else args.tstates
    if args.replay_output and tstates != config["tstates"]:
        parser.error("probe replay requires original checkpoint timing")
    result = evaluate(policy, range(args.seed, args.seed+args.games), tstates=tstates,
                      max_steps=args.max_steps, envs=args.envs, log=lambda row: print(row, flush=True),
                      allow_enter=config.get("allow_enter", False),
                      observation_stride=config.get("observation_stride", 1))
    if sha256(args.checkpoint) != before:
        raise RuntimeError("Checkpoint changed during evaluation; use frozen weights")
    result.update(checkpoint_sha256=before, config=config,
                  policy=policy_description(config, args.temperature),
                  temperature=args.temperature, evaluation_only=True,
                  promotion_eligible=False, evaluation_seed=args.seed,
                  evaluation_tstates=tstates, training_tstates=config["tstates"],
                  temporal_override=tstates != config["tstates"],
                  observation_stride=config.get("observation_stride", 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k not in ("games", "config")})
    if result["incomplete_games"]:
        raise SystemExit("Incomplete evaluation; truncated games do not count as complete results")
    if args.replay_output:
        record_probe(args.checkpoint, result, args.replay_output)


if __name__ == "__main__":
    main()
