"""Evaluate a frozen Cosmic Fighter policy on complete original-boot games."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .cosmic import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256
from .cosmic_learning import (evaluate, load_policy, record_game, sha256,
                              verify_policy_trace, write_json)
from .defense_smoke import write_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=80_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args()
    if min(args.games, args.envs) < 1 or args.output.exists() or (
            args.replay_output and args.replay_output.exists()):
        parser.error("games/envs must be positive and output paths must not exist")
    before = sha256(args.checkpoint)
    policy, config = load_policy(args.checkpoint)
    result = evaluate(policy, range(args.seed, args.seed+args.games),
                      tstates=config["tstates"], max_steps=config["eval_max_steps"],
                      observation_stride=config["observation_stride"], envs=args.envs)
    if sha256(args.checkpoint) != before:
        raise RuntimeError("Cosmic checkpoint changed during frozen evaluation")
    result.update(game="cosmic", game_sha256=GAME_SHA256,
                  environment_version=ENVIRONMENT_VERSION,
                  checkpoint_sha256=before, config=config, evaluation_only=True,
                  evaluation_seed=args.seed)
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key not in ("games", "config")})
    if result["incomplete_games"]:
        raise SystemExit("Incomplete Cosmic games do not count as complete results")
    if args.replay_output:
        candidate = max(result["games"], key=lambda game: game["score"])
        frames, actions, rewards, found = record_game(
            policy, candidate["seed"], tstates=config["tstates"],
            max_steps=config["eval_max_steps"],
            observation_stride=config["observation_stride"])
        if found != candidate:
            raise RuntimeError("Cosmic replay differs from frozen evaluation")
        verification = verify_policy_trace(args.checkpoint, frames, actions, rewards, found)
        if sha256(args.checkpoint) != before:
            raise RuntimeError("Cosmic checkpoint changed during replay verification")
        directory = args.replay_output
        directory.mkdir(parents=True)
        metadata = dict(game="Cosmic Fighter", game_sha256=GAME_SHA256,
                        environment_version=ENVIRONMENT_VERSION,
                        action_names=ACTION_NAMES, policy="learned Q-values, greedy",
                        trained_model=True, evaluation_only=True,
                        promotion_eligible=False, temperature=1.0,
                        checkpoint_sha256=before, verified_actions=len(actions),
                        tstates=config["tstates"], max_steps=config["eval_max_steps"],
                        observation_stride=config["observation_stride"], result=found)
        for name in ("model.safetensors", "state.json"):
            shutil.copy2(args.checkpoint.parent/name, directory/name)
        write_json(directory/"evaluation.json", result)
        write_json(directory/"verification.json", verification)
        np.savez_compressed(directory/"trace.npz", frames=frames, actions=actions,
                            rewards=rewards, metadata=json.dumps(metadata))
        write_replay(directory/"replay.html", frames, actions, metadata)
        write_json(directory/"manifest.json", dict(metadata=metadata,
                   hashes={path.name: sha256(path) for path in sorted(directory.iterdir())}))


if __name__ == "__main__":
    main()
