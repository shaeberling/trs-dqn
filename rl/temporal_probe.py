"""Evaluation-only action/history probe; never a training or promotion path."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import checkpoint_config, evaluate, load_policy, policy_description


class TemporalPolicy:
    """Space four raw frames without changing the model or its sampling rule.

    The evaluator supplies a distinct RNG object for each game. Those objects
    identify histories only: neither their identity nor seed enters the model.
    Strong dictionary keys prevent identifier reuse when worker slots recycle.
    This short-lived probe retains at most one seven-frame history per game.
    """

    def __init__(self, policy, stride=2):
        if stride not in (1, 2):
            raise ValueError("history stride must be 1 or 2")
        if not hasattr(policy, "sample_with_rngs"):
            raise ValueError("probe requires the learned categorical policy")
        self.policy = policy
        self.stride = stride
        self.histories = {}
        self.serial_rng = np.random.default_rng(0)

    def reset_seed(self, seed):
        self.histories.clear()
        self.serial_rng = np.random.default_rng(seed)

    def sample_with_rngs(self, observations, rngs):
        observations = np.asarray(observations)
        if observations.shape != (len(rngs), 4, 16, 64) or observations.dtype != np.uint8:
            raise ValueError("provide uint8 screen stacks [games, 4, 16, 64]")
        if len(set(rngs)) != len(rngs):
            raise ValueError("each active game needs its own RNG object")
        if self.stride == 1:
            return self.policy.sample_with_rngs(observations, rngs)
        spaced = []
        history_length = 3*self.stride+1
        for observation, rng in zip(observations, rngs, strict=True):
            newest = observation[-1].copy()
            if rng not in self.histories:
                if not np.all(observation == newest):
                    raise ValueError("a new history must begin at the repeated boot frame")
                history = deque((newest.copy() for _ in range(history_length)),
                                maxlen=history_length)
                self.histories[rng] = history
            else:
                history = self.histories[rng]
                history.append(newest)
            spaced.append(np.stack([history[i] for i in range(0, history_length, self.stride)]))
        return self.policy.sample_with_rngs(np.stack(spaced), rngs)

    def __call__(self, observations):
        if len(observations) != 1:
            raise ValueError("batched evaluation must supply per-game RNGs")
        return self.sample_with_rngs(observations, [self.serial_rng])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stride", type=int, choices=(1, 2), required=True)
    parser.add_argument("--tstates", type=int, choices=(50_000, 100_000), default=50_000)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.games <= 20 or not 10_000 <= args.seed <= 10_020-args.games:
        parser.error("this diagnostic is restricted to reused primary seeds 10000-10019")
    if args.envs < 1:
        parser.error("--envs must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    config = checkpoint_config(args.checkpoint)
    if config.get("algorithm") != "ppo":
        parser.error("this probe requires a categorical PPO checkpoint")
    source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy = TemporalPolicy(load_policy(args.checkpoint), stride=args.stride)
    # Equal nominal emulated-time guards; score-change HUD settling is extra.
    result = evaluate(policy, range(args.seed, args.seed+args.games),
                      tstates=args.tstates, max_steps=10_000_000_000//args.tstates,
                      envs=args.envs, verbose=True)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("checkpoint changed during evaluation")
    result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=source_sha,
                  policy=policy_description(config), deterministic_override=False,
                  evaluation_only=True, promotion_eligible=False,
                  observation_stride=args.stride,
                  nominal_history_span_tstates=3*args.stride*args.tstates,
                  checkpoint_tstates=config.get("tstates", 100_000))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "games"}), flush=True)
    if result["incomplete_games"]:
        raise SystemExit("Diagnostic incomplete; the entire suite is disqualified")


if __name__ == "__main__":
    main()
