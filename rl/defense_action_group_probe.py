"""Read-only canonical fire-key probe preserving frozen policy probability.

At stage one commands 9..17 are alternate forward-fire key combinations.
The probe sums their categorical mass and sends it through canonical Space.
No weights, rewards, screen inputs or global best are modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_learning import evaluate, load_policy, sha256, write_json
from .evaluate import categorical_policy


COMMAND_MAP = np.array((*range(10), 18, 19), dtype=np.int32)


def group_fire_logits(logits):
    values = np.asarray(logits)
    if (values.ndim != 2 or values.shape[1] != len(action_names())
            or not np.isfinite(values).all()):
        raise ValueError("requires finite 20-command PPO logits")
    fire = np.logaddexp.reduce(values[:, 9:18], axis=1, keepdims=True)
    return np.concatenate((values[:, :9], fire, values[:, 18:20]), axis=1)


class CanonicalFirePolicy:
    def __init__(self, infer_logits):
        self.inner = categorical_policy(lambda obs: group_fire_logits(infer_logits(obs)))

    def sample_with_rngs(self, obs, rngs):
        return COMMAND_MAP[self.inner.sample_with_rngs(obs, rngs)]

    def __call__(self, obs):
        return COMMAND_MAP[self.inner(obs)]

    def reset_seed(self, seed):
        self.inner.reset_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-training-seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--envs", type=int, default=16)
    args = parser.parse_args()
    if (args.output.exists() or min(args.games, args.envs) < 1
            or args.first_training_seed < 70000):
        parser.error("fresh output and positive counts on training-only seeds required")

    import mlx.core as mx
    from .model import QNetwork

    mx.set_cache_limit(256*1024*1024)
    before = sha256(args.checkpoint)
    baseline, config = load_policy(args.checkpoint)
    if (config.get("algorithm", "ppo") != "ppo"
            or config.get("architecture") is not None
            or config.get("allow_enter", False)
            or config.get("action_names") != list(action_names())):
        parser.error("probe requires the ordinary 20-command feedforward Defense PPO policy")
    model = QNetwork(action_count=20)
    model.load_weights(str(args.checkpoint))
    mx.eval(model.state)
    predict = mx.compile(model.policy_value, inputs=model.state)
    grouped = CanonicalFirePolicy(lambda obs: np.array(predict(mx.array(obs))[0]))
    seeds = range(args.first_training_seed, args.first_training_seed+args.games)
    settings = dict(tstates=config["tstates"], max_steps=0, envs=args.envs,
                    allow_enter=False, observation_stride=config.get("observation_stride", 1))
    parent = evaluate(baseline, seeds, **settings)
    canonical = evaluate(grouped, seeds, **settings)
    if sha256(args.checkpoint) != before:
        raise RuntimeError("checkpoint changed during canonical-fire probe")
    differences = [{"seed": left["seed"], "baseline": left["score"],
                    "canonical": right["score"]}
                   for left, right in zip(parent["games"], canonical["games"], strict=True)
                   if left["score"] != right["score"]]
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=before,
                  source_sha256=sha256(Path(__file__)), game_sha256=config["game_sha256"],
                  first_training_seed=args.first_training_seed, games_per_policy=args.games,
                  action_profile="fixed 12 groups: 0..8, sum probability of 9..17 to Space, retain 18/19",
                  command_map=COMMAND_MAP.tolist(),
                  baseline=parent, canonical_fire=canonical,
                  mean_score_change=canonical["mean_score"]-parent["mean_score"],
                  differing_score_games=differences,
                  diagnostic_only=True, promotion_eligible=False,
                  parameter_updates=0, reward_changed=False, screen_inputs_changed=False,
                  limitation="Grouping stage-one keyboard commands is not a learned new policy; "
                             "later-stage equivalence is unverified.")
    write_json(args.output, report)
    print(json.dumps({"games_per_policy": args.games,
                      "baseline_mean": parent["mean_score"],
                      "canonical_mean": canonical["mean_score"],
                      "mean_gain": report["mean_score_change"],
                      "differing_score_games": len(differences),
                      "baseline_stage": parent["highest_stage"],
                      "canonical_stage": canonical["highest_stage"]}), flush=True)


if __name__ == "__main__":
    main()
