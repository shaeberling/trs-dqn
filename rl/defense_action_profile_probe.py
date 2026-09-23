"""Diagnostic: fixed ten distinct stage-one commands on frozen PPO logits.

No weights are updated and this masked-action probe is not eligible to
replace a trained-policy replay. The command subset is fixed for every
screen, including later stages; it is not a route or state-dependent rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_learning import evaluate, load_policy, sha256, write_json
from .evaluate import categorical_policy


UNIQUE_STAGE_ONE_COMMANDS = tuple(range(10))


def distinct_stage_one_logits(logits):
    """Retain movement/no-op and Space; omit synonymous stage-one fire keys."""
    values = np.asarray(logits)
    if (values.ndim != 2 or values.shape[1] != len(action_names())
            or not np.isfinite(values).all()):
        raise ValueError("requires finite 20-command PPO logits")
    return values[:, UNIQUE_STAGE_ONE_COMMANDS]


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
    subset = categorical_policy(lambda obs: distinct_stage_one_logits(
        np.array(predict(mx.array(obs))[0])))
    seeds = range(args.first_training_seed, args.first_training_seed+args.games)
    settings = dict(tstates=config["tstates"], max_steps=0, envs=args.envs,
                    allow_enter=False, observation_stride=config.get("observation_stride", 1))
    parent = evaluate(baseline, seeds, **settings)
    masked = evaluate(subset, seeds, **settings)
    if sha256(args.checkpoint) != before:
        raise RuntimeError("checkpoint changed during action-profile probe")
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=before,
                  source_sha256=sha256(Path(__file__)), game_sha256=config["game_sha256"],
                  first_training_seed=args.first_training_seed, games_per_policy=args.games,
                  action_profile="fixed commands 0..9 on every screen; no stage-dependent switch",
                  retained_commands=list(action_names()[:10]),
                  masked_commands=list(action_names()[10:]),
                  baseline=parent, fixed_profile=masked,
                  mean_score_change=masked["mean_score"]-parent["mean_score"],
                  diagnostic_only=True, promotion_eligible=False,
                  parameter_updates=0, reward_changed=False, screen_inputs_changed=False,
                  limitation="Stage-one command equivalence need not hold in later stages; "
                             "this untrained mask is a diagnostic, not a learned policy result.")
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in
          ("games_per_policy", "mean_score_change", "diagnostic_only")}
          | {"baseline_mean": parent["mean_score"],
             "masked_mean": masked["mean_score"],
             "baseline_stage": parent["highest_stage"],
             "masked_stage": masked["highest_stage"]}), flush=True)


if __name__ == "__main__":
    main()
