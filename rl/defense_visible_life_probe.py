"""Evaluation-only timing of visible ship losses in frozen Defense policies.

Action counts are a survival proxy, NOT decoded obstacle-course progress.
This reads only the ordinary screen-derived life-loss events and never
updates, rewards, selects, or steers a policy.
"""

import argparse
from pathlib import Path

import numpy as np

from .defense import ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import evaluate, load_policy, sha256, write_json


def life_summary(evaluation):
    if evaluation["incomplete_games"] or not evaluation["games"]:
        raise ValueError("requires complete original-boot games")
    first, durations = [], [[] for _ in range(4)]
    for game in evaluation["games"]:
        losses = game.get("visible_life_losses")
        if (not game["game_over"] or game["lives"] != 0 or
                not isinstance(losses, list) or len(losses) != 4 or
                losses[-1]["action"] != game["steps"] or
                losses[-1]["displayed_score"] != game["score"] or
                [event["visible_lives"] for event in losses] != [3, 2, 1, 0]):
            raise ValueError("visible loss sequence is incomplete or inconsistent")
        previous = 0
        for index, event in enumerate(losses):
            action = event["action"]
            if not isinstance(action, int) or action <= previous:
                raise ValueError("visible loss actions must increase")
            durations[index].append(action-previous)
            previous = action
        first.append(losses[0]["action"])
    return dict(first_visible_loss_actions=dict(
                    mean=float(np.mean(first)), median=float(np.median(first)),
                    minimum=min(first), maximum=max(first)),
                mean_actions_between_visible_losses=[float(np.mean(group)) for group in durations],
                caveat="Counts include original animations; they are not course-row or collision timestamps.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=621000)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    if args.games < 1 or args.envs < 1 or args.output.exists():
        parser.error("games/envs must be positive and output must not exist")
    before = sha256(args.checkpoint)
    policy, config = load_policy(args.checkpoint)
    evaluation = evaluate(policy, range(args.seed, args.seed+args.games),
                          tstates=config["tstates"], max_steps=0, envs=args.envs,
                          allow_enter=config.get("allow_enter", False),
                          observation_stride=config.get("observation_stride", 1),
                          record_visible_life_losses=True)
    if sha256(args.checkpoint) != before:
        raise RuntimeError("frozen policy weights changed")
    summary = life_summary(evaluation)
    report = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=before, source_sha256=sha256(__file__),
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  fixed_action_tstates=config["tstates"],
                  observation_stride=config.get("observation_stride", 1),
                  evaluation_seed=args.seed, complete_games=args.games,
                  timing=summary, evaluation=evaluation,
                  diagnostic_only=True, hidden_ram_reads=0,
                  model_updates=0, reward_changes=0,
                  action_override=False, checkpoint_selection_eligible=False,
                  replay_promotion_eligible=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(dict(checkpoint_sha256=before, games=args.games,
               mean_score=evaluation["mean_score"], highest_stage=evaluation["highest_stage"],
               first_visible_loss_actions=summary["first_visible_loss_actions"]))


if __name__ == "__main__":
    main()
