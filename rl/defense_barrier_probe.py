"""Diagnostic-only held-key counterfactuals from a verified own-policy state.

This does not train, select, or steer a model. Counterfactual action sequences
are never added to an experience buffer or replay promoted as learned play.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, action_names
from .defense_learning import sha256
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


def hold_until_boundary(env, action, maximum):
    start_score, start_lives = env.score, env.lives
    for index in range(maximum):
        _, _, terminated, truncated, info = env.step(action)
        if (info["life_lost"] or info["highest_stage"] >= 2
                or info["mission_completed"] or terminated or truncated):
            return dict(actions=index+1, score=info["score"],
                        score_gain=info["score"]-start_score,
                        lives=start_lives, lives_after=info["lives"],
                        highest_stage=info["highest_stage"],
                        life_lost=info["life_lost"],
                        mission_completed=info["mission_completed"],
                        terminated=terminated, truncated=truncated)
    return dict(actions=maximum, score=env.score,
                score_gain=env.score-start_score, lives=start_lives,
                lives_after=env.lives, highest_stage=env.highest_stage,
                life_lost=False, mission_completed=False,
                terminated=False, truncated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--lead", type=int, default=128,
                        help="own-policy decisions before first visible life loss")
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic")
    if args.lead < 1 or args.max_actions < 1:
        parser.error("lead and max-actions must be positive")

    report, frames, actions = analyze(args.bundle)
    metadata = json.loads((args.bundle/"manifest.json").read_text())["metadata"]
    first_loss = report["lives"][0]["visible_loss_frame"]
    anchor = first_loss-args.lead
    if anchor < 0:
        parser.error("lead exceeds the first observed life")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]

    env = DefenseEnv(tstates=report["tstates"], max_steps=metadata["max_steps"])
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        for index in range(anchor):
            obs, reward, done, truncated, _ = env.step(int(actions[index]))
            np.testing.assert_array_equal(obs[-1], frames[index+1])
            if reward != rewards[index] or done or truncated:
                raise RuntimeError("verified prefix did not reproduce")
        saved = capture(env)

        restore(env, saved)
        baseline = None
        for index in range(anchor, first_loss):
            obs, reward, done, truncated, info = env.step(int(actions[index]))
            np.testing.assert_array_equal(obs[-1], frames[index+1])
            if reward != rewards[index] or done or truncated:
                raise RuntimeError("verified original suffix did not reproduce")
            if info["life_lost"]:
                baseline = dict(actions=index-anchor+1, score=info["score"],
                                score_gain=info["score"]-saved.score,
                                lives=saved.lives, lives_after=info["lives"],
                                highest_stage=info["highest_stage"],
                                life_lost=True, mission_completed=info["mission_completed"])
        if baseline is None or anchor+baseline["actions"] != first_loss:
            raise RuntimeError("verified life boundary did not reproduce")

        alternatives = []
        for action, name in enumerate(action_names()):
            restore(env, saved)
            alternatives.append(dict(action=action, name=name,
                                     **hold_until_boundary(env, action, args.max_actions)))
    finally:
        env.close()

    output = dict(bundle=str(args.bundle),
                  replay_trace_sha256=sha256(args.bundle/"trace.npz"),
                  probe_source_sha256=sha256(Path(__file__)),
                  first_visible_loss_frame=first_loss,
                  anchor_frame=anchor, lead=args.lead,
                  anchor_score=saved.score,
                  anchor_lives=saved.lives,
                  verified_original_suffix=baseline,
                  constant_key_counterfactuals=alternatives,
                  diagnostic_only=True, training_data_written=False,
                  model_updates=0, promotion_eligible=False,
                  limitations=[
                      "One selected learned-policy state, not a representative game population.",
                      "Constant physical keys are diagnostic interventions, not a trained policy.",
                      "Visible life loss can lag the physical collision.",
                      "No counterfactual sequence is supplied to a learner or promoted as a replay."
                  ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(dict(anchor=anchor, first_loss=first_loss,
                          original=baseline,
                          alternatives=[dict(name=row["name"], actions=row["actions"],
                                             score=row["score"], stage=row["highest_stage"],
                                             life_lost=row["life_lost"])
                                        for row in alternatives]), indent=2))


if __name__ == "__main__":
    main()
