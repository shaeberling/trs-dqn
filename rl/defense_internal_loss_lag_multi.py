"""Forensic-only native ship-loss timing across fixed verified Defense replays.

Nonvideo RAM is read only by this diagnostic. It never chooses an action,
changes a reward, trains a model, or promotes a replay.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_gate_timing import visible_ship_column, visible_wall_runs
from .defense_internal_loss_lag import (INTERNAL_SHIPS, first_decrement_threshold,
                                        private_ships, run_recorded_key)
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


SOURCES = (
    Path("results/defense/learned/best"),
    Path("results/defense/training/ppo-balanced-fire-continuation-161/fresh-comparison/continuation-612000-replay"),
    Path("results/defense/training/ppo-grouped-duration-fresh-163/fresh-comparison/grouped-615000-replay"),
    Path("results/defense/training/ppo-movement-only-164/extension/fresh-comparison/extension-620600-replay"),
)


def visible_geometry(frames, index, life_start):
    sampled = [max(life_start, index-offset) for offset in (32, 16, 8, 0)]
    return [dict(frame=frame, ship_column=visible_ship_column(frames[frame]),
                 wall_runs_row_12=visible_wall_runs(frames[frame], 12),
                 wall_runs_row_13=visible_wall_runs(frames[frame], 13))
            for frame in sampled]


def probe_bundle(bundle):
    # Resolve the shared-best symlink before recording provenance so later
    # best-policy promotions cannot make this report point at different bytes.
    bundle = Path(bundle).resolve(strict=True)
    report, frames, actions = analyze(bundle)
    with np.load(bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]
        metadata = json.loads(str(trace["metadata"]))
    if (report["result"]["highest_stage"] != 1 or
            len(report["lives"]) != 4 or
            report["source_hashes"]["model.safetensors"] != sha256(bundle/"model.safetensors")):
        raise RuntimeError("requires a verified complete stage-one replay")
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    try:
        observation = env.reset(report["result"]["seed"])
        np.testing.assert_array_equal(observation[-1], frames[0])
        if private_ships(env) != 4:
            raise RuntimeError("private ship counter did not start at four")
        losses, pending, life_start = [], None, 0
        for index, action in enumerate(actions):
            before = private_ships(env)
            if pending is None:
                if before != env.lives:
                    raise RuntimeError("private and visible ship counts diverged before loss")
                saved = capture(env)
                base_after = run_recorded_key(env, action, env.tstates)
                restore(env, saved)
                if base_after < before:
                    if base_after != before-1:
                        raise RuntimeError("unexpected private ship-count change")
                    threshold = first_decrement_threshold(env, saved, action, before)
                    pending = dict(action_number=index+1,
                                   requested_tstates_threshold=threshold,
                                   during_base_action=True,
                                   private_ships_before=before,
                                   private_ships_after=base_after,
                                   visible_score_before=env.score,
                                   visible_geometry=visible_geometry(frames, index, life_start))
            else:
                saved, base_after = None, None
            observation, reward, terminated, truncated, info = env.step(int(action))
            if (reward != rewards[index] or
                    not np.array_equal(observation[-1], frames[index+1]) or
                    truncated or (terminated and index+1 != len(actions)) or
                    info["life_lost"] != any(index+1 == life["visible_loss_frame"]
                                            for life in report["lives"])):
                raise RuntimeError(f"original verified replay diverged at action {index+1}")
            after = private_ships(env)
            if pending is None and after < before:
                if saved is None or base_after != before or after != before-1:
                    raise RuntimeError("unexpected private count change during replay")
                pending = dict(action_number=index+1,
                               requested_tstates_threshold=None,
                               during_base_action=False,
                               private_ships_before=before,
                               private_ships_after=after,
                               visible_score_before=info["score"]-reward,
                               visible_geometry=visible_geometry(frames, index, life_start))
            if pending is not None and after != pending["private_ships_after"]:
                raise RuntimeError("private ship count changed twice before visible loss")
            if info["life_lost"]:
                if pending is None or after != info["lives"]:
                    raise RuntimeError("private and visible losses did not pair")
                life = report["lives"][len(losses)]
                if (index+1 != life["visible_loss_frame"] or
                        info["score"] != life["visible_score_at_loss"]):
                    raise RuntimeError("recorded visible loss differs from replay")
                losses.append(dict(life=len(losses)+1, private_decrement=pending,
                                   visible_loss_action=index+1,
                                   visible_loss_score=info["score"],
                                   reporting_lag_actions=index+1-pending["action_number"]))
                pending, life_start = None, index+1
        if len(losses) != 4 or pending is not None or not info["game_over"] or private_ships(env) != 0:
            raise RuntimeError("complete four-life game did not reconcile")
        return dict(bundle=str(bundle), seed=report["result"]["seed"],
                    model_sha256=report["source_hashes"]["model.safetensors"],
                    trace_sha256=report["source_hashes"]["trace.npz"],
                    verified_actions=len(actions), displayed_score=info["score"],
                    losses=losses,
                    lag_min=min(row["reporting_lag_actions"] for row in losses),
                    lag_max=max(row["reporting_lag_actions"] for row in losses))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a diagnostic report")
    bundles = [probe_bundle(bundle) for bundle in SOURCES]
    lags = [loss["reporting_lag_actions"] for bundle in bundles for loss in bundle["losses"]]
    result = dict(game_sha256=GAME_SHA256, probe_source_sha256=sha256(__file__),
                  audited_private_counter_address=f"0x{INTERNAL_SHIPS:04X}",
                  fixed_source_bundles=[str(path) for path in SOURCES],
                  bundles=bundles, total_verified_actions=sum(row["verified_actions"] for row in bundles),
                  loss_events=len(lags), lag_min=min(lags), lag_max=max(lags),
                  lags=lags, diagnostic_only=True, model_updates=0,
                  training_data_written=False, hidden_counter_used_by_policy_or_reward=False,
                  promotion_eligible=False,
                  limitations=["Selected verified replays, not a random sample of policies or seeds.",
                               "The private decrement follows original overlap detection; it is not exact collision time.",
                               "Visible ship glyphs may be occluded and wall codes are rendered geometry only.",
                               "No private-counter value may enter training, evaluation selection or acting."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(loss_events=len(lags), lag_min=min(lags), lag_max=max(lags),
                          lags=lags)))


if __name__ == "__main__":
    main()
