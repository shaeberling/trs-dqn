"""Fail-closed post-run comparison for movement-only pilot 164.

It never changes training, promotes a model, resumes a learner or prunes data.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .defense_balanced_compare import digest
from .defense_balanced_followup import read_json
from .defense_continuation_compare import candidate, verify_fixed_stage
from .defense_disk_watch import expected_training_command, process_command
from .defense_seed42_compare import run_fresh


RUN = Path("runs/defense-ppo-movement-only-164-pilot")
ARTIFACTS = Path("runs/defense-ppo-movement-only-164-pilot-artifacts")
SMOKE = Path("results/defense/training/ppo-movement-only-164/smoke/config.json")
BASELINE = Path("results/defense/training/ppo-movement-only-164/baseline/checkpoint")
BASELINE_SHA256 = "e22114747b5cd18611a13ea052ab5476ec1b308738ea9d3208e21b5407e43bf1"
TARGET = 1_048_576
INTERVAL = 262_144
FRESH_SEED = 620_200
FRESH_GAMES = 128
POLL_SECONDS = 30
MIN_FREE_GIB = 6.0


def write_status(event, **details):
    row = dict(event=event, checked_unix=time.time(), **details)
    target = RUN/"comparison-status.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(row), flush=True)


def checked_config():
    smoke = read_json(SMOKE)
    config = read_json(RUN/"config.json")
    baseline = read_json(BASELINE/"state.json")
    if (not isinstance(smoke, dict) or smoke.get("steps") != 16_384
            or smoke.get("movement_only") is not True
            or not isinstance(config, dict)
            or config != {**smoke, "run": str(RUN), "artifacts": str(ARTIFACTS),
                          "steps": TARGET, "eval_every": INTERVAL}
            or not isinstance(baseline, dict)
            or digest(BASELINE/"model.safetensors") != BASELINE_SHA256
            or baseline.get("steps") != 0
            or baseline.get("config", {}).get("movement_only") is not True
            or baseline.get("config", {}).get("game_sha256") != config.get("game_sha256")
            or baseline.get("config", {}).get("environment_version") != config.get("environment_version")):
        raise RuntimeError("pilot configuration or frozen baseline differs from trial plan")
    return config


def run_condition():
    """Return live, complete, uncertain, stopped_early or incompatible."""
    status = read_json(RUN/"status.json")
    if not isinstance(status, dict):
        return "uncertain"
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and expected_training_command(process_command(pid), RUN):
        return "live"
    if status.get("event") != "stopped":
        return "uncertain"
    try:
        config = checked_config()
    except (OSError, RuntimeError):
        return "incompatible"
    final = read_json(RUN/"latest/state.json")
    if not isinstance(final, dict) or final.get("steps") != TARGET:
        return "stopped_early"
    if (final.get("config") != config
            or not (RUN/"latest/model.safetensors").is_file()
            or not (RUN/"latest/optimizer.npz").is_file()):
        return "incompatible"
    return "complete"


def select_checkpoint():
    config = checked_config()
    steps = list(range(INTERVAL, TARGET + 1, INTERVAL))
    names = sorted(path.name for path in RUN.glob("step-[0-9]*") if path.is_dir())
    if names != [f"step-{step:012d}" for step in steps]:
        raise RuntimeError("expected exactly four fixed checkpoint directories")
    checks = [candidate(RUN/f"step-{step:012d}", step, config) for step in steps]
    selected = max(checks, key=lambda row: (row["mission_games"] > 0,
                                           row["highest_stage"], row["mean_score"], -row["steps"]))
    return dict(selected=selected, candidates=checks)


def paired_summary(movement, baseline):
    left = [game["score"] for game in movement["games"]]
    right = [game["score"] for game in baseline["games"]]
    movement_steps = sum(game["steps"] for game in movement["games"])/FRESH_GAMES
    baseline_steps = sum(game["steps"] for game in baseline["games"])/FRESH_GAMES
    return dict(movement_mean=movement["mean_score"], baseline_mean=baseline["mean_score"],
                movement_median=movement["median_score"], baseline_median=baseline["median_score"],
                movement_best=movement["best_score"], baseline_best=baseline["best_score"],
                movement_mean_steps=movement_steps, baseline_mean_steps=baseline_steps,
                movement_highest_stage=movement["highest_stage"],
                baseline_highest_stage=baseline["highest_stage"],
                movement_mission_games=movement["mission_games"],
                baseline_mission_games=baseline["mission_games"],
                paired_movement_wins=sum(a > b for a, b in zip(left, right)),
                paired_baseline_wins=sum(a < b for a, b in zip(left, right)),
                paired_ties=sum(a == b for a, b in zip(left, right)),
                stage_one_extension_gate=(movement["highest_stage"] == 1
                                          and movement["mean_score"] >= baseline["mean_score"] + 20
                                          and movement_steps >= baseline_steps + 20))


def main():
    if not RUN.is_dir():
        raise RuntimeError(f"missing run: {RUN}")
    checked_config()
    with (RUN/"comparison.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uncertain = 0
        polls = 0
        while True:
            condition = run_condition()
            if condition == "complete":
                break
            if condition in ("incompatible", "stopped_early"):
                write_status(condition)
                return
            uncertain = uncertain + 1 if condition == "uncertain" else 0
            if uncertain >= 3:
                write_status("unverified_trainer_stop")
                return
            polls += 1
            if polls == 1 or polls % 10 == 0:
                status = read_json(RUN/"status.json") or {}
                write_status("monitoring", monitor_pid=os.getpid(),
                             training_steps=status.get("training_steps"), condition=condition)
            time.sleep(POLL_SECONDS)
        selection = select_checkpoint()
        folder = RUN/"fresh-comparison"
        folder.mkdir(exist_ok=True)
        selection_path = folder/"selection.json"
        if selection_path.exists():
            if read_json(selection_path) != selection:
                raise RuntimeError("existing checkpoint selection differs")
        else:
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        write_status("training_complete", selected=selection["selected"])
        proof = verify_fixed_stage(selection["selected"], folder)
        selected = selection["selected"]
        models = (("movement", Path(selected["checkpoint"])/"model.safetensors",
                   selected["checkpoint_sha256"]),
                  ("baseline", BASELINE/"model.safetensors", BASELINE_SHA256))
        evaluations = {}
        for arm, model, model_hash in models:
            while shutil.disk_usage(RUN).free / 1024**3 < MIN_FREE_GIB:
                write_status("waiting_for_disk", arm=arm, seed=FRESH_SEED)
                time.sleep(POLL_SECONDS)
            write_status("evaluating", arm=arm, seed=FRESH_SEED)
            evaluations[arm] = run_fresh(arm, model, model_hash, FRESH_SEED, folder)
        summary = paired_summary(evaluations["movement"], evaluations["baseline"])
        report = dict(selection=selection, fixed_stage_proof=proof, fresh_set=summary,
                      promotion_attempted=False,
                      promotion_decision="review original-boot stage and fresh-game evidence first")
        report_path = folder/"report.json"
        if report_path.exists():
            raise RuntimeError("refusing to overwrite an existing completed report")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        write_status("comparison_complete", fresh_set=summary)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN.is_dir():
            write_status("error", error=repr(error))
        raise
