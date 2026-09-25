"""Fail-closed post-run comparison for fresh grouped-duration trial 163.

Waits for the exact planned trainer, selects only from complete fixed checks,
and evaluates two untouched matched seed sets with native-verified replays.
It never changes training, promotes a model, or prunes any checkpoints.
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


RUN = Path("runs/defense-ppo-grouped-duration-163")
ARTIFACTS = Path("runs/defense-ppo-grouped-duration-163-artifacts")
SMOKE = Path("results/defense/training/ppo-grouped-duration-fresh-163/smoke/config.json")
PARENT = Path("results/defense/training/ppo-balanced-fire-continuation-161/milestone-000015728640")
PARENT_SHA256 = "51945f630ef3c0ad3b92033321c03f22c8b08e11d31efbe9615a57de85fadc28"
TARGET = 16_777_216
INTERVAL = 1_048_576
FRESH_SEEDS = (615000, 615200)
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


def expected_config():
    smoke = read_json(SMOKE)
    if not isinstance(smoke, dict) or smoke.get("steps") != 16_384:
        raise RuntimeError("archived smoke configuration is missing or invalid")
    return {**smoke, "run": str(RUN), "artifacts": str(ARTIFACTS),
            "steps": TARGET, "eval_every": INTERVAL}


def checked_config():
    config = read_json(RUN/"config.json")
    parent = read_json(PARENT/"state.json")
    if (config != expected_config() or not isinstance(parent, dict)
            or digest(PARENT/"model.safetensors") != PARENT_SHA256
            or parent.get("steps") != 15_728_640
            or parent.get("config", {}).get("game_sha256") != config.get("game_sha256")
            or parent.get("config", {}).get("environment_version") != config.get("environment_version")
            or parent.get("config", {}).get("canonical_fire") is not True):
        raise RuntimeError("production config or frozen comparison parent differs from trial plan")
    return config


def run_condition():
    """Return live, complete, uncertain, stopped_early, or incompatible."""
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
        raise RuntimeError("expected exactly sixteen fixed checkpoint directories")
    checks = [candidate(RUN/f"step-{step:012d}", step, config) for step in steps]
    selected = max(checks, key=lambda row: (row["mission_games"] > 0,
                                           row["highest_stage"], row["mean_score"], -row["steps"]))
    return dict(selected=selected, candidates=checks)


def paired_summary(grouped, parent):
    left = [game["score"] for game in grouped["games"]]
    right = [game["score"] for game in parent["games"]]
    return dict(grouped_mean=grouped["mean_score"], parent_mean=parent["mean_score"],
                grouped_median=grouped["median_score"], parent_median=parent["median_score"],
                grouped_best=grouped["best_score"], parent_best=parent["best_score"],
                grouped_highest_stage=grouped["highest_stage"],
                parent_highest_stage=parent["highest_stage"],
                grouped_mission_games=grouped["mission_games"],
                parent_mission_games=parent["mission_games"],
                paired_grouped_wins=sum(a > b for a, b in zip(left, right)),
                paired_parent_wins=sum(a < b for a, b in zip(left, right)),
                paired_ties=sum(a == b for a, b in zip(left, right)))


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
        models = (("grouped", Path(selected["checkpoint"])/"model.safetensors",
                   selected["checkpoint_sha256"]),
                  ("parent", PARENT/"model.safetensors", PARENT_SHA256))
        summaries = {}
        for seed in FRESH_SEEDS:
            evaluations = {}
            for arm, model, model_hash in models:
                while shutil.disk_usage(RUN).free / 1024**3 < MIN_FREE_GIB:
                    write_status("waiting_for_disk", arm=arm, seed=seed)
                    time.sleep(POLL_SECONDS)
                write_status("evaluating", arm=arm, seed=seed)
                evaluations[arm] = run_fresh(arm, model, model_hash, seed, folder)
            summaries[str(seed)] = paired_summary(evaluations["grouped"], evaluations["parent"])
        report = dict(selection=selection, fixed_stage_proof=proof, fresh_sets=summaries,
                      promotion_attempted=False,
                      promotion_decision="review native-verified stage and fresh evidence first")
        report_path = folder/"report.json"
        if report_path.exists():
            raise RuntimeError("refusing to overwrite an existing completed report")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        write_status("comparison_complete", fresh_sets=summaries)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN.is_dir():
            write_status("error", error=repr(error))
        raise
