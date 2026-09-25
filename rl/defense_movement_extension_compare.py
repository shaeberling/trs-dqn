"""Fail-closed post-run comparison for movement-only extension 164.

The watcher never changes training, promotes a model or prunes data.
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


RUN = Path("runs/defense-ppo-movement-only-164-extension")
ARTIFACTS = Path("runs/defense-ppo-movement-only-164-extension-artifacts")
PILOT = Path("results/defense/training/ppo-movement-only-164/pilot/checkpoint")
PILOT_CONFIG = Path("results/defense/training/ppo-movement-only-164/pilot/config.json")
PILOT_SHA256 = "664ded8ace39fafcc8b3e462bb8d5c93b8283207af1be67388685bd58167e34b"
PILOT_STEPS = 1_048_576
TARGET = 4_194_304
INTERVAL = 262_144
FRESH_SEEDS = (620_400, 620_600)
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
    pilot_config = read_json(PILOT_CONFIG)
    config = read_json(RUN/"resume-config.json")
    pilot = read_json(PILOT/"state.json")
    if (not isinstance(pilot_config, dict) or pilot_config.get("steps") != PILOT_STEPS
            or pilot_config.get("movement_only") is not True
            or not isinstance(config, dict)
            or config != {**pilot_config, "run": str(RUN), "artifacts": str(ARTIFACTS),
                          "resume": str(PILOT), "steps": TARGET}
            or not isinstance(pilot, dict) or pilot.get("steps") != PILOT_STEPS
            or pilot.get("config") != pilot_config
            or digest(PILOT/"model.safetensors") != PILOT_SHA256
            or pilot.get("config", {}).get("game_sha256") != config.get("game_sha256")
            or pilot.get("config", {}).get("environment_version") != config.get("environment_version")):
        raise RuntimeError("extension configuration or frozen pilot differs from trial plan")
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
    steps = list(range(PILOT_STEPS + INTERVAL, TARGET + 1, INTERVAL))
    names = sorted(path.name for path in RUN.glob("step-[0-9]*") if path.is_dir())
    if names != [f"step-{step:012d}" for step in steps]:
        raise RuntimeError("expected exactly twelve extension checkpoint directories")
    checks = [candidate(RUN/f"step-{step:012d}", step, config) for step in steps]
    selected = max(checks, key=lambda row: (row["mission_games"] > 0,
                                           row["highest_stage"], row["mean_score"], -row["steps"]))
    return dict(selected=selected, candidates=checks)


def paired_summary(extension, pilot):
    left = [game["score"] for game in extension["games"]]
    right = [game["score"] for game in pilot["games"]]
    extension_steps = sum(game["steps"] for game in extension["games"])/FRESH_GAMES
    pilot_steps = sum(game["steps"] for game in pilot["games"])/FRESH_GAMES
    return dict(extension_mean=extension["mean_score"], pilot_mean=pilot["mean_score"],
                extension_median=extension["median_score"], pilot_median=pilot["median_score"],
                extension_best=extension["best_score"], pilot_best=pilot["best_score"],
                extension_mean_steps=extension_steps, pilot_mean_steps=pilot_steps,
                extension_highest_stage=extension["highest_stage"],
                pilot_highest_stage=pilot["highest_stage"],
                extension_mission_games=extension["mission_games"],
                pilot_mission_games=pilot["mission_games"],
                paired_extension_wins=sum(a > b for a, b in zip(left, right)),
                paired_pilot_wins=sum(a < b for a, b in zip(left, right)),
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
        models = (("extension", Path(selected["checkpoint"])/"model.safetensors",
                   selected["checkpoint_sha256"]),
                  ("pilot", PILOT/"model.safetensors", PILOT_SHA256))
        summaries = {}
        for seed in FRESH_SEEDS:
            evaluations = {}
            for arm, model, model_hash in models:
                while shutil.disk_usage(RUN).free / 1024**3 < MIN_FREE_GIB:
                    write_status("waiting_for_disk", arm=arm, seed=seed)
                    time.sleep(POLL_SECONDS)
                write_status("evaluating", arm=arm, seed=seed)
                evaluations[arm] = run_fresh(arm, model, model_hash, seed, folder)
            summaries[str(seed)] = paired_summary(evaluations["extension"], evaluations["pilot"])
        report = dict(selection=selection, fixed_stage_proof=proof, fresh_sets=summaries,
                      promotion_attempted=False,
                      promotion_decision="review original-boot stage and fresh evidence first")
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
