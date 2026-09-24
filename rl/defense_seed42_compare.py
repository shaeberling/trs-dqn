"""Unattended, fail-closed comparison for fresh balanced PPO seed 42.

The watcher never changes training or promotes a model. It requires an exact
target stop, sixteen complete fixed checks, and verified original-boot
replays before reporting either of two untouched matched fresh seed sets.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .defense_balanced_compare import digest
from .defense_balanced_followup import MATCHED_CONFIG, read_json
from .defense_continuation_compare import candidate, verify_fixed_stage
from .defense_disk_watch import expected_training_command, process_command


RUN = Path("runs/defense-ppo-balanced-seed42-162")
PARENT = Path("results/defense/training/ppo-balanced-fire-continuation-161/milestone-000015728640")
PARENT_SHA256 = "51945f630ef3c0ad3b92033321c03f22c8b08e11d31efbe9615a57de85fadc28"
TARGET = 16_777_216
INTERVAL = 1_048_576
FRESH_SEEDS = (614000, 614200)
FRESH_GAMES = 128
POLL_SECONDS = 30
MIN_FREE_GIB = 6.0
EXPECTED = {**MATCHED_CONFIG, "seed": 42, "steps": TARGET}


def write_status(event, **details):
    row = dict(event=event, checked_unix=time.time(), **details)
    path = RUN/"comparison-status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(row), flush=True)


def checked_config():
    config = read_json(RUN/"config.json")
    parent = read_json(PARENT/"state.json")
    if not isinstance(config, dict) or not isinstance(parent, dict):
        raise RuntimeError("missing training config or parent checkpoint state")
    if (config.get("run") != str(RUN) or config.get("resume") is not None
            or any(config.get(key) != value for key, value in EXPECTED.items())):
        raise RuntimeError("fresh run differs from predeclared seed-42 configuration")
    if (digest(PARENT/"model.safetensors") != PARENT_SHA256
            or parent.get("steps") != 15_728_640
            or parent.get("config", {}).get("game_sha256") != config.get("game_sha256")
            or parent.get("config", {}).get("environment_version") != config.get("environment_version")
            or parent.get("config", {}).get("canonical_fire") is not True):
        raise RuntimeError("frozen comparison parent differs from predeclared source")
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


def checked_fresh(output, replay, model_hash, seed):
    evaluation = read_json(output)
    verification = read_json(replay/"verification.json")
    if (not isinstance(evaluation, dict) or not isinstance(verification, dict)
            or evaluation.get("complete_games") != FRESH_GAMES
            or evaluation.get("incomplete_games") != 0
            or [game.get("seed") for game in evaluation.get("games", [])]
            != list(range(seed, seed + FRESH_GAMES))
            or evaluation.get("checkpoint_sha256") != model_hash
            or verification.get("verified") is not True
            or verification.get("checkpoint_sha256") != model_hash
            or not (replay/"replay.html").is_file()):
        raise RuntimeError(f"fresh evaluation or native replay verification failed: {output}")
    return evaluation


def run_fresh(arm, model, model_hash, seed, folder):
    if digest(model) != model_hash:
        raise RuntimeError(f"{arm} model changed")
    output = folder/f"{arm}-{seed}.json"
    replay = folder/f"{arm}-{seed}-replay"
    if output.exists() or replay.exists():
        return checked_fresh(output, replay, model_hash, seed)
    command = [sys.executable, "-u", "-m", "rl.defense_evaluate", str(model),
               "--output", str(output), "--replay-output", str(replay),
               "--games", str(FRESH_GAMES), "--seed", str(seed), "--envs", "16"]
    with (folder/f"{arm}-{seed}.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"fresh {arm}/{seed} evaluation exited {result.returncode}")
    return checked_fresh(output, replay, model_hash, seed)


def paired_summary(seed42, parent):
    left = [game["score"] for game in seed42["games"]]
    right = [game["score"] for game in parent["games"]]
    return dict(seed42_mean=seed42["mean_score"], parent_mean=parent["mean_score"],
                seed42_median=seed42["median_score"], parent_median=parent["median_score"],
                seed42_best=seed42["best_score"], parent_best=parent["best_score"],
                seed42_highest_stage=seed42["highest_stage"],
                parent_highest_stage=parent["highest_stage"],
                seed42_mission_games=seed42["mission_games"],
                parent_mission_games=parent["mission_games"],
                paired_seed42_wins=sum(a > b for a, b in zip(left, right)),
                paired_parent_wins=sum(a < b for a, b in zip(left, right)),
                paired_ties=sum(a == b for a, b in zip(left, right)))


def main():
    if not RUN.is_dir():
        raise RuntimeError(f"missing run: {RUN}")
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
        models = (("seed42", Path(selected["checkpoint"])/"model.safetensors",
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
            summaries[str(seed)] = paired_summary(evaluations["seed42"], evaluations["parent"])
        report = dict(selection=selection, fixed_stage_proof=proof, fresh_sets=summaries,
                      promotion_attempted=False,
                      promotion_decision="review independently verified stage and fresh evidence first")
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
