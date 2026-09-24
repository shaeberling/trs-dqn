"""Fail-closed post-run comparison for the balanced-fire continuation 161.

This waits for the exact trainer to stop at its planned target, selects from
the eight fixed complete-game checks, and evaluates frozen selected/parent
weights on untouched matched seeds. It never promotes or changes a model.
"""

import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .defense_balanced_compare import digest
from .defense_balanced_followup import MATCHED_CONFIG, read_json
from .defense_disk_watch import expected_training_command, process_command


RUN = Path("runs/defense-ppo-balanced-fire-continuation-161")
PARENT = Path("results/defense/training/ppo-balanced-fire-from-scratch-160/terminal-checkpoint")
PARENT_SHA256 = "c0f84420424b25ef129803e7f3144764ebc15bb9754e40704da9a7808d4bc241"
START = 8_388_608
TARGET = 16_777_216
INTERVAL = 1_048_576
FIXED_SEEDS = list(range(10000, 10010))
FRESH_SEED = 612000
FRESH_GAMES = 128
POLL_SECONDS = 30
MIN_FREE_GIB = 6.0
INHERITED_KEYS = tuple(key for key in MATCHED_CONFIG if key != "steps")


def write_status(event, **details):
    row = dict(event=event, checked_unix=time.time(), **details)
    target = RUN/"comparison-status.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(row), flush=True)


def checked_config():
    parent = read_json(PARENT/"state.json")
    config = read_json(RUN/"resume-config.json")
    if not isinstance(parent, dict) or not isinstance(config, dict):
        raise RuntimeError("missing parent state or continuation config")
    if parent.get("steps") != START or digest(PARENT/"model.safetensors") != PARENT_SHA256:
        raise RuntimeError("parent checkpoint differs from predeclared source")
    if (config.get("run") != str(RUN) or config.get("resume") != str(PARENT)
            or config.get("steps") != TARGET
            or any(config.get(key) != MATCHED_CONFIG[key] for key in INHERITED_KEYS)
            or any(config.get(key) != parent.get("config", {}).get(key)
                   for key in INHERITED_KEYS)):
        raise RuntimeError("continuation does not match predeclared source/configuration")
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
    state = read_json(RUN/"latest/state.json")
    if not isinstance(state, dict) or state.get("steps") != TARGET:
        return "stopped_early"
    if state.get("config") != config:
        return "incompatible"
    if not all((RUN/"latest"/name).is_file()
               for name in ("model.safetensors", "optimizer.npz")):
        return "incompatible"
    return "complete"


def candidate(checkpoint, expected_step, config):
    state = read_json(checkpoint/"state.json")
    evaluation = read_json(checkpoint/"evaluation.json")
    model = checkpoint/"model.safetensors"
    if (not isinstance(state, dict) or not isinstance(evaluation, dict)
            or state.get("steps") != expected_step or state.get("config") != config
            or not model.is_file() or not (checkpoint/"optimizer.npz").is_file()):
        raise RuntimeError(f"incomplete fixed checkpoint: {checkpoint}")
    model_hash = digest(model)
    if (evaluation.get("complete_games") != 10 or evaluation.get("incomplete_games") != 0
            or [game.get("seed") for game in evaluation.get("games", [])] != FIXED_SEEDS
            or evaluation.get("checkpoint_sha256") not in (None, model_hash)
            or not isinstance(evaluation.get("highest_stage"), int)
            or not isinstance(evaluation.get("mission_games"), int)
            or not isinstance(evaluation.get("mean_score"), (int, float))
            or not math.isfinite(evaluation["mean_score"])):
        raise RuntimeError(f"invalid fixed evaluation: {checkpoint}")
    return dict(checkpoint=str(checkpoint), steps=expected_step,
                checkpoint_sha256=model_hash,
                highest_stage=evaluation["highest_stage"],
                mission_games=evaluation["mission_games"],
                mean_score=evaluation["mean_score"])


def select_checkpoint():
    config = checked_config()
    expected = list(range(START + INTERVAL, TARGET + 1, INTERVAL))
    actual = sorted(path.name for path in RUN.glob("step-[0-9]*") if path.is_dir())
    if actual != [f"step-{step:012d}" for step in expected]:
        raise RuntimeError("expected exactly eight fixed checkpoint directories")
    checks = [candidate(RUN/f"step-{step:012d}", step, config) for step in expected]
    selected = max(checks, key=lambda c: (c["mission_games"] > 0,
                                         c["highest_stage"], c["mean_score"], -c["steps"]))
    return dict(selected=selected, candidates=checks)


def checked_fresh(output, replay, model_hash):
    evaluation = read_json(output)
    verification = read_json(replay/"verification.json")
    if (not isinstance(evaluation, dict) or not isinstance(verification, dict)
            or evaluation.get("complete_games") != FRESH_GAMES
            or evaluation.get("incomplete_games") != 0
            or [game.get("seed") for game in evaluation.get("games", [])]
            != list(range(FRESH_SEED, FRESH_SEED + FRESH_GAMES))
            or evaluation.get("checkpoint_sha256") != model_hash
            or verification.get("verified") is not True
            or verification.get("checkpoint_sha256") != model_hash
            or not (replay/"replay.html").is_file()):
        raise RuntimeError(f"fresh evaluation or native replay verification failed: {output}")
    return evaluation


def run_fresh(arm, model, model_hash, folder):
    if digest(model) != model_hash:
        raise RuntimeError(f"{arm} checkpoint changed")
    output = folder/f"{arm}-{FRESH_SEED}.json"
    replay = folder/f"{arm}-{FRESH_SEED}-replay"
    if output.exists() or replay.exists():
        return checked_fresh(output, replay, model_hash)
    command = [sys.executable, "-u", "-m", "rl.defense_evaluate", str(model),
               "--output", str(output), "--replay-output", str(replay),
               "--games", str(FRESH_GAMES), "--seed", str(FRESH_SEED), "--envs", "16"]
    with (folder/f"{arm}-{FRESH_SEED}.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"fresh {arm} evaluation exited {result.returncode}")
    return checked_fresh(output, replay, model_hash)


def verify_fixed_stage(selected, folder):
    """Re-run a claimed later-stage fixed game from original boot."""
    if selected["highest_stage"] <= 1 and selected["mission_games"] == 0:
        return None
    evaluation = read_json(Path(selected["checkpoint"])/"evaluation.json")
    game = max(evaluation["games"], key=lambda row: (row["mission_completed"],
                                                    row["highest_stage"], row["score"]))
    model = Path(selected["checkpoint"])/"model.safetensors"
    output = folder/"fixed-stage-proof.json"
    replay = folder/"fixed-stage-proof-replay"
    if not output.exists() and not replay.exists():
        command = [sys.executable, "-u", "-m", "rl.defense_evaluate", str(model),
                   "--output", str(output), "--replay-output", str(replay),
                   "--games", "1", "--seed", str(game["seed"]), "--envs", "1"]
        with (folder/"fixed-stage-proof.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"fixed stage proof exited {result.returncode}")
    repeated = read_json(output)
    verification = read_json(replay/"verification.json")
    if (not isinstance(repeated, dict) or not isinstance(verification, dict)
            or repeated.get("games") != [game]
            or repeated.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or verification.get("verified") is not True
            or verification.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or not (replay/"replay.html").is_file()):
        raise RuntimeError("fixed later-stage result failed original-boot replay verification")
    return dict(seed=game["seed"], stage=game["highest_stage"],
                mission_completed=game["mission_completed"], replay=str(replay/"replay.html"))


def paired_result(continuation, parent):
    left = [game["score"] for game in continuation["games"]]
    right = [game["score"] for game in parent["games"]]
    return dict(continuation_mean=continuation["mean_score"], parent_mean=parent["mean_score"],
                continuation_median=continuation["median_score"], parent_median=parent["median_score"],
                continuation_best=continuation["best_score"], parent_best=parent["best_score"],
                continuation_highest_stage=continuation["highest_stage"],
                parent_highest_stage=parent["highest_stage"],
                continuation_mission_games=continuation["mission_games"],
                parent_mission_games=parent["mission_games"],
                paired_continuation_wins=sum(a > b for a, b in zip(left, right)),
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
        target = folder/"selection.json"
        if target.exists():
            if read_json(target) != selection:
                raise RuntimeError("existing checkpoint selection differs")
        else:
            target.write_text(json.dumps(selection, indent=2) + "\n")
        write_status("training_complete", selected=selection["selected"])
        selected = selection["selected"]
        proof = verify_fixed_stage(selected, folder)
        models = (("continuation", Path(selected["checkpoint"])/"model.safetensors",
                   selected["checkpoint_sha256"]),
                  ("parent", PARENT/"model.safetensors", PARENT_SHA256))
        evaluations = {}
        for arm, model, model_hash in models:
            while shutil.disk_usage(RUN).free / 1024**3 < MIN_FREE_GIB:
                write_status("waiting_for_disk", arm=arm)
                time.sleep(POLL_SECONDS)
            write_status("evaluating", arm=arm)
            evaluations[arm] = run_fresh(arm, model, model_hash, folder)
        summary = paired_result(evaluations["continuation"], evaluations["parent"])
        report = dict(selection=selection, fresh_seed=FRESH_SEED,
                      fresh_games=FRESH_GAMES, fixed_stage_proof=proof, paired=summary,
                      promotion_attempted=False,
                      promotion_decision="review native-verified evidence first")
        report_path = folder/"report.json"
        if report_path.exists():
            raise RuntimeError("refusing to overwrite an existing completed report")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        write_status("comparison_complete", paired=summary)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN.is_dir():
            write_status("error", error=repr(error))
        raise
