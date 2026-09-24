"""Unattended, fail-closed fresh comparison for Defense experiment 160.

Wait for the exact matched control to finish, then evaluate the frozen
validation-selected models on two predeclared untouched seed sets. This
never alters training, the protected best, or the promotion decision.
"""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .defense_balanced_followup import (MATCHED_CONFIG, MIN_LAUNCH_FREE_GIB,
                                        POLL_SECONDS, ROOT, TARGET, free_gib, read_json,
                                        treatment_condition, verify_terminal_evaluation)
from .defense_disk_watch import expected_training_command, process_command


FRESH_SETS = (611000, 611200)
FRESH_GAMES = 64
FIXED_SEEDS = list(range(10000, 10010))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_status(root, event, **details):
    row = dict(event=event, checked_unix=time.time(), **details)
    path = root/"comparison-status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2)+"\n")
    temporary.replace(path)
    print(json.dumps(row), flush=True)


def control_condition(root=ROOT):
    """Return live, complete, stopped_early, incompatible or uncertain."""
    run = root/"control"
    status = read_json(run/"status.json")
    if not isinstance(status, dict):
        return "uncertain"
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and expected_training_command(process_command(pid), run):
        return "live"
    if status.get("event") != "stopped":
        return "uncertain"
    original = read_json(run/"config.json")
    latest = read_json(run/"latest/state.json")
    if not isinstance(original, dict) or not isinstance(latest, dict):
        return "uncertain"
    expected = {**MATCHED_CONFIG, "balanced_canonical_init": False}
    if (any(original.get(key) != value for key, value in expected.items())
            or any(latest.get("config", {}).get(key) != value for key, value in expected.items())):
        return "incompatible"
    if latest.get("steps") != TARGET:
        return "stopped_early"
    if not (run/"latest/model.safetensors").is_file() or not (run/"latest/optimizer.npz").is_file():
        return "incompatible"
    return "complete"


def checked_candidate(checkpoint, evaluation_path, *, arm, expected_steps=None):
    state = read_json(checkpoint/"state.json")
    evaluation = read_json(evaluation_path)
    model = checkpoint/"model.safetensors"
    if not isinstance(state, dict) or not isinstance(evaluation, dict) or not model.is_file():
        raise RuntimeError(f"incomplete fixed candidate: {checkpoint}")
    steps = state.get("steps")
    if not isinstance(steps, int) or (expected_steps is not None and steps != expected_steps):
        raise RuntimeError(f"checkpoint step mismatch: {checkpoint}")
    expected = {**MATCHED_CONFIG, "balanced_canonical_init": arm == "treatment"}
    config = state.get("config", {})
    if (any(config.get(key) != value for key, value in expected.items())
            or config.get("run") != str(checkpoint.parent)):
        raise RuntimeError(f"checkpoint configuration mismatch: {checkpoint}")
    if (evaluation.get("complete_games") != 10 or evaluation.get("incomplete_games") != 0
            or [game.get("seed") for game in evaluation.get("games", [])] != FIXED_SEEDS):
        raise RuntimeError(f"invalid fixed complete-game evaluation: {evaluation_path}")
    model_hash = digest(model)
    if evaluation.get("checkpoint_sha256") not in (None, model_hash):
        raise RuntimeError(f"evaluation/checkpoint hash mismatch: {checkpoint}")
    if not (checkpoint/"optimizer.npz").is_file():
        raise RuntimeError(f"missing full optimizer state: {checkpoint}")
    if (not isinstance(evaluation.get("mean_score"), (int, float))
            or not math.isfinite(evaluation["mean_score"])
            or not isinstance(evaluation.get("highest_stage"), int)
            or not isinstance(evaluation.get("mission_games"), int)):
        raise RuntimeError(f"invalid checkpoint rank: {evaluation_path}")
    return dict(checkpoint=str(checkpoint), evaluation=str(evaluation_path), steps=steps,
                checkpoint_sha256=model_hash, mean_score=evaluation["mean_score"],
                highest_stage=evaluation["highest_stage"],
                mission_games=evaluation["mission_games"])


def select_checkpoint(root, arm):
    run = root/arm
    candidates = []
    for checkpoint in sorted(run.glob("step-[0-9]*")):
        if not checkpoint.is_dir():
            continue
        steps = int(checkpoint.name.removeprefix("step-"))
        candidates.append(checked_candidate(checkpoint, checkpoint/"evaluation.json",
                                            arm=arm, expected_steps=steps))
    if arm == "treatment":
        candidates.append(checked_candidate(run/"latest", root/"treatment-terminal-evaluation.json",
                                            arm=arm, expected_steps=TARGET))
    if len(candidates) != 8:
        raise RuntimeError(f"{arm} has {len(candidates)} complete checks, expected 8")
    if any(candidate["steps"] > TARGET for candidate in candidates):
        raise RuntimeError(f"{arm} checkpoint exceeds target")
    selected = max(candidates, key=lambda c: (c["mission_games"] > 0,
                                             c["highest_stage"], c["mean_score"], -c["steps"]))
    return dict(selected=selected, candidates=candidates)


def checked_fresh(evaluation_path, replay, model_hash, seed):
    evaluation = read_json(evaluation_path)
    verification = read_json(replay/"verification.json")
    if not (isinstance(evaluation, dict) and isinstance(verification, dict)
            and evaluation.get("complete_games") == FRESH_GAMES
            and evaluation.get("incomplete_games") == 0
            and [game.get("seed") for game in evaluation.get("games", [])]
            == list(range(seed, seed + FRESH_GAMES))
            and evaluation.get("checkpoint_sha256") == model_hash
            and verification.get("verified") is True
            and verification.get("checkpoint_sha256") == model_hash
            and (replay/"replay.html").is_file()):
        raise RuntimeError(f"fresh evaluation or native replay verification failed: {evaluation_path}")
    return evaluation


def run_fresh(root, arm, selected, seed, python):
    folder = root/"fresh-comparison"
    checkpoint = Path(selected["checkpoint"])/"model.safetensors"
    if digest(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError(f"selected {arm} weights changed")
    output = folder/f"{arm}-{seed}.json"
    replay = folder/f"{arm}-{seed}-replay"
    if output.exists() or replay.exists():
        return checked_fresh(output, replay, selected["checkpoint_sha256"], seed)
    command = [python, "-u", "-m", "rl.defense_evaluate", str(checkpoint),
               "--output", str(output), "--replay-output", str(replay),
               "--games", str(FRESH_GAMES), "--seed", str(seed), "--envs", "16"]
    with (folder/f"{arm}-{seed}.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"fresh {arm}/{seed} evaluation exited {result.returncode}")
    return checked_fresh(output, replay, selected["checkpoint_sha256"], seed)


def paired_summary(treatment, control):
    left = [game["score"] for game in treatment["games"]]
    right = [game["score"] for game in control["games"]]
    return dict(treatment_mean=treatment["mean_score"], control_mean=control["mean_score"],
                treatment_median=treatment["median_score"], control_median=control["median_score"],
                treatment_best=treatment["best_score"], control_best=control["best_score"],
                treatment_highest_stage=treatment["highest_stage"],
                control_highest_stage=control["highest_stage"],
                treatment_mission_games=treatment["mission_games"],
                control_mission_games=control["mission_games"],
                paired_treatment_wins=sum(a > b for a, b in zip(left, right)),
                paired_control_wins=sum(a < b for a, b in zip(left, right)),
                paired_ties=sum(a == b for a, b in zip(left, right)))


def main():
    root = ROOT
    if not root.is_dir():
        raise RuntimeError(f"missing experiment root: {root}")
    with (root/"comparison.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uncertain = 0
        polls = 0
        write_status(root, "monitoring_control", monitor_pid=os.getpid())
        while True:
            condition = control_condition(root)
            if condition == "complete":
                break
            if condition in ("incompatible", "stopped_early"):
                write_status(root, condition)
                return
            uncertain = uncertain + 1 if condition == "uncertain" else 0
            if uncertain >= 3:
                write_status(root, "unverified_control_stop")
                return
            polls += 1
            if polls % 10 == 0:
                status = read_json(root/"control/status.json") or {}
                write_status(root, "monitoring_control", monitor_pid=os.getpid(),
                             control_steps=status.get("training_steps"), condition=condition)
            time.sleep(POLL_SECONDS)
        if treatment_condition(root) != "complete":
            raise RuntimeError("treatment terminal state no longer verified complete")
        verify_terminal_evaluation(root)
        selections = {arm: select_checkpoint(root, arm) for arm in ("treatment", "control")}
        folder = root/"fresh-comparison"
        folder.mkdir(exist_ok=True)
        selection_file = folder/"selection.json"
        if selection_file.exists():
            if read_json(selection_file) != selections:
                raise RuntimeError("existing checkpoint selection differs")
        else:
            selection_file.write_text(json.dumps(selections, indent=2)+"\n")
        write_status(root, "control_complete", selected={k: v["selected"]["steps"]
                                                      for k, v in selections.items()})
        summaries = {}
        for seed in FRESH_SETS:
            evaluations = {}
            for arm in ("treatment", "control"):
                while free_gib(root) < MIN_LAUNCH_FREE_GIB:
                    write_status(root, "waiting_for_disk", free_gib=free_gib(root))
                    time.sleep(POLL_SECONDS)
                write_status(root, "evaluating", arm=arm, seed=seed)
                evaluations[arm] = run_fresh(root, arm, selections[arm]["selected"], seed,
                                             sys.executable)
            summaries[str(seed)] = paired_summary(evaluations["treatment"], evaluations["control"])
        report = dict(selections=selections, fresh_sets=summaries,
                      promotion_attempted=False, promotion_decision="review verified evidence first")
        report_path = folder/"report.json"
        if report_path.exists():
            raise RuntimeError("refusing to overwrite an existing completed comparison")
        report_path.write_text(json.dumps(report, indent=2)+"\n")
        write_status(root, "comparison_complete", fresh_sets=summaries)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ROOT.is_dir():
            write_status(ROOT, "error", error=repr(error))
        raise
