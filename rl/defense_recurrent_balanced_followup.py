"""Fail-closed unattended completion of the fresh recurrent-control pair.

The monitor does not affect either learner's actions, rewards or weights.
It starts the predeclared matched control only after a complete treatment,
then verifies fixed checkpoints, original-boot stage claims and two untouched
fresh comparison sets. It never promotes the protected best policy.
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
from .defense_balanced_followup import read_json
from .defense_continuation_compare import candidate, verify_fixed_stage
from .defense_disk_watch import expected_training_command, process_command


NAME = "defense-ppo-recurrent-balanced-169"
TREATMENT = Path("runs") / f"{NAME}-treatment-full"
CONTROL = Path("runs") / f"{NAME}-control-full"
MONITOR = Path("runs") / f"{NAME}-followup"
TARGET = 8_388_608
INTERVAL = 1_048_576
FRESH_SEEDS = (622000, 622200)
FRESH_GAMES = 64
MIN_FREE_GIB = 6.0
POLL_SECONDS = 30
EXPECTED = {
    "game": "defense", "algorithm": "ppo", "seed": 41, "steps": TARGET,
    "envs": 16, "rollout": 256, "batch_size": 512, "epochs": 4,
    "learning_rate": .00025, "entropy": .01, "value_coefficient": .5,
    "gamma": .997, "gae_lambda": .95, "reward_scale": .01,
    "tstates": 100_000, "observation_stride": 1,
    "eval_every": INTERVAL, "eval_games": 10, "eval_envs": 10,
    "eval_seed": 10_000, "mlx_cache_mb": 512,
    "canonical_fire": True, "balanced_canonical_init": True,
    "recurrent_hidden": 128, "recurrent_own_action": True,
    "sequence_length": 32, "life_terminal": True,
    "freeze_recurrent_base": False, "allow_enter": False,
    "repeat_previous_action": False, "learned_durations": [],
    "grouped_duration": False, "spatial_residual": False,
    "movement_only": False, "sil_updates": 0,
    "novelty_beta": 0., "alive_beta": 0.,
    "curriculum_probability": 0., "policy_bias_noise": 0.,
    "policy_weight_noise": 0., "policy_key_noise": 0.,
    "policy_duration_noise": 0., "max_episode_steps": 0,
    "eval_max_steps": 0,
}


def write_status(event, **details):
    row = dict(event=event, checked_unix=time.time(), monitor_pid=os.getpid(), **details)
    target = MONITOR / "status.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(row), flush=True)


def free_gib():
    return shutil.disk_usage(MONITOR).free / 1024**3


def checked_config(run, arm):
    config = read_json(run / "config.json")
    if (not isinstance(config, dict)
            or config.get("run") != str(run)
            or config.get("artifacts") != str(Path(str(run) + "-artifacts"))
            or config.get("resume") is not None
            or config.get("initialize_policy") is not None
            or config.get("initialize_encoder") is not None
            or config.get("memory_scale") != (1. if arm == "treatment" else 0.)
            or any(config.get(key) != value for key, value in EXPECTED.items())):
        raise RuntimeError(f"{arm} configuration differs from the frozen protocol")
    if arm == "control":
        treatment = read_json(TREATMENT / "config.json")
        if not isinstance(treatment, dict):
            raise RuntimeError("missing frozen treatment configuration")
        differences = {key for key in config.keys() | treatment.keys()
                       if config.get(key) != treatment.get(key)}
        if differences != {"run", "artifacts", "memory_scale"}:
            raise RuntimeError(f"matched configurations differ beyond memory scale: {sorted(differences)}")
    return config


def run_condition(run, arm):
    """Return live, complete, uncertain, stopped_early or incompatible."""
    status = read_json(run / "status.json")
    if not isinstance(status, dict):
        return "uncertain"
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and expected_training_command(process_command(pid), run):
        return "live"
    if status.get("event") != "stopped":
        return "uncertain"
    try:
        config = checked_config(run, arm)
    except RuntimeError:
        return "incompatible"
    final = read_json(run / "latest/state.json")
    if not isinstance(final, dict) or final.get("steps") != TARGET or status.get("stop_requested"):
        return "stopped_early"
    if (final.get("config") != config
            or not (run / "latest/model.safetensors").is_file()
            or not (run / "latest/optimizer.npz").is_file()):
        return "incompatible"
    return "complete"


def fixed_selection(run, arm):
    config = checked_config(run, arm)
    steps = tuple(range(INTERVAL, TARGET + 1, INTERVAL))
    names = sorted(path.name for path in run.glob("step-[0-9]*") if path.is_dir())
    if names != [f"step-{step:012d}" for step in steps]:
        raise RuntimeError(f"{arm} lacks exactly eight planned fixed checkpoints")
    rows = [candidate(run / f"step-{step:012d}", step, config) for step in steps]
    for row in rows:
        evaluation = read_json(Path(row["checkpoint"]) / "evaluation.json")
        if evaluation.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} fixed evaluation lacks an exact model hash")
    selected = max(rows, key=lambda row: (row["mission_games"], row["highest_stage"],
                                          row["mean_score"], -row["steps"]))
    return dict(selected=selected, candidates=rows)


def wait_for_run(run, arm):
    uncertain = 0
    polls = 0
    while True:
        condition = run_condition(run, arm)
        if condition == "complete":
            return
        if condition in ("incompatible", "stopped_early"):
            raise RuntimeError(f"{arm} {condition}; not continuing automatically")
        uncertain = uncertain + 1 if condition == "uncertain" else 0
        if uncertain >= 3:
            raise RuntimeError(f"{arm} status uncertain for three polls")
        polls += 1
        if polls == 1 or polls % 10 == 0:
            status = read_json(run / "status.json") or {}
            write_status(f"monitoring_{arm}", condition=condition,
                         training_steps=status.get("training_steps"), free_gib=free_gib())
        time.sleep(POLL_SECONDS)


def control_command(python=sys.executable):
    return [python, "-u", "-m", "rl.defense_train",
            "--run", str(CONTROL), "--artifacts", str(Path(str(CONTROL) + "-artifacts")),
            "--canonical-fire", "--balanced-canonical-init", "--recurrent-hidden", "128",
            "--recurrent-own-action", "--sequence-length", "32", "--memory-scale", "0",
            "--life-terminal", "--steps", str(TARGET), "--seed", "41", "--envs", "16",
            "--rollout", "256", "--batch-size", "512", "--epochs", "4",
            "--learning-rate", ".00025", "--entropy", ".01", "--gamma", ".997",
            "--gae-lambda", ".95", "--reward-scale", ".01", "--tstates", "100000",
            "--eval-every", str(INTERVAL), "--eval-games", "10", "--eval-envs", "10",
            "--mlx-cache-mb", "512"]


def start_control():
    if CONTROL.exists() or Path(str(CONTROL) + "-artifacts").exists():
        raise RuntimeError("control already exists; refusing duplicate learner")
    output = (MONITOR / "control-trainer.log").open("x")
    try:
        trainer = subprocess.Popen(control_command(), stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        output.close()
    try:
        for _ in range(120):
            status = read_json(CONTROL / "status.json")
            if isinstance(status, dict) and status.get("pid") == trainer.pid:
                checked_config(CONTROL, "control")
                break
            if trainer.poll() is not None:
                raise RuntimeError(f"control exited during startup: {trainer.returncode}")
            time.sleep(1)
        else:
            raise RuntimeError("control did not publish its PID within 120 seconds")
        command = [sys.executable, "-u", "-m", "rl.defense_disk_watch",
                   "--run", str(CONTROL), "--min-free-gib", "5.1", "--interval", "30"]
        output = (MONITOR / "control-disk-watch.log").open("x")
        try:
            watchdog = subprocess.Popen(command, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            output.close()
        time.sleep(1)
        if watchdog.poll() is not None:
            raise RuntimeError(f"control disk watchdog exited during startup: {watchdog.returncode}")
    except BaseException:
        if trainer.poll() is None:
            trainer.terminate()
            trainer.wait(timeout=60)
        raise
    write_status("control_started", trainer_pid=trainer.pid, watchdog_pid=watchdog.pid,
                 free_gib=free_gib())


def checked_fresh(output, replay, model_hash, seed):
    evaluation = read_json(output)
    verification = read_json(replay / "verification.json")
    manifest = read_json(replay / "manifest.json")
    if (not isinstance(evaluation, dict) or not isinstance(verification, dict)
            or not isinstance(manifest, dict)
            or evaluation.get("complete_games") != FRESH_GAMES
            or evaluation.get("incomplete_games") != 0
            or [game.get("seed") for game in evaluation.get("games", [])]
            != list(range(seed, seed + FRESH_GAMES))
            or evaluation.get("checkpoint_sha256") != model_hash
            or verification.get("verified") is not True
            or verification.get("checkpoint_sha256") != model_hash
            or not (replay / "replay.html").is_file()):
        raise RuntimeError(f"fresh evaluation or replay verification failed: {output}")
    best = manifest.get("result", {})
    if (evaluation.get("highest_stage", 1) > 1 or evaluation.get("mission_games", 0) > 0):
        if (best.get("highest_stage") != evaluation.get("highest_stage")
                or (evaluation.get("mission_games", 0) > 0
                    and best.get("missions_completed", 0) < 1)):
            raise RuntimeError(f"fresh later-stage claim lacks its verified best replay: {output}")
    return evaluation


def run_fresh(arm, selected, seed):
    model = Path(selected["checkpoint"]) / "model.safetensors"
    model_hash = selected["checkpoint_sha256"]
    if digest(model) != model_hash:
        raise RuntimeError(f"selected {arm} model changed")
    output = MONITOR / f"{arm}-{seed}.json"
    replay = MONITOR / f"{arm}-{seed}-replay"
    if output.exists() or replay.exists():
        return checked_fresh(output, replay, model_hash, seed)
    command = [sys.executable, "-u", "-m", "rl.defense_evaluate", str(model),
               "--output", str(output), "--replay-output", str(replay),
               "--games", str(FRESH_GAMES), "--seed", str(seed), "--envs", "16"]
    with (MONITOR / f"{arm}-{seed}.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"fresh {arm}/{seed} evaluation exited {result.returncode}")
    return checked_fresh(output, replay, model_hash, seed)


def summary(treatment, control):
    left = [game["score"] for game in treatment["games"]]
    right = [game["score"] for game in control["games"]]
    return dict(treatment_mean=treatment["mean_score"], control_mean=control["mean_score"],
                treatment_median=treatment["median_score"], control_median=control["median_score"],
                treatment_best=treatment["best_score"], control_best=control["best_score"],
                treatment_below_9000=sum(score < 9000 for score in left),
                control_below_9000=sum(score < 9000 for score in right),
                treatment_highest_stage=treatment["highest_stage"],
                control_highest_stage=control["highest_stage"],
                treatment_mission_games=treatment["mission_games"],
                control_mission_games=control["mission_games"],
                paired_treatment_wins=sum(a > b for a, b in zip(left, right)),
                paired_control_wins=sum(a < b for a, b in zip(left, right)),
                paired_ties=sum(a == b for a, b in zip(left, right)))


def main():
    MONITOR.mkdir(parents=True, exist_ok=True)
    with (MONITOR / "followup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_for_run(TREATMENT, "treatment")
        treatment = fixed_selection(TREATMENT, "treatment")
        proof = verify_fixed_stage(treatment["selected"], MONITOR)
        write_status("treatment_complete", selected=treatment["selected"], stage_proof=proof)
        while free_gib() < MIN_FREE_GIB:
            write_status("waiting_for_disk_before_control", free_gib=free_gib())
            time.sleep(POLL_SECONDS)
        if not CONTROL.exists():
            start_control()
        wait_for_run(CONTROL, "control")
        control = fixed_selection(CONTROL, "control")
        control_proof_dir = MONITOR / "control-stage-proof"
        control_proof_dir.mkdir(exist_ok=True)
        control_proof = verify_fixed_stage(control["selected"], control_proof_dir)
        write_status("control_complete", selected=control["selected"], stage_proof=control_proof)
        selections = dict(treatment=treatment, control=control)
        selection_path = MONITOR / "selection.json"
        if selection_path.exists():
            if read_json(selection_path) != selections:
                raise RuntimeError("existing selection differs")
        else:
            selection_path.write_text(json.dumps(selections, indent=2) + "\n")
        sets = {}
        for seed in FRESH_SEEDS:
            evaluations = {}
            for arm, selection in (("treatment", treatment), ("control", control)):
                while free_gib() < MIN_FREE_GIB:
                    write_status("waiting_for_disk_before_fresh", arm=arm, seed=seed,
                                 free_gib=free_gib())
                    time.sleep(POLL_SECONDS)
                write_status("evaluating", arm=arm, seed=seed)
                evaluations[arm] = run_fresh(arm, selection["selected"], seed)
            sets[str(seed)] = summary(evaluations["treatment"], evaluations["control"])
        report = dict(selections=selections, treatment_stage_proof=proof,
                      control_stage_proof=control_proof, fresh_sets=sets,
                      promotion_attempted=False,
                      disposition="review verified later-stage evidence; otherwise archive negative score-only comparison")
        report_path = MONITOR / "report.json"
        if report_path.exists():
            raise RuntimeError("refusing to overwrite an existing completed report")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        write_status("comparison_complete", fresh_sets=sets)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if MONITOR.is_dir():
            write_status("error", error=repr(error))
        raise
