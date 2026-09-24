"""One-shot, disk-guarded handoff from Defense experiment 160 to its control.

The treatment already runs independently. This monitor never changes its
actions, weights or rewards. After a normal target stop it evaluates the
terminal weights at the original fixed seeds (the resumed trainer's last
scheduled check lies 4,096 actions beyond its target), then starts the
predeclared matched control and its exact-run disk watchdog.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .defense_disk_watch import expected_training_command, process_command


ROOT = Path("runs/defense-ppo-balanced-fire-160")
TARGET = 8_388_608
MIN_LAUNCH_FREE_GIB = 6.0
POLL_SECONDS = 30
MATCHED_CONFIG = {
    "seed": 41, "envs": 16, "rollout": 256, "batch_size": 512,
    "epochs": 4, "learning_rate": 0.00025, "entropy": 0.01,
    "eval_every": 1_048_576, "eval_games": 10, "eval_envs": 10,
    "mlx_cache_mb": 512, "canonical_fire": True, "life_terminal": True,
    "balanced_canonical_init": True, "steps": TARGET,
    "gamma": 0.997, "gae_lambda": 0.95, "reward_scale": 0.01,
    "novelty_beta": 0.0, "alive_beta": 0.0,
    "policy_bias_noise": 0, "policy_weight_noise": 0,
    "curriculum_probability": 0, "tstates": 100_000,
    "observation_stride": 1, "allow_enter": False,
    "learned_durations": [], "max_episode_steps": 0,
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def treatment_condition(root=ROOT):
    """Return live, complete, stopped_early or uncertain from current state."""
    run = root/"treatment"
    status = read_json(run/"status.json")
    if not isinstance(status, dict):
        return "uncertain"
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and expected_training_command(process_command(pid), run):
        return "live"
    if status.get("event") != "stopped":
        return "uncertain"
    latest = read_json(run/"latest/state.json")
    original = read_json(run/"config.json")
    if not isinstance(latest, dict) or not isinstance(original, dict):
        return "uncertain"
    if (any(original.get(key) != value for key, value in MATCHED_CONFIG.items())
            or any(latest.get("config", {}).get(key) != value
                   for key, value in MATCHED_CONFIG.items())):
        return "incompatible"
    if latest.get("steps") != TARGET or latest.get("config", {}).get("steps") != TARGET:
        return "stopped_early"
    if not (run/"latest/model.safetensors").is_file() or not (run/"latest/optimizer.npz").is_file():
        return "incompatible"
    return "complete"


def control_command(root=ROOT, python=sys.executable):
    return [python, "-u", "-m", "rl.defense_train",
            "--run", str(root/"control"),
            "--artifacts", str(root/"control-artifacts"),
            "--canonical-fire", "--no-balanced-canonical-init", "--life-terminal",
            "--steps", str(TARGET), "--seed", "41", "--envs", "16",
            "--rollout", "256", "--batch-size", "512", "--epochs", "4",
            "--learning-rate", ".00025", "--entropy", ".01",
            "--gamma", ".997", "--gae-lambda", ".95",
            "--reward-scale", ".01", "--value-coefficient", ".5",
            "--tstates", "100000", "--observation-stride", "1",
            "--max-episode-steps", "0",
            "--eval-every", "1048576", "--eval-games", "10",
            "--eval-envs", "10", "--eval-seed", "10000",
            "--eval-max-steps", "0", "--mlx-cache-mb", "512"]


def write_status(root, event, **details):
    row = dict(event=event, checked_unix=time.time(), **details)
    path = root/"followup-status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2)+"\n")
    temporary.replace(path)
    print(json.dumps(row), flush=True)


def free_gib(root):
    return shutil.disk_usage(root).free/1024**3


def verify_terminal_evaluation(root):
    evaluation = read_json(root/"treatment-terminal-evaluation.json")
    verification = read_json(root/"treatment-terminal-replay/verification.json")
    if not (isinstance(evaluation, dict) and evaluation.get("complete_games") == 10
            and evaluation.get("incomplete_games") == 0
            and isinstance(verification, dict) and verification.get("verified") is True
            and verification.get("checkpoint_sha256") == evaluation.get("checkpoint_sha256")):
        raise RuntimeError("terminal fixed evaluation or native replay verification failed")
    return evaluation


def run_terminal_evaluation(root, python):
    if (root/"treatment-terminal-evaluation.json").exists() or (root/"treatment-terminal-replay").exists():
        raise RuntimeError("refusing to overwrite an existing terminal treatment evaluation")
    command = [python, "-u", "-m", "rl.defense_evaluate",
               str(root/"treatment/latest/model.safetensors"),
               "--output", str(root/"treatment-terminal-evaluation.json"),
               "--replay-output", str(root/"treatment-terminal-replay"),
               "--games", "10", "--seed", "10000", "--envs", "10"]
    with (root/"followup-evaluation.log").open("x") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"terminal treatment evaluation exited {result.returncode}")
    return verify_terminal_evaluation(root)


def start_control(root, python):
    if (root/"control").exists() or (root/"control-artifacts").exists():
        raise RuntimeError("control already exists; refusing a duplicate learner")
    command = control_command(root, python)
    output = (root/"followup-control.log").open("x")
    try:
        trainer = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    finally:
        output.close()
    try:
        for _ in range(120):
            status = read_json(root/"control/status.json")
            if isinstance(status, dict) and status.get("pid") == trainer.pid:
                break
            if trainer.poll() is not None:
                raise RuntimeError(f"matched control exited during startup: {trainer.returncode}")
            time.sleep(1)
        else:
            raise RuntimeError("matched control did not publish a PID within 120 seconds")
        watch_command = [python, "-u", "-m", "rl.defense_disk_watch",
                         "--run", str(root/"control"), "--min-free-gib", "5.1", "--interval", "30"]
        watch_output = (root/"followup-control-disk-watch.log").open("x")
        try:
            watcher = subprocess.Popen(watch_command, stdout=watch_output, stderr=subprocess.STDOUT)
        finally:
            watch_output.close()
        time.sleep(1)
        if watcher.poll() is not None:
            raise RuntimeError(f"matched-control disk watchdog exited during startup: {watcher.returncode}")
    except BaseException:
        if trainer.poll() is None:
            trainer.terminate()
            trainer.wait(timeout=60)
        raise
    write_status(root, "control_started", trainer_pid=trainer.pid, watchdog_pid=watcher.pid,
                 free_gib=free_gib(root))
    return trainer, watcher


def main():
    root = ROOT
    if not root.is_dir():
        raise RuntimeError(f"missing experiment root: {root}")
    with (root/"followup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uncertain = 0
        polls = 0
        write_status(root, "monitoring_treatment", monitor_pid=os.getpid())
        while True:
            condition = treatment_condition(root)
            if condition == "complete":
                break
            if condition in ("incompatible", "stopped_early"):
                write_status(root, condition)
                return
            uncertain = uncertain + 1 if condition == "uncertain" else 0
            if uncertain >= 3:
                write_status(root, "unverified_treatment_stop")
                return
            polls += 1
            if polls % 10 == 0:
                status = read_json(root/"treatment/status.json") or {}
                write_status(root, "monitoring_treatment", monitor_pid=os.getpid(),
                             treatment_steps=status.get("training_steps"), condition=condition)
            time.sleep(POLL_SECONDS)
        write_status(root, "treatment_complete", steps=TARGET)
        while free_gib(root) < MIN_LAUNCH_FREE_GIB:
            write_status(root, "waiting_for_disk", free_gib=free_gib(root))
            time.sleep(POLL_SECONDS)
        evaluation = run_terminal_evaluation(root, sys.executable)
        write_status(root, "terminal_evaluation_complete", mean_score=evaluation["mean_score"],
                     highest_stage=evaluation["highest_stage"])
        while free_gib(root) < MIN_LAUNCH_FREE_GIB:
            write_status(root, "waiting_for_disk", free_gib=free_gib(root))
            time.sleep(POLL_SECONDS)
        trainer, watcher = start_control(root, sys.executable)
        returncode = trainer.wait()
        try:
            watcher.wait(timeout=45)
        except subprocess.TimeoutExpired:
            watcher.terminate()
            watcher.wait(timeout=10)
        state = read_json(root/"control/latest/state.json") or {}
        write_status(root, "control_stopped", returncode=returncode, steps=state.get("steps"),
                     target_reached=returncode == 0 and state.get("steps") == TARGET)


if __name__ == "__main__":
    main()
