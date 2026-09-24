"""Stop one exact Defense learner gracefully before disk space becomes unsafe.

This is a process/storage watchdog, not part of learning or action selection.
It only signals the PID recorded by an already-running training directory,
after verifying the current process command still names that exact run.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def expected_training_command(command, run):
    words = command.split()
    return ("rl.defense_train" in words and "--run" in words
            and words.index("--run") + 1 < len(words)
            and words[words.index("--run") + 1] == str(run))


def process_command(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=5.1)
    parser.add_argument("--interval", type=float, default=30.)
    args = parser.parse_args()
    if (not args.run.is_dir() or not 5 <= args.min_free_gib <= 20
            or not 1 <= args.interval <= 60):
        parser.error("existing run, 5..20 GiB floor and 1..60 s interval required")
    status_path = args.run/"status.json"
    status = json.loads(status_path.read_text())
    pid = status.get("pid")
    if (not isinstance(pid, int) or pid <= 0
            or not expected_training_command(process_command(pid), args.run)):
        parser.error("status does not identify a live exact Defense trainer")
    with (args.run/"disk-watch.jsonl").open("a", buffering=1) as stream:
        while True:
            status = json.loads(status_path.read_text())
            command = process_command(pid)
            if status.get("pid") != pid or not expected_training_command(command, args.run):
                row = dict(event="trainer_stopped", pid=pid, checked_unix=time.time())
                stream.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
                return
            free_gib = shutil.disk_usage(args.run).free/1024**3
            if free_gib < args.min_free_gib:
                os.kill(pid, signal.SIGTERM)
                row = dict(event="disk_floor_signal", pid=pid, free_gib=free_gib,
                           min_free_gib=args.min_free_gib, checked_unix=time.time())
                stream.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
