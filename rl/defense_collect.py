"""Collect verified Defense improvements from isolated concurrent learners.

One collector owns the shared best archive. It never changes a learner, uses
evaluation-only sampling probes, or publishes without frozen-policy replay.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .defense_learning import game_rank, publish_best, sha256, write_json


def collect(sources, output, log=None):
    candidates = []
    for source in sources:
        best = Path(source)/"best"
        if not best.exists():
            continue  # A live learner may not have its first validation yet.
        bundle = best.resolve(strict=True)  # Pin an immutable version, not the changing link.
        manifest = json.loads((bundle/"manifest.json").read_text())
        for name in ("model.safetensors", "state.json", "evaluation.json", "verification.json"):
            if sha256(bundle/name) != manifest["hashes"][name]:
                raise ValueError(f"Source bundle checksum mismatch: {bundle/name}")
        evaluation = json.loads((bundle/"evaluation.json").read_text())
        verification = json.loads((bundle/"verification.json").read_text())
        if (not verification.get("verified") or verification.get("temperature", 1) != 1
                or verification["checkpoint_sha256"] != manifest["hashes"]["model.safetensors"]
                or evaluation.get("evaluation_only") or evaluation["incomplete_games"]):
            raise ValueError(f"Source is not a complete original-policy validation: {bundle}")
        rank = game_rank(manifest["result"])
        if rank is None or tuple(manifest["rank"]) != rank:
            raise ValueError(f"Invalid source rank: {bundle}")
        candidates.append((rank, str(bundle), evaluation))
    if not candidates:
        return None
    _, bundle, evaluation = max(candidates, key=lambda row: (row[0], row[1]))
    # publish_best independently checks the current global rank, reloads the
    # model, reproduces its complete game twice, and only then switches best.
    result = publish_best(Path(bundle)/"model.safetensors", evaluation, output, log=log)
    if result is not None and log:
        log(dict(event="collected", source=bundle, version=result["version"], rank=result["rank"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="isolated learner artifact root; repeat for multiple learners")
    parser.add_argument("--output", type=Path, default=Path("results/defense/learned"))
    parser.add_argument("--run", type=Path, default=Path("runs/defense-collector"))
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    if any(source.resolve() == args.output.resolve() for source in args.source):
        parser.error("source and destination archives must be separate")
    args.output.mkdir(parents=True, exist_ok=True)
    args.run.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    def log(row):
        row = dict(updated_unix=time.time(), pid=os.getpid(), **row)
        with (args.run/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        write_json(args.run/"status.json", row)
        print(json.dumps(row), flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # Prevent duplicate collector processes for the same destination. Other
    # learners must keep their own artifact roots; this is the global writer.
    with (args.output/".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another collector owns this destination")
        log(dict(event="start", sources=[str(p) for p in args.source], output=str(args.output)))
        while not stop:
            result = collect(args.source, args.output, log)
            log(dict(event="checked", promoted=result is not None))
            if args.once:
                break
            deadline = time.monotonic()+args.interval
            while not stop and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline-time.monotonic())))
        log(dict(event="stopped"))


if __name__ == "__main__":
    main()
