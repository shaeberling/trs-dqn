"""Read a training log without importing MLX or disturbing the running learner."""

import argparse
import json
from pathlib import Path
import time


def read_status(directory):
    path = Path(directory)/"metrics.jsonl"
    events = []
    # The writer can be in the middle of its final line while we read.
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not events:
        return {"run": str(directory), "status": "no complete log records"}
    start = next((row for row in reversed(events) if row["event"] == "start"), {})
    progress = next((row for row in reversed(events) if row["event"] == "progress"), {})
    validation = [row for row in events if row["event"] == "validation"]
    errors = [row for row in events if row["event"] == "error"]
    last = events[-1]
    evaluation_progress = next((row for row in reversed(events)
                                if row["event"] == "evaluation_progress"), None)
    if (last["event"] not in ("validation_start", "evaluation_progress")
            or evaluation_progress is None or evaluation_progress.get("steps") != last.get("steps")):
        evaluation_progress = None
    return {
        "run": str(directory),
        "last_event": last["event"],
        "last_log_age_seconds": round(time.time()-path.stat().st_mtime, 1),
        "steps": last.get("steps"),
        "start_steps": start.get("steps"),
        "budget_end_steps": start.get("config", {}).get("steps"),
        "wall_seconds": last.get("wall_seconds"),
        "target_met": last.get("target_met", False),
        "highest_training_level": max((row.get("level", 1) for row in events
                                        if row["event"] == "episode" and row.get("terminated")
                                        and row.get("full_game", True)), default=1),
        "best_training_score": max((row.get("score", 0) for row in events
                                     if row["event"] == "episode" and row.get("terminated")
                                     and row.get("full_game", True)), default=0),
        "curriculum": {
            "archive_additions": sum(row["event"] == "curriculum_archive" for row in events),
            "segments": sum(row["event"] == "curriculum_episode" for row in events),
            "highest_segment_level": max((row.get("level", 1) for row in events
                                           if row["event"] == "curriculum_episode"), default=None),
        },
        "progress": {key: progress.get(key) for key in
                     ("steps_per_second", "mean_score_100", "level_reach_counts_100", "entropy", "approx_kl",
                      "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes")},
        "evaluation_progress": evaluation_progress,
        "recent_validation": [{key: row.get(key) for key in
                               ("steps", "mean_score", "best_score", "highest_complete_level",
                                "complete_games", "incomplete_games", "level_reach_counts")}
                              for row in validation[-6:]],
        "last_error": errors[-1] if errors else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_status(args.run), indent=2))
