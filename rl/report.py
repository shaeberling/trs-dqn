"""Plot recorded training metrics; incomplete validation suites get no mean point."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/matplotlib-cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="run directories or archived JSONL training logs")
    parser.add_argument("--output", type=Path, default=Path("results/training-curve"))
    parser.add_argument("--baseline", type=Path, default=Path("results/random.json"))
    parser.add_argument("--target-level", type=int, default=2,
                        help="add a complete-game level-reach panel for targets above level 2")
    parser.add_argument("--highlight-level", type=int,
                        help="displayed level whose verified reaches get stars; defaults to target-level")
    args = parser.parse_args()
    if args.target_level < 2:
        parser.error("target-level must be at least 2")
    if args.highlight_level is None:
        args.highlight_level = args.target_level
    if args.highlight_level < 2:
        parser.error("highlight-level must be at least 2")
    heights = [3, 1.2, 1] if args.target_level > 2 else [3, 1]
    fig, axes = plt.subplots(len(heights), 1, figsize=(10, 8 if len(heights) == 3 else 6.5),
                             sharex=True, gridspec_kw={"height_ratios": heights}, layout="constrained")
    scores, completion = axes[0], axes[-1]
    levels = axes[1] if len(heights) == 3 else None
    highest_level = args.target_level
    fig.suptitle("Breakdown: learning from screen and score", fontsize=16)
    colors = ["#087e8b", "#bb4b00", "#7353ba", "#427a2e", "#ba3164", "#2f59bd"]
    records = []
    clear_label_used = False
    for run, color in zip(args.runs, colors*20):
        archived = run.is_file()
        label = run.stem if archived else run.name
        lines = (run if archived else run/"metrics.jsonl").read_text().splitlines()
        rows = []
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise
                # A live writer may not have finished its last record yet.
        if archived:
            config = next(row["config"] for row in rows if row["event"] == "start")
        else:
            config_path = run/"config.json"
            if not config_path.exists():
                config_path = run/"resume-config.json"
            config = json.loads(config_path.read_text())
        offset = config.get("prior_training_actions", 0)
        if not offset and config.get("algorithm") == "ppo" and config.get("initialize"):
            parent_state = Path(config["initialize"]).parent/"state.json"
            offset = json.loads(parent_state.read_text())["steps"]
        progress = [r for r in rows if r["event"] == "progress" and r.get("mean_score_100") is not None]
        evaluations = [r for r in rows if r["event"] == "validation"]
        valid = [r for r in evaluations if r["incomplete_games"] == 0]
        scores.plot([(r["steps"]+offset)/1000 for r in progress], [r["mean_score_100"] for r in progress],
                    color=color, alpha=0.8, linewidth=1.6, label=f"{label}: training (last 100 games)")
        scores.scatter([(r["steps"]+offset)/1000 for r in valid], [r["mean_score"] for r in valid],
                       color=color, marker="o", s=32, edgecolor="white", linewidth=0.5,
                       label=f"{label}: complete validation suite", zorder=3)
        clears = [r for r in valid if
                  (r.get("level_1_clears", 0) if args.highlight_level == 2 else
                   r.get("level_reach_counts", {}).get(str(args.highlight_level), 0))]
        if clears:
            scores.scatter([(r["steps"]+offset)/1000 for r in clears], [r["mean_score"] for r in clears],
                           color="#e1ad01", marker="*", s=180, edgecolor="#473700", linewidth=.8,
                           label=None if clear_label_used else
                           ("Verified level-1 clear" if args.highlight_level == 2 else
                            f"Verified level-{args.highlight_level} reach"), zorder=4)
            clear_label_used = True
        completion.plot([(r["steps"]+offset)/1000 for r in evaluations],
                        [r["complete_games"]/r["games_requested"] for r in evaluations],
                        color=color, marker="o", markersize=4, linewidth=1)
        if levels is not None:
            # Do not use highest_level_100: it may include truncated games.
            complete_progress = [r for r in progress if r.get("complete_games_100", 0)
                                 and "level_reach_counts_100" in r]
            depths = [max([1]+[int(level) for level, count in r["level_reach_counts_100"].items()
                              if count]) for r in complete_progress]
            complete_validation = [r for r in valid if r.get("complete_games", 0)
                                   and "highest_complete_level" in r]
            validation_depths = [r["highest_complete_level"] for r in complete_validation]
            highest_level = max([highest_level]+depths+validation_depths)
            levels.step([(r["steps"]+offset)/1000 for r in complete_progress], depths,
                        where="post", color=color, alpha=.6, linewidth=1)
            levels.scatter([(r["steps"]+offset)/1000 for r in complete_validation],
                           validation_depths, color=color, s=28, edgecolor="white", linewidth=.5,
                           zorder=3)
        records.append({"run": str(run), "prior_training_actions": offset, "validation": evaluations,
                        "latest_progress": progress[-1] if progress else None})
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())["mean_score"]
        scores.axhline(baseline, color="#687482", linestyle=":", linewidth=1,
                       label=f"Random baseline: {baseline:g}")
    scores.set_ylabel("Mean score")
    scores.set_ylim(bottom=0)
    scores.legend(fontsize=8, loc="lower left" if levels is not None else "upper left", ncol=2,
                  **({"bbox_to_anchor": (0, 1.02)} if levels is not None else {}))
    scores.grid(alpha=0.2)
    completion.set_ylabel("Validation\ncompletion")
    completion.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
    completion.set_ylim(-0.1, 1.1)
    completion.set_xlabel("Environment actions along checkpoint lineage (thousands, including initialization)")
    completion.grid(alpha=0.2)
    if levels is not None:
        levels.axhline(args.target_level, color="#687482", linestyle=":", linewidth=1)
        levels.set_ylabel("Highest level\n(complete games)")
        levels.set_yticks(range(1, highest_level+1))
        levels.set_ylim(.7, highest_level+.3)
        levels.grid(alpha=.2)
        levels.set_title("Lines: last 100 training games · dots: frozen validation suites", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=160, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
