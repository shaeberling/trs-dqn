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
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/training-curve"))
    parser.add_argument("--baseline", type=Path, default=Path("results/random.json"))
    args = parser.parse_args()
    fig, (scores, completion) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    fig.suptitle("Breakdown: learning from screen and score", fontsize=16)
    colors = ["#087e8b", "#bb4b00", "#7353ba", "#427a2e", "#ba3164"]
    records = []
    clear_label_used = False
    for run, color in zip(args.runs, colors*20):
        rows = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
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
                    color=color, alpha=0.8, linewidth=1.6, label=f"{run.name}: training (last 100 games)")
        scores.scatter([(r["steps"]+offset)/1000 for r in valid], [r["mean_score"] for r in valid],
                       color=color, marker="o", s=32, edgecolor="white", linewidth=0.5,
                       label=f"{run.name}: complete validation suite", zorder=3)
        clears = [r for r in valid if r.get("level_1_clears", 0)]
        if clears:
            scores.scatter([(r["steps"]+offset)/1000 for r in clears], [r["mean_score"] for r in clears],
                           color="#e1ad01", marker="*", s=180, edgecolor="#473700", linewidth=.8,
                           label=None if clear_label_used else "Verified level-1 clear", zorder=4)
            clear_label_used = True
        completion.plot([(r["steps"]+offset)/1000 for r in evaluations],
                        [r["complete_games"]/r["games_requested"] for r in evaluations],
                        color=color, marker="o", markersize=4, linewidth=1)
        records.append({"run": str(run), "prior_training_actions": offset, "validation": evaluations,
                        "latest_progress": progress[-1] if progress else None})
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())["mean_score"]
        scores.axhline(baseline, color="#687482", linestyle=":", linewidth=1,
                       label=f"Random baseline: {baseline:g}")
    scores.set_ylabel("Mean score")
    scores.set_ylim(bottom=0)
    scores.legend(fontsize=8, loc="upper left", ncol=2)
    scores.grid(alpha=0.2)
    completion.set_ylabel("Validation\ncompletion")
    completion.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
    completion.set_ylim(-0.1, 1.1)
    completion.set_xlabel("Environment actions along checkpoint lineage (thousands, including initialization)")
    completion.grid(alpha=0.2)
    for ax in (scores, completion):
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    args.output.with_suffix(".json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
