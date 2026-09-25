"""Integrity of the completed, compact trial-169 comparison archive."""

import json
from pathlib import Path
import unittest

from rl.defense_learning import sha256
from rl.defense_recurrent_balanced_followup import (
    FRESH_GAMES, FRESH_SEEDS, INTERVAL, TARGET, checked_fixed_confirmation,
    checked_fresh, summary,
)


ROOT = Path("results/defense/training/ppo-recurrent-balanced-fresh-169")


def read(path):
    return json.loads(path.read_text())


class RecurrentBalancedArchiveTests(unittest.TestCase):
    def test_selected_states_and_all_fixed_rechecks(self):
        report = read(ROOT / "comparison/report.json")
        self.assertEqual(report["selections"], read(ROOT / "comparison/selection.json"))
        for arm in ("treatment", "control"):
            with self.subTest(arm=arm):
                run = ROOT / f"{arm}-run"
                selected = ROOT / f"{arm}-selected-terminal"
                state = read(selected / "state.json")
                self.assertEqual(state["steps"], TARGET)
                self.assertEqual(state["config"], read(run / "config.json"))
                self.assertEqual(read(run / "status.json")["event"], "stopped")
                self.assertFalse(read(run / "status.json")["stop_requested"])
                self.assertGreater((selected / "optimizer.npz").stat().st_size, 0)
                self.assertGreater((run / "metrics.jsonl").stat().st_size, 0)
                chosen = report["selections"][arm]["selected"]
                self.assertEqual(chosen["steps"], TARGET)
                self.assertEqual(sha256(selected / "model.safetensors"),
                                 chosen["checkpoint_sha256"])
                fixed = ROOT / f"{arm}-fixed"
                names = sorted(path.name for path in fixed.glob("*.json"))
                self.assertEqual(names, [f"step-{step:012d}.json"
                                         for step in range(INTERVAL, TARGET + 1, INTERVAL)])
                for candidate in report["selections"][arm]["candidates"]:
                    step = candidate["steps"]
                    original = read(fixed / f"step-{step:012d}.json")
                    repeated = read(ROOT / "comparison" / f"{arm}-fixed-{step}-recheck.json")
                    checked_fixed_confirmation(original, repeated,
                                               candidate["checkpoint_sha256"])
                    self.assertEqual(original["highest_stage"], 1)
                    self.assertEqual(original["mission_games"], 0)
                    self.assertEqual(original["mean_score"], candidate["mean_score"])

    def test_fresh_sets_and_complete_replay_bundles(self):
        report = read(ROOT / "comparison/report.json")
        comparison = ROOT / "comparison"
        self.assertEqual(set(report["fresh_sets"]), {str(seed) for seed in FRESH_SEEDS})
        for seed in FRESH_SEEDS:
            evaluations = {}
            for arm in ("treatment", "control"):
                with self.subTest(seed=seed, arm=arm):
                    selected = report["selections"][arm]["selected"]
                    replay = comparison / f"{arm}-{seed}-replay"
                    evaluations[arm] = checked_fresh(
                        comparison / f"{arm}-{seed}.json", replay,
                        selected["checkpoint_sha256"], seed)
                    self.assertEqual(evaluations[arm]["complete_games"], FRESH_GAMES)
                    self.assertEqual(evaluations[arm]["highest_stage"], 1)
                    self.assertEqual(evaluations[arm]["mission_games"], 0)
                    manifest = read(replay / "manifest.json")
                    for name, digest in manifest["hashes"].items():
                        self.assertEqual(sha256(replay / name), digest)
            self.assertEqual(summary(evaluations["treatment"], evaluations["control"]),
                             report["fresh_sets"][str(seed)])

    def test_best_effort_bundles_and_archived_course_depth(self):
        for arm, expected in (("treatment", 4), ("control", 6)):
            versions = ROOT / f"{arm}-best-efforts"
            bundles = sorted(path for path in versions.iterdir() if path.is_dir())
            self.assertEqual(len(bundles), expected)
            for bundle in bundles:
                with self.subTest(bundle=bundle):
                    manifest = read(bundle / "manifest.json")
                    for name, digest in manifest["hashes"].items():
                        self.assertEqual(sha256(bundle / name), digest)
                    self.assertTrue(read(bundle / "verification.json")["verified"])
                    self.assertTrue((bundle / "replay.html").is_file())
        for label, rows in (("seventh", [30, 29, 29, 29]),
                            ("eighth", [33, 34, 34, 34])):
            original = read(ROOT / "course-probes" / f"control-{label}.json")
            archived = read(ROOT / "course-probes" / f"archived-{label}/report.json")
            self.assertEqual(original["bundles"][0]["model_sha256"],
                             archived["bundles"][0]["model_sha256"])
            self.assertEqual(original["bundles"][0]["trace_sha256"],
                             archived["bundles"][0]["trace_sha256"])
            self.assertEqual([loss["decoded_rows"] for loss in archived["bundles"][0]["losses"]],
                             rows)


if __name__ == "__main__":
    unittest.main()
