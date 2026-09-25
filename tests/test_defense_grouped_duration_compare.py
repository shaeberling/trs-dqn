"""Fail-closed production selection and matched-result gates for trial 163."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_grouped_duration_compare as compare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class GroupedDurationComparisonTests(unittest.TestCase):
    def make_run(self, root):
        run, parent = root/"run", root/"parent"
        parent.mkdir()
        (parent/"model.safetensors").write_bytes(b"parent")
        write_json(parent/"state.json", {"steps": 15_728_640,
                                          "config": {"game_sha256": "game",
                                                     "environment_version": "env",
                                                     "canonical_fire": True}})
        config = {"run": str(run), "steps": compare.TARGET,
                  "game_sha256": "game", "environment_version": "env"}
        write_json(run/"config.json", config)
        write_json(run/"status.json", {"event": "stopped", "pid": 1234})
        self.checkpoint(run/"latest", config, compare.TARGET)
        return run, parent, config

    @staticmethod
    def checkpoint(path, config, steps, *, score=None, stage=1):
        path.mkdir(parents=True, exist_ok=True)
        write_json(path/"state.json", {"steps": steps, "config": config})
        (path/"model.safetensors").write_bytes(f"model-{steps}".encode())
        (path/"optimizer.npz").write_bytes(b"optimizer")
        if score is not None:
            write_json(path/"evaluation.json", {
                "complete_games": 10, "incomplete_games": 0,
                "games": [{"seed": seed, "score": score} for seed in range(10000, 10010)],
                "mean_score": score, "highest_stage": stage, "mission_games": 0})

    def context(self, run, parent, config):
        return (patch.object(compare, "RUN", run), patch.object(compare, "PARENT", parent),
                patch.object(compare, "PARENT_SHA256", compare.digest(parent/"model.safetensors")),
                patch.object(compare, "expected_config", return_value=config))

    def test_exact_target_parent_and_configuration_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, parent, config = self.make_run(Path(temporary))
            patches = self.context(run, parent, config)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(compare, "process_command", return_value=""):
                self.assertEqual(compare.run_condition(), "complete")
                self.checkpoint(run/"latest", config, compare.TARGET - 4096)
                self.assertEqual(compare.run_condition(), "stopped_early")
                self.checkpoint(run/"latest", config, compare.TARGET)
                write_json(run/"config.json", {**config, "steps": compare.TARGET + 1})
                self.assertEqual(compare.run_condition(), "incompatible")
                write_json(run/"config.json", config)
                write_json(parent/"state.json", {"steps": 15_728_640,
                                                 "config": {"game_sha256": "wrong"}})
                self.assertEqual(compare.run_condition(), "incompatible")

    def test_selection_uses_all_sixteen_and_prefers_stage_then_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, parent, config = self.make_run(Path(temporary))
            for index, steps in enumerate(range(compare.INTERVAL, compare.TARGET + 1,
                                                compare.INTERVAL)):
                self.checkpoint(run/f"step-{steps:012d}", config, steps,
                                score=900 if index == 0 else 500,
                                stage=2 if index in (7, 8) else 1)
            patches = self.context(run, parent, config)
            with patches[0], patches[1], patches[2], patches[3]:
                self.assertEqual(compare.select_checkpoint()["selected"]["steps"],
                                 8*compare.INTERVAL)
                bad = run/f"step-{compare.INTERVAL:012d}"/"evaluation.json"
                row = json.loads(bad.read_text())
                row["games"][0]["seed"] = 999
                write_json(bad, row)
                with self.assertRaisesRegex(RuntimeError, "invalid fixed evaluation"):
                    compare.select_checkpoint()

    def test_paired_summary_keeps_arm_names_and_counts(self):
        def arm(scores):
            return {"games": [{"score": score} for score in scores],
                    "mean_score": sum(scores)/3, "median_score": 2,
                    "best_score": max(scores), "highest_stage": 1, "mission_games": 0}
        result = compare.paired_summary(arm([1, 3, 2]), arm([2, 2, 2]))
        self.assertEqual((result["paired_grouped_wins"], result["paired_parent_wins"],
                          result["paired_ties"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
