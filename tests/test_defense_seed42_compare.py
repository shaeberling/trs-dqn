"""Fail-closed selection and paired-evaluation checks for fresh seed 42."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_seed42_compare as compare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Seed42ComparisonTests(unittest.TestCase):
    def make_run(self, root):
        run, parent = root/"run", root/"parent"
        parent.mkdir()
        (parent/"model.safetensors").write_bytes(b"parent")
        write_json(parent/"state.json", {"steps": 15_728_640,
                                          "config": {"game_sha256": "game",
                                                     "environment_version": "env",
                                                     "canonical_fire": True}})
        config = {**compare.EXPECTED, "run": str(run), "resume": None,
                  "game_sha256": "game", "environment_version": "env"}
        write_json(run/"config.json", config)
        write_json(run/"status.json", {"event": "stopped", "pid": 1234})
        self.make_checkpoint(run/"latest", config, compare.TARGET)
        return run, parent, config

    @staticmethod
    def make_checkpoint(path, config, steps):
        path.mkdir(parents=True, exist_ok=True)
        write_json(path/"state.json", {"steps": steps, "config": config})
        (path/"model.safetensors").write_bytes(f"model-{steps}".encode())
        (path/"optimizer.npz").write_bytes(b"optimizer")

    @staticmethod
    def evaluation(score, stage=1):
        return {"complete_games": 10, "incomplete_games": 0,
                "games": [{"seed": seed, "score": score} for seed in range(10000, 10010)],
                "mean_score": score, "highest_stage": stage, "mission_games": 0}

    def context(self, run, parent):
        return (patch.object(compare, "RUN", run), patch.object(compare, "PARENT", parent),
                patch.object(compare, "PARENT_SHA256",
                             compare.digest(parent/"model.safetensors")))

    def test_exact_target_and_configuration_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, parent, config = self.make_run(Path(temporary))
            patches = self.context(run, parent)
            with patches[0], patches[1], patches[2], patch.object(compare, "process_command", return_value=""):
                self.assertEqual(compare.run_condition(), "complete")
                write_json(run/"status.json", {"event": "progress", "pid": 1234})
                with patch.object(compare, "process_command",
                                  return_value=f"python -m rl.defense_train --run {run}"):
                    self.assertEqual(compare.run_condition(), "live")
                write_json(run/"status.json", {"event": "stopped", "pid": 1234})
                self.make_checkpoint(run/"latest", config, compare.TARGET - 4096)
                self.assertEqual(compare.run_condition(), "stopped_early")
                self.make_checkpoint(run/"latest", config, compare.TARGET)
                config["seed"] = 41
                write_json(run/"config.json", config)
                self.assertEqual(compare.run_condition(), "incompatible")

    def test_selection_prefers_stage_then_mean_and_checks_all_sixteen(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, parent, config = self.make_run(Path(temporary))
            for index, steps in enumerate(range(compare.INTERVAL, compare.TARGET + 1,
                                                compare.INTERVAL)):
                checkpoint = run/f"step-{steps:012d}"
                self.make_checkpoint(checkpoint, config, steps)
                write_json(checkpoint/"evaluation.json",
                           self.evaluation(900 if index == 0 else 500,
                                           stage=2 if index in (7, 8) else 1))
            patches = self.context(run, parent)
            with patches[0], patches[1], patches[2]:
                self.assertEqual(compare.select_checkpoint()["selected"]["steps"],
                                 8*compare.INTERVAL)
                bad = run/f"step-{compare.INTERVAL:012d}"/"evaluation.json"
                row = json.loads(bad.read_text())
                row["games"][0]["seed"] = 999
                write_json(bad, row)
                with self.assertRaisesRegex(RuntimeError, "invalid fixed evaluation"):
                    compare.select_checkpoint()

    def test_fresh_requires_all_seeds_and_matching_native_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, replay = root/"fresh.json", root/"replay"
            replay.mkdir()
            (replay/"replay.html").write_text("verified")
            row = {"complete_games": compare.FRESH_GAMES, "incomplete_games": 0,
                   "games": [{"seed": 614000+i, "score": 10} for i in range(compare.FRESH_GAMES)],
                   "checkpoint_sha256": "hash"}
            write_json(output, row)
            write_json(replay/"verification.json",
                       {"verified": True, "checkpoint_sha256": "hash"})
            self.assertEqual(compare.checked_fresh(output, replay, "hash", 614000), row)
            row["games"][0]["seed"] = 999
            write_json(output, row)
            with self.assertRaisesRegex(RuntimeError, "fresh evaluation"):
                compare.checked_fresh(output, replay, "hash", 614000)

    def test_paired_summary_has_seed42_and_parent_names(self):
        def arm(scores):
            return {"games": [{"score": score} for score in scores],
                    "mean_score": sum(scores)/3, "median_score": 2,
                    "best_score": max(scores), "highest_stage": 1, "mission_games": 0}
        result = compare.paired_summary(arm([1, 3, 2]), arm([2, 2, 2]))
        self.assertEqual((result["paired_seed42_wins"], result["paired_parent_wins"],
                          result["paired_ties"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
