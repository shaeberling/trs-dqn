"""Fail-closed checks for the balanced-fire continuation watcher."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_continuation_compare as compare
from rl.defense_balanced_followup import MATCHED_CONFIG


def write_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


class ContinuationComparisonTests(unittest.TestCase):
    def make_run(self, root):
        run = root/"run"
        parent = root/"parent"
        parent.mkdir()
        (parent/"model.safetensors").write_bytes(b"frozen parent")
        write_json(parent/"state.json", {"steps": compare.START,
                                          "config": dict(MATCHED_CONFIG)})
        config = {**MATCHED_CONFIG, "run": str(run), "resume": str(parent),
                  "steps": compare.TARGET}
        write_json(run/"resume-config.json", config)
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
                "games": [{"seed": seed, "score": score} for seed in compare.FIXED_SEEDS],
                "mean_score": score, "highest_stage": stage, "mission_games": 0}

    def context(self, run, parent):
        return (patch.object(compare, "RUN", run),
                patch.object(compare, "PARENT", parent),
                patch.object(compare, "PARENT_SHA256",
                             compare.digest(parent/"model.safetensors")))

    def test_exact_stop_and_config_are_required(self):
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
                self.make_checkpoint(run/"latest", config, compare.TARGET-4096)
                self.assertEqual(compare.run_condition(), "stopped_early")
                self.make_checkpoint(run/"latest", config, compare.TARGET)
                config["entropy"] = .02
                write_json(run/"resume-config.json", config)
                self.assertEqual(compare.run_condition(), "incompatible")

    def test_stage_selection_and_complete_fixed_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, parent, config = self.make_run(Path(temporary))
            for index, steps in enumerate(range(compare.START + compare.INTERVAL,
                                                compare.TARGET + 1, compare.INTERVAL)):
                checkpoint = run/f"step-{steps:012d}"
                self.make_checkpoint(checkpoint, config, steps)
                write_json(checkpoint/"evaluation.json",
                           self.evaluation(1000 if index == 0 else 500,
                                           stage=2 if index in (3, 4) else 1))
            patches = self.context(run, parent)
            with patches[0], patches[1], patches[2]:
                selected = compare.select_checkpoint()["selected"]
                self.assertEqual(selected["steps"], compare.START + 4*compare.INTERVAL)
                bad = run/f"step-{compare.START + compare.INTERVAL:012d}"/"evaluation.json"
                row = json.loads(bad.read_text())
                row["games"][0]["seed"] = 999
                write_json(bad, row)
                with self.assertRaisesRegex(RuntimeError, "invalid fixed evaluation"):
                    compare.select_checkpoint()

    def test_fresh_requires_every_seed_and_native_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, replay = root/"fresh.json", root/"replay"
            replay.mkdir()
            (replay/"replay.html").write_text("replay")
            row = {"complete_games": compare.FRESH_GAMES, "incomplete_games": 0,
                   "games": [{"seed": compare.FRESH_SEED + i, "score": 10}
                             for i in range(compare.FRESH_GAMES)],
                   "checkpoint_sha256": "hash"}
            write_json(output, row)
            write_json(replay/"verification.json",
                       {"verified": True, "checkpoint_sha256": "hash"})
            self.assertEqual(compare.checked_fresh(output, replay, "hash"), row)
            row["games"][-1]["seed"] = 0
            write_json(output, row)
            with self.assertRaisesRegex(RuntimeError, "fresh evaluation"):
                compare.checked_fresh(output, replay, "hash")

    def test_paired_result_uses_clear_arm_names(self):
        def arm(scores):
            return {"games": [{"score": score} for score in scores],
                    "mean_score": sum(scores)/3, "median_score": 2,
                    "best_score": max(scores), "highest_stage": 1, "mission_games": 0}
        summary = compare.paired_result(arm([1, 3, 2]), arm([2, 2, 2]))
        self.assertEqual((summary["paired_continuation_wins"],
                          summary["paired_parent_wins"], summary["paired_ties"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
