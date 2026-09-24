"""Fail-closed checkpoint selection and fresh comparison for experiment 160."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.defense_balanced_compare import (checked_fresh, control_condition, digest,
                                         paired_summary, select_checkpoint)
from rl.defense_balanced_followup import MATCHED_CONFIG, TARGET


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class BalancedComparisonTests(unittest.TestCase):
    def make_run(self, root, arm, *, event="stopped", steps=TARGET, changes=None):
        run = root/arm
        config = {**MATCHED_CONFIG, "balanced_canonical_init": arm == "treatment",
                  "run": str(run), **(changes or {})}
        write_json(run/"config.json", config)
        write_json(run/"status.json", {"event": event, "pid": 1234})
        self.make_checkpoint(run/"latest", config, steps)
        return run

    @staticmethod
    def make_checkpoint(checkpoint, config, steps):
        checkpoint.mkdir(parents=True, exist_ok=True)
        write_json(checkpoint/"state.json", {"steps": steps, "config": config})
        (checkpoint/"model.safetensors").write_bytes(f"model-{steps}".encode())
        (checkpoint/"optimizer.npz").write_bytes(b"optimizer")

    @staticmethod
    def evaluation(score, *, stage=1, mission=0, sha=None):
        row = {"complete_games": 10, "incomplete_games": 0,
               "games": [{"seed": seed, "score": score} for seed in range(10000, 10010)],
               "mean_score": score, "highest_stage": stage, "mission_games": mission}
        if sha is not None:
            row["checkpoint_sha256"] = sha
        return row

    def make_candidates(self, root, arm, scores, *, stage_at=None):
        run = self.make_run(root, arm)
        config = json.loads((run/"config.json").read_text())
        count = 7 if arm == "treatment" else 8
        for index in range(count):
            steps = (index+1)*1_048_576
            checkpoint = run/f"step-{steps:012d}"
            self.make_checkpoint(checkpoint, config, steps)
            stage = 2 if stage_at == index else 1
            write_json(checkpoint/"evaluation.json", self.evaluation(scores[index], stage=stage))
        if arm == "treatment":
            write_json(root/"treatment-terminal-evaluation.json",
                       self.evaluation(scores[7], sha=digest(run/"latest/model.safetensors")))

    def test_control_condition_requires_exact_live_command_or_complete_matching_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root, "control", event="progress")
            with patch("rl.defense_balanced_compare.process_command",
                       return_value=f"python -m rl.defense_train --run {root/'control'}"):
                self.assertEqual(control_condition(root), "live")
            with patch("rl.defense_balanced_compare.process_command", return_value=""):
                self.assertEqual(control_condition(root), "uncertain")
                write_json(root/"control/status.json", {"event": "stopped", "pid": 1234})
                self.assertEqual(control_condition(root), "complete")
                state_path = root/"control/latest/state.json"
                state = json.loads(state_path.read_text())
                state["steps"] = TARGET-4096
                write_json(state_path, state)
                self.assertEqual(control_condition(root), "stopped_early")
                state["steps"] = TARGET
                state["config"]["entropy"] = .02
                write_json(state_path, state)
                self.assertEqual(control_condition(root), "incompatible")

    def test_selection_prefers_stage_then_mean_then_earliest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_candidates(root, "treatment", [100, 200, 300, 400, 500, 600, 700, 800])
            self.assertEqual(select_checkpoint(root, "treatment")["selected"]["steps"], TARGET)
            self.make_candidates(root, "control", [900, 900, 300, 400, 500, 600, 700, 800],
                                 stage_at=4)
            self.assertEqual(select_checkpoint(root, "control")["selected"]["steps"], 5*1_048_576)
            control = root/"control"
            fourth = control/f"step-{4*1_048_576:012d}"/"evaluation.json"
            fifth = control/f"step-{5*1_048_576:012d}"/"evaluation.json"
            write_json(fourth, self.evaluation(500, stage=2))
            write_json(fifth, self.evaluation(500, stage=2))
            self.assertEqual(select_checkpoint(root, "control")["selected"]["steps"], 4*1_048_576)

    def test_selection_rejects_incomplete_or_mismatched_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_candidates(root, "treatment", [100]*8)
            terminal = root/"treatment-terminal-evaluation.json"
            write_json(terminal, self.evaluation(100, sha="wrong"))
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                select_checkpoint(root, "treatment")
            write_json(terminal, self.evaluation(100, sha=digest(root/"treatment/latest/model.safetensors")))
            checkpoint = root/"treatment"/f"step-{1_048_576:012d}"
            state = json.loads((checkpoint/"state.json").read_text())
            state["config"]["balanced_canonical_init"] = False
            write_json(checkpoint/"state.json", state)
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                select_checkpoint(root, "treatment")

    def test_fresh_results_require_all_seeds_and_matching_verified_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root/"fresh.json"
            replay = root/"replay"
            replay.mkdir()
            (replay/"replay.html").write_text("verified replay")
            row = {"complete_games": 64, "incomplete_games": 0,
                   "games": [{"seed": 611000+i, "score": i} for i in range(64)],
                   "checkpoint_sha256": "abc", "mean_score": 31.5,
                   "median_score": 31.5, "best_score": 63, "highest_stage": 1,
                   "mission_games": 0}
            write_json(evaluation, row)
            write_json(replay/"verification.json", {"verified": True,
                                                    "checkpoint_sha256": "abc"})
            self.assertEqual(checked_fresh(evaluation, replay, "abc", 611000), row)
            row["games"][0]["seed"] = 999
            write_json(evaluation, row)
            with self.assertRaisesRegex(RuntimeError, "fresh evaluation"):
                checked_fresh(evaluation, replay, "abc", 611000)

    def test_paired_summary_is_seedwise(self):
        def row(scores):
            return {"games": [{"score": score} for score in scores],
                    "mean_score": sum(scores)/len(scores), "median_score": 2,
                    "best_score": max(scores), "highest_stage": 1, "mission_games": 0}
        result = paired_summary(row([1, 3, 2]), row([2, 2, 2]))
        self.assertEqual((result["paired_treatment_wins"], result["paired_control_wins"],
                          result["paired_ties"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
