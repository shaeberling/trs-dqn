"""Exact-target watcher for the score-only movement continuation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_movement_extension_compare as compare


class MovementExtensionCompareTests(unittest.TestCase):
    def test_selection_checks_all_twelve_and_prefers_stage_then_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = list(range(compare.PILOT_STEPS+compare.INTERVAL,
                               compare.TARGET+1, compare.INTERVAL))
            self.assertEqual(len(steps), 12)
            for step in steps:
                (root/f"step-{step:012d}").mkdir()
            def candidate(path, step, config):
                return dict(checkpoint=str(path), steps=step, mission_games=0,
                            highest_stage=2 if step in (steps[4], steps[8]) else 1,
                            mean_score=600 if step in (steps[4], steps[8]) else 1000)
            with patch.object(compare, "RUN", root), patch.object(compare, "checked_config", return_value={}), \
                    patch.object(compare, "candidate", side_effect=candidate):
                selected = compare.select_checkpoint()
            self.assertEqual(len(selected["candidates"]), 12)
            self.assertEqual(selected["selected"]["steps"], steps[4])

    def test_normal_exact_target_stop_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"latest").mkdir()
            (root/"status.json").write_text(json.dumps(dict(event="stopped", pid=1)))
            (root/"latest/state.json").write_text(json.dumps(dict(steps=compare.TARGET, config={})))
            (root/"latest/model.safetensors").touch()
            (root/"latest/optimizer.npz").touch()
            with patch.object(compare, "RUN", root), \
                    patch.object(compare, "checked_config", return_value={}), \
                    patch.object(compare, "process_command", return_value=""):
                self.assertEqual(compare.run_condition(), "complete")
                (root/"latest/state.json").write_text(json.dumps(dict(steps=compare.TARGET-1, config={})))
                self.assertEqual(compare.run_condition(), "stopped_early")
                (root/"latest/state.json").write_text(json.dumps(dict(steps=compare.TARGET, config={})))
                (root/"latest/model.safetensors").unlink()
                self.assertEqual(compare.run_condition(), "incompatible")

    def test_paired_summary_keeps_fresh_arm_and_step_counts(self):
        def evaluation(score, steps):
            return dict(mean_score=float(score), median_score=float(score), best_score=score,
                        highest_stage=1, mission_games=0,
                        games=[dict(score=score, steps=steps) for _ in range(compare.FRESH_GAMES)])
        result = compare.paired_summary(evaluation(400, 1700), evaluation(380, 1650))
        self.assertEqual(result["extension_mean"], 400)
        self.assertEqual(result["pilot_mean_steps"], 1650)
        self.assertEqual(result["paired_extension_wins"], 128)
        self.assertEqual(result["paired_pilot_wins"], 0)


if __name__ == "__main__":
    unittest.main()
