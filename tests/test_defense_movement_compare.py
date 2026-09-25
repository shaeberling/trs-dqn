"""Fail-closed fixed selection and fresh comparison for movement pilot 164."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_movement_compare as compare


class MovementCompareTests(unittest.TestCase):
    def test_fixed_selector_prioritizes_stage_then_mean_then_earliest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = list(range(compare.INTERVAL, compare.TARGET + 1, compare.INTERVAL))
            for step in steps:
                (root/f"step-{step:012d}").mkdir()
            rows = {steps[0]: dict(mission_games=0, highest_stage=1, mean_score=9000),
                    steps[1]: dict(mission_games=0, highest_stage=2, mean_score=100),
                    steps[2]: dict(mission_games=0, highest_stage=2, mean_score=100),
                    steps[3]: dict(mission_games=0, highest_stage=1, mean_score=9500)}
            def candidate(path, step, config):
                return dict(checkpoint=str(path), steps=step, **rows[step])
            with patch.object(compare, "RUN", root), patch.object(compare, "checked_config", return_value={}), \
                    patch.object(compare, "candidate", side_effect=candidate):
                selected = compare.select_checkpoint()
            self.assertEqual(len(selected["candidates"]), 4)
            self.assertEqual(selected["selected"]["steps"], steps[1])

    def test_exact_target_stop_required_before_fresh_work(self):
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
                (root/"latest/optimizer.npz").unlink()
                self.assertEqual(compare.run_condition(), "incompatible")
                (root/"latest/optimizer.npz").touch()
                (root/"latest/state.json").write_text(json.dumps(dict(steps=compare.TARGET-1, config={})))
                self.assertEqual(compare.run_condition(), "stopped_early")

    def test_score_and_game_length_gate_is_diagnostic_only(self):
        def evaluation(score, steps, stage=1):
            return dict(mean_score=float(score), median_score=float(score), best_score=score,
                        highest_stage=stage, mission_games=0,
                        games=[dict(score=score, steps=steps) for _ in range(compare.FRESH_GAMES)])
        passed = compare.paired_summary(evaluation(321, 1531), evaluation(300, 1510))
        self.assertTrue(passed["stage_one_extension_gate"])
        self.assertEqual(passed["paired_movement_wins"], compare.FRESH_GAMES)
        self.assertFalse(compare.paired_summary(evaluation(319, 1531), evaluation(300, 1510))
                         ["stage_one_extension_gate"])
        self.assertFalse(compare.paired_summary(evaluation(321, 1529), evaluation(300, 1510))
                         ["stage_one_extension_gate"])
        self.assertFalse(compare.paired_summary(evaluation(321, 1531, stage=2), evaluation(300, 1510))
                         ["stage_one_extension_gate"])


if __name__ == "__main__":
    unittest.main()
