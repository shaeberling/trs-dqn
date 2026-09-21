import json
from pathlib import Path
import tempfile
import unittest

from rl.status import read_status


class StatusTests(unittest.TestCase):
    def test_curriculum_segments_are_separate_from_full_game_records(self):
        rows = [dict(event="start", steps=0, config={"steps": 1000}),
                dict(event="episode", steps=100, level=2, score=80, terminated=True),
                dict(event="curriculum_archive", steps=120, level=5),
                dict(event="curriculum_episode", steps=140, level=10, score=700,
                     terminated=True, full_game=False)]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"metrics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows)+"\n")
            result = read_status(directory)
        self.assertEqual(result["highest_training_level"], 2)
        self.assertEqual(result["best_training_score"], 80)
        self.assertEqual(result["curriculum"], dict(archive_additions=1, segments=1,
                                                   highest_segment_level=10))
        self.assertFalse(result["target_met"])

    def test_evaluation_progress_is_visible_only_for_the_current_suite(self):
        evaluation = dict(event="evaluation_progress", steps=100, active_games=1,
                          active=[dict(seed=10010, score=292, level=5, steps=90000, waiting=True)])
        rows = [dict(event="start", steps=0, config={"steps": 1000}),
                dict(event="validation_start", steps=100), evaluation]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            self.assertEqual(read_status(directory)["evaluation_progress"], evaluation)
            for last in [dict(event="progress", steps=120),
                         dict(event="validation_start", steps=200),
                         dict(event="validation_cancelled", steps=100),
                         dict(event="stopped", steps=100)]:
                path.write_text("\n".join(json.dumps(row) for row in rows+[last])+"\n")
                self.assertIsNone(read_status(directory)["evaluation_progress"])

    def test_stopped_error_and_partial_final_line_are_reported(self):
        rows = [dict(event="start", steps=10, config={"steps": 100}),
                dict(event="episode", steps=20, level=3, terminated=True),
                dict(event="episode", steps=30, level=5, terminated=False),
                dict(event="error", steps=40, error="screen error"),
                dict(event="stopped", steps=40, target_met=False)]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"metrics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows)+'\n{"event":')
            result = read_status(directory)
        self.assertEqual(result["last_event"], "stopped")
        self.assertEqual(result["steps"], 40)
        self.assertEqual(result["start_steps"], 10)
        self.assertEqual(result["budget_end_steps"], 100)
        self.assertEqual(result["highest_training_level"], 3)
        self.assertEqual(result["last_error"]["error"], "screen error")

    def test_empty_log_does_not_claim_running_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"metrics.jsonl").write_text("")
            result = read_status(directory)
        self.assertEqual(result["status"], "no complete log records")


if __name__ == "__main__":
    unittest.main()
