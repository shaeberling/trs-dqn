import json
from pathlib import Path
import tempfile
import unittest

from rl.status import read_status


class StatusTests(unittest.TestCase):
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
