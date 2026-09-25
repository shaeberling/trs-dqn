import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MultiLossLagTests(unittest.TestCase):
    def test_native_all_lives_match_four_verified_original_boot_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"lags.json"
            completed = subprocess.run([sys.executable, "-m", "rl.defense_internal_loss_lag_multi",
                "--output", str(output)], capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report["loss_events"], 16)
        self.assertEqual(len(report["bundles"]), 4)
        self.assertEqual(report["total_verified_actions"], sum(
            bundle["verified_actions"] for bundle in report["bundles"]))
        self.assertEqual(report["lag_min"], min(report["lags"]))
        self.assertEqual(report["lag_max"], max(report["lags"]))
        self.assertEqual(report["audited_private_counter_address"], "0x7CEF")
        for bundle in report["bundles"]:
            self.assertEqual(len(bundle["losses"]), 4)
            self.assertEqual(bundle["lag_min"], min(
                row["reporting_lag_actions"] for row in bundle["losses"]))
            for row in bundle["losses"]:
                self.assertEqual(row["reporting_lag_actions"],
                                 row["visible_loss_action"]-row["private_decrement"]["action_number"])
                self.assertEqual(len(row["private_decrement"]["visible_geometry"]), 4)
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["model_updates"], 0)
        self.assertFalse(report["training_data_written"])
        self.assertFalse(report["hidden_counter_used_by_policy_or_reward"])
        self.assertFalse(report["promotion_eligible"])

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exists.json"
            output.write_text("untouched")
            completed = subprocess.run([sys.executable, "-m", "rl.defense_internal_loss_lag_multi",
                "--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
