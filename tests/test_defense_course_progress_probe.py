import unittest

from rl.defense_course_progress_probe import course_rows, probe_bundle, probe_sources


class DefenseCourseProgressProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundaries = course_rows()

    def test_original_course_row_contract(self):
        self.assertEqual(len(self.boundaries), 127)
        self.assertEqual(self.boundaries[0x76A9], 0)
        self.assertEqual(self.boundaries[0x7733], 33)
        self.assertEqual(self.boundaries[0x7735], 34)
        self.assertEqual(self.boundaries[0x7BD8], 126)

    def test_verified_own_replay_reexecutes_with_read_only_loss_rows(self):
        report = probe_bundle(
            "results/defense/training/ppo-continue-121/fresh-confirm-selected-replay",
            self.boundaries)
        self.assertEqual(report["verified_neural_actions"], 2551)
        self.assertEqual([row["decoded_rows"] for row in report["losses"]], [34, 33, 33, 34])
        self.assertEqual([row["displayed_score"] for row in report["losses"]],
                         [2620, 5240, 7860, 10480])

    def test_verified_own_source_states_show_same_early_course_cluster(self):
        report = probe_sources(
            "results/defense/training/ars-score-gated-69/run/own-loss-states",
            self.boundaries)
        self.assertEqual(report["own_games"], 12)
        self.assertEqual(report["own_life_states"], 48)
        self.assertEqual(report["decoded_row_histogram"],
                         {28: 1, 31: 1, 32: 8, 33: 6, 34: 32})

    def test_fine_cadence_calibration_replays_remain_quarantined_and_reexecutable(self):
        base = "results/defense/training/dqn-age-frontier-fine-178/calibration"
        for arm, actions, rows in (("treatment", 3824, [21, 21, 21, 20]),
                                   ("control", 3477, [16, 16, 15, 16])):
            with self.subTest(arm=arm):
                report = probe_bundle(f"{base}/{arm}/recheck", self.boundaries)
                self.assertEqual(report["verified_neural_actions"], actions)
                self.assertEqual([loss["decoded_rows"] for loss in report["losses"]], rows)


if __name__ == "__main__":
    unittest.main()
