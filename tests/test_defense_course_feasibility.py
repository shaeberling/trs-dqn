"""Guards for the quarantined original-course feasibility continuation."""

from pathlib import Path
import unittest

from rl.defense_course_feasibility import (
    SOURCE_ANCHOR, SOURCE_FRAME, SOURCE_POSITION, SOURCE_ROWS, SOURCE_SCORE,
    load_source, run,
)


BUNDLE = Path("results/defense/learned/best")
DISCOVERY = Path("results/defense/diagnostics/position-survival-172/wide")


class CourseFeasibilityTests(unittest.TestCase):
    def test_verified_input_archive_is_bound_to_expected_endpoint(self):
        report, frames, prefix, rewards, metadata, route = load_source(BUNDLE, DISCOVERY)
        self.assertEqual(len(prefix), SOURCE_ANCHOR)
        self.assertEqual(len(rewards), SOURCE_ANCHOR)
        self.assertEqual(len(route), SOURCE_FRAME - SOURCE_ANCHOR)
        self.assertGreater(len(frames), SOURCE_FRAME)
        self.assertTrue(report["original_native_verification"]["verified"])
        self.assertEqual(metadata["result"]["score"], 10_480)
        self.assertEqual((SOURCE_POSITION, SOURCE_SCORE, SOURCE_ROWS), (92, 390, 37))

    def test_invalid_or_overwriting_plan_rejected_before_emulator_work(self):
        for arguments in ({"beam_width": 0}, {"target_rows": SOURCE_ROWS},
                          {"max_frame": SOURCE_FRAME}, {"seed": -1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                run(BUNDLE, DISCOVERY, Path("runs/unused-course-feasibility-test"), **arguments)
        with self.assertRaises(ValueError):
            run(BUNDLE, DISCOVERY, DISCOVERY)


if __name__ == "__main__":
    unittest.main()
