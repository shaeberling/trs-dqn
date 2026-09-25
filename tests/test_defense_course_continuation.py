"""Guards for the quarantined row-50 course continuation."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from rl.defense_course_continuation import (
    SOURCE_FORK, SOURCE_FRAME, SOURCE_POSITION, SOURCE_ROWS, SOURCE_SCORE,
    load_source, run,
)
from rl.defense_learning import sha256


BUNDLE = Path("results/defense/learned/best")
DISCOVERY = Path("results/defense/diagnostics/course-feasibility-174/wide")


class CourseContinuationTests(unittest.TestCase):
    def test_row50_source_is_bound_to_protected_prefix(self):
        report, frames, prefix, rewards, metadata, actions = load_source(BUNDLE, DISCOVERY)
        self.assertEqual(len(prefix), SOURCE_FORK)
        self.assertEqual(len(rewards), SOURCE_FORK)
        self.assertEqual(len(actions), SOURCE_FRAME)
        self.assertGreater(len(frames), SOURCE_FRAME)
        self.assertEqual(metadata["result"]["score"], 10_480)
        self.assertTrue(report["original_native_verification"]["verified"])
        self.assertEqual((SOURCE_POSITION, SOURCE_SCORE, SOURCE_ROWS), (108, 450, 50))

    def test_archived_frontier_is_hash_bound_and_forensic_only(self):
        folder = Path("results/defense/diagnostics/course-continuation-176/full")
        report = json.loads((folder / "report.json").read_text())
        proof = json.loads((folder / "frontier-witness.json").read_text())
        record = folder / "frontier-witness.npz"
        self.assertEqual(report["frontier_witness"], proof)
        self.assertEqual(sha256(record), proof["diagnostic_trace_sha256"])
        with np.load(record, allow_pickle=False) as data:
            actions = data["actions"].copy()
            self.assertTrue(bool(data["diagnostic_only"]))
        self.assertEqual(len(actions), 684)
        self.assertEqual(hashlib.sha256(actions.tobytes()).hexdigest(),
                         proof["outcome"]["action_sha256"])
        self.assertTrue(proof["verification"]["every_screen_and_reward_matched"])
        self.assertFalse(proof["verification"]["promotion_eligible"])
        self.assertEqual((proof["outcome"]["stage"], proof["outcome"]["rows"]), (1, 60))

    def test_invalid_plan_or_existing_output_is_rejected_before_emulator_work(self):
        fresh = Path("runs/unused-course-continuation-test")
        for arguments in ({"beam_width": 0}, {"target_rows": SOURCE_ROWS},
                          {"max_frame": SOURCE_FRAME}, {"seed": -1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                run(BUNDLE, DISCOVERY, fresh, **arguments)
        with self.assertRaises(ValueError):
            run(BUNDLE, DISCOVERY, DISCOVERY)


if __name__ == "__main__":
    unittest.main()
