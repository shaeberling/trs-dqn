"""Guards for the quarantined mixed-cadence feasibility check."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from rl.defense import DefenseEnv
from rl.defense_course_continuation import load_source
from rl.defense_course_feasibility import verify_candidate
from rl.defense_course_progress_probe import course_rows
from rl.defense_fine_cadence_feasibility import (
    FINE_TSTATES, SOURCE_TSTATES, run, verify_mixed,
)
from rl.defense_learning import sha256


BUNDLE = Path("results/defense/learned/best")
SOURCE = Path("results/defense/diagnostics/course-feasibility-174/wide")


class FineCadenceFeasibilityTests(unittest.TestCase):
    def test_source_is_protected_and_original_cadence(self):
        report, _, _, _, _, actions = load_source(BUNDLE, SOURCE)
        self.assertEqual(report["tstates"], SOURCE_TSTATES)
        self.assertEqual(len(actions), 573)
        self.assertEqual(FINE_TSTATES, SOURCE_TSTATES // 2)

    def test_source_record_is_quarantined(self):
        path = SOURCE / "diagnostic-discovery.npz"
        proof = json.loads((SOURCE / "discovery.json").read_text())
        self.assertEqual(sha256(path), proof["diagnostic_trace_sha256"])
        with np.load(path, allow_pickle=False) as record:
            actions = record["actions"]
            self.assertTrue(bool(record["diagnostic_only"]))
        self.assertEqual(hashlib.sha256(actions.tobytes()).hexdigest(),
                         proof["outcome"]["action_sha256"])
        self.assertFalse(proof["verification"]["promotion_eligible"])

    def test_invalid_plan_rejected_before_emulator_work(self):
        unused = Path("runs/unused-fine-cadence-test")
        for arguments in ({"beam_width": 0}, {"target_rows": 50},
                          {"max_fine_actions": 0}, {"seed": -1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                run(BUNDLE, SOURCE, unused, **arguments)
        with self.assertRaises(ValueError):
            run(BUNDLE, SOURCE, SOURCE)

    def test_archived_row65_path_reexecutes_from_original_boot(self):
        directory = Path("results/defense/diagnostics/fine-cadence-survival-177/wide")
        report = json.loads((directory / "report.json").read_text())
        proof = json.loads((directory / "discovery.json").read_text())
        record = directory / "diagnostic-discovery.npz"
        self.assertEqual(report["discovery"], proof)
        self.assertEqual(sha256(directory / "source.py"), report["config"]["source_sha256"])
        self.assertEqual(sha256(record), proof["diagnostic_trace_sha256"])
        self.assertEqual((proof["outcome"]["rows"], proof["outcome"]["stage"]), (65, 1))
        self.assertFalse(proof["verification"]["promotion_eligible"])
        with np.load(record, allow_pickle=False) as data:
            actions = data["actions"].copy()
            self.assertTrue(bool(data["diagnostic_only"]))
        self.assertEqual(hashlib.sha256(actions.tobytes()).hexdigest(),
                         proof["outcome"]["action_sha256"])
        _, frames, prefix, rewards, metadata, source_actions = load_source(BUNDLE, SOURCE)
        np.testing.assert_array_equal(actions[:len(source_actions)], source_actions)
        fine = actions[len(source_actions):]
        self.assertEqual(len(fine), 333)
        env = DefenseEnv(tstates=SOURCE_TSTATES, max_steps=0,
                         observation_stride=metadata.get("observation_stride", 1))
        try:
            repeated = verify_mixed(env, metadata["result"]["seed"], frames, prefix,
                                    rewards, source_actions, fine, course_rows(),
                                    proof["outcome"])
        finally:
            env.close()
        self.assertEqual(repeated, proof["verification"])

    def test_archived_original_cadence_control_reexecutes_from_boot(self):
        directory = Path("results/defense/diagnostics/fine-cadence-survival-177/original-control")
        report = json.loads((directory / "report.json").read_text())
        proof = json.loads((directory / "frontier-witness.json").read_text())
        record = directory / "frontier-witness.npz"
        self.assertEqual(report["frontier_witness"], proof)
        self.assertIsNone(report["discovery"])
        self.assertEqual(report["final"]["max_live_rows"], 60)
        self.assertEqual(sha256(directory / "source.py"), report["config"]["source_sha256"])
        self.assertEqual(sha256(record), proof["diagnostic_trace_sha256"])
        self.assertFalse(proof["verification"]["promotion_eligible"])
        with np.load(record, allow_pickle=False) as data:
            actions = data["actions"].copy()
            self.assertTrue(bool(data["diagnostic_only"]))
        self.assertEqual(hashlib.sha256(actions.tobytes()).hexdigest(),
                         proof["outcome"]["action_sha256"])
        _, frames, prefix, rewards, metadata, source_actions = load_source(BUNDLE, SOURCE)
        np.testing.assert_array_equal(actions[:len(source_actions)], source_actions)
        env = DefenseEnv(tstates=SOURCE_TSTATES, max_steps=0,
                         observation_stride=metadata.get("observation_stride", 1))
        try:
            repeated = verify_candidate(env, metadata["result"]["seed"], frames,
                                        prefix, rewards, actions, course_rows(),
                                        proof["outcome"])
        finally:
            env.close()
        self.assertEqual(repeated, proof["verification"])


if __name__ == "__main__":
    unittest.main()
