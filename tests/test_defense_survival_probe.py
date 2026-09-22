import unittest

from rl.defense_survival_probe import probe, summarize


class DefenseSurvivalProbeTests(unittest.TestCase):
    def test_counts_and_censoring(self):
        result = summarize([5, 64, 100, 320], [True, True, True, False], 64, [1, 1, 1, 2])
        self.assertEqual(result['visible_losses'], 3)
        self.assertEqual(result['horizon_censored'], 1)
        self.assertEqual(result['alive_beyond_original_visible_loss'], 2)
        self.assertEqual(result['alive_beyond_original_plus_32'], 2)
        self.assertEqual(result['alive_beyond_original_plus_128'], 1)
        self.assertEqual(result['stage_2_or_higher'], 1)
        with self.assertRaises(ValueError):
            summarize([10], [False], 64, [1])
        with self.assertRaises(ValueError):
            summarize([321], [True], 64, [1])

    def test_native_reproduction_repeatability_and_no_exported_routes(self):
        bundle = 'results/defense/training/dqn-50-restored-life-focused/replay-10460'
        a = probe(bundle, branches=2)
        self.assertEqual(a, probe(bundle, branches=2))
        self.assertEqual(a['original_native_trajectory_reproduced'], 2568)
        self.assertEqual(len(a['anchors']), 12)
        self.assertFalse(a['training_data_written'] or a['ranking_eligible'] or a['snapshots_decoded'])
        self.assertFalse(a['chosen_action_or_route_output'])
        self.assertEqual(a['parameter_updates'], 0)
        for anchor in a['anchors']:
            self.assertEqual(anchor['branches'], 2)
            self.assertEqual(anchor['visible_losses'] + anchor['horizon_censored'], 2)
            self.assertNotIn('actions', anchor)
            self.assertNotIn('score', anchor)
        for invalid in [0, -1, True, 4097]:
            with self.assertRaises(ValueError):
                probe(bundle, branches=invalid)


if __name__ == '__main__':
    unittest.main()
