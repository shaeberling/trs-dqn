import copy
from pathlib import Path
import unittest

import numpy as np

from rl.defense_response_probe import probe, screen_diversity


class DefenseResponseProbeTests(unittest.TestCase):
    def test_gameplay_diversity_ignores_hud_and_non_graphics_text(self):
        frame=np.full((16,64),128,np.uint8)
        branches=[[frame.copy()] for _ in range(20)]
        self.assertEqual(screen_diversity(branches)['distinct_gameplay_graphics_sequences'],1)
        branches[1][0][0,0]=ord('9')
        branches[2][0][3,4]=ord('A')
        result=screen_diversity(branches)
        self.assertEqual(result['distinct_screen_sequences'],3)
        self.assertEqual(result['distinct_gameplay_graphics_sequences'],1)
        branches[3][0][3,4]=129
        self.assertTrue(screen_diversity(branches)['any_gameplay_graphics_response'])

    def test_common_prefix_and_bad_shapes(self):
        frame=np.full((16,64),128,np.uint8)
        branches=[[frame.copy()] for _ in range(20)]
        longer=copy.deepcopy(branches)
        longer[0].append(np.full((16,64),191,np.uint8))
        self.assertEqual(screen_diversity(branches),screen_diversity(longer))
        for invalid in [branches[:19], [[]]+branches[1:], [[np.zeros((16,64))]]*20]:
            with self.assertRaises(ValueError):screen_diversity(invalid)

    def test_native_diagnostic_repeats_without_mutating_sources_or_importing_training(self):
        bundle=Path('results/defense/training/balanced-actions-calibration-01/replay')
        a=probe(bundle);b=probe(bundle)
        self.assertEqual(a,b)
        self.assertEqual(a['original_native_trajectory_reproduced'],2540)
        self.assertEqual(len(a['anchors']),16)
        self.assertFalse(a['training_data_written'] or a['ranking_eligible'] or a['snapshots_decoded'])
        self.assertFalse(a['chosen_action_or_route_output'])
        self.assertEqual(a['parameter_updates'],0)
        for anchor in a['anchors']:
            self.assertLessEqual(anchor['common_observed_decisions'],4)
            self.assertTrue(1<=anchor['distinct_gameplay_graphics_sequences']<=20)
        # These observed anchors are responsive; this is not a survival assertion.
        self.assertTrue(all(a['any_gameplay_graphics_response'] for a in a['anchors']
                            if a['decisions_before_visible_loss']==64))


if __name__=='__main__':unittest.main()
