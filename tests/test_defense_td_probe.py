import unittest

import numpy as np

from rl.defense_td_probe import probe, td_summary


class DefenseTDProbeTests(unittest.TestCase):
    def test_huber_scale_groups_and_degenerate_loss(self):
        report = td_summary([0., 2., 0.], [0., 0., 4.], [0., 0., 4.])
        self.assertEqual(report['transitions'], 3)
        self.assertEqual(report['mean_absolute_td'], 2.)
        self.assertEqual(report['linear_huber_fraction'], 2/3)
        self.assertAlmostEqual(report['mean_unit_weight_huber'], 5/3)
        self.assertEqual(report['groups'][0]['transitions'], 2)
        self.assertAlmostEqual(report['groups'][0]['huber_loss_share'], .3)
        self.assertAlmostEqual(report['groups'][1]['huber_loss_share'], .7)
        empty_group = td_summary([1.], [1.], [0.])
        self.assertEqual(empty_group['mean_unit_weight_huber'], 0.)
        self.assertEqual(empty_group['groups'][1]['huber_loss_share'], 0.)
        self.assertIsNone(empty_group['groups'][1]['mean_absolute_td'])
        for values in [([], [], []), ([0], [0, 1], [0]), ([np.nan], [0], [0])]:
            with self.assertRaises(ValueError):
                td_summary(*values)

    def test_frozen_one_and_five_option_traces_and_matching_target(self):
        for name, count, horizon in [('repeat-control-calibration-01', 1770, 1),
                                     ('repeat-five-variable-calibration-01', 2409, 5)]:
            root = 'results/defense/training/' + name
            result = probe(root+'/replay', root+'/checkpoint')
            self.assertEqual(result['commands_reproduced'], count)
            self.assertEqual(result['repeat_n_step'], horizon)
            self.assertEqual(result['td']['transitions'], result['completed_options'])
            self.assertGreaterEqual(result['terminal_returns'], 4)
            self.assertEqual(sum(result['raw_reward_counts'].values()), count)
            self.assertFalse(result['native_reexecution'] or result['training_data_written'] or result['ranking_eligible'])
            self.assertEqual(result['parameter_updates'], 0)
            self.assertTrue(result['original_native_verification']['verified'])
            self.assertEqual(result, probe(root+'/replay', root+'/checkpoint'))
        with self.assertRaises(ValueError):
            probe('results/defense/training/repeat-five-variable-calibration-01/replay',
                  'results/defense/training/repeat-five-control-calibration-01/checkpoint')


if __name__ == '__main__':
    unittest.main()
