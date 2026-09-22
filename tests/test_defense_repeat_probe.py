import unittest

from rl.defense_repeat_probe import duration_summary, probe


class DefenseRepeatProbeTests(unittest.TestCase):
    def test_window_counts_do_not_invent_a_start_at_window_edge(self):
        result = duration_summary([1, 1, 0, 1], [False, False, True, True], (1, 4))
        self.assertEqual(result['decisions_by_duration'], [1, 1])
        self.assertEqual(result['executed_commands_by_duration'], [1, 3])
        self.assertEqual(result['long_option_command_fraction'], .75)
        for indices, starts in [([], []), ([2], [True]), ([0], [1]), ([0, 1], [True])]:
            with self.assertRaises(ValueError):
                duration_summary(indices, starts, (1, 4))

    def test_frozen_control_and_variable_commands_and_source_integrity(self):
        control = probe('results/defense/training/repeat-control-calibration-01/replay')
        variable = probe('results/defense/training/repeat-variable-calibration-01/replay')
        self.assertEqual(variable, probe('results/defense/training/repeat-variable-calibration-01/replay'))
        self.assertEqual(control['replay_commands_reproduced'], 1770)
        self.assertEqual(variable['replay_commands_reproduced'], 2394)
        self.assertEqual(control['whole_replay']['neural_option_decisions'], 1770)
        self.assertEqual(control['whole_replay']['long_option_command_fraction'], 0.)
        for report in [control, variable]:
            self.assertFalse(report['native_reexecution'] or report['training_data_written'] or report['ranking_eligible'])
            self.assertEqual(report['parameter_updates'], 0)
            self.assertEqual(len(report['pre_visible_loss_windows']), 4)
            summary = report['whole_replay']
            self.assertEqual(sum(summary['executed_commands_by_duration']), report['replay_commands_reproduced'])
            planned = sum(d*n for d,n in zip(report['durations'], summary['decisions_by_duration']))
            self.assertEqual(planned, report['replay_commands_reproduced'] + report['cancelled_base_commands'])


if __name__ == '__main__':
    unittest.main()
