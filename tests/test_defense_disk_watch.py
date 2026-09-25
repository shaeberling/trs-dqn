import unittest

from rl.defense_disk_watch import expected_training_command


class DefenseDiskWatchTests(unittest.TestCase):
    def test_only_exact_training_run_can_be_signalled(self):
        run = 'runs/defense-ppo-balanced-fire-160/treatment'
        valid = f'venv/bin/python -u -m rl.defense_train --run {run} --steps 8388608'
        self.assertTrue(expected_training_command(valid, run))
        self.assertTrue(expected_training_command(valid.replace('rl.defense_train',
                                                        'rl.defense_dqn'), run))
        self.assertFalse(expected_training_command(valid, run+'-other'))
        self.assertFalse(expected_training_command(valid.replace('rl.defense_train',
                                                                 'rl.defense_collect'), run))
        self.assertFalse(expected_training_command(valid.replace('--run', '--artifacts'), run))


if __name__ == '__main__':
    unittest.main()
