import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_alias_probe import probability_summary, probe, replay_observation, statistics_probability_row


class AliasProbeTests(unittest.TestCase):
    def test_reporting_normalization_does_not_change_sampling_array(self):
        original = np.full(20, .05, np.float32)*np.float32(1.00001)
        before = original.copy()
        row, error = statistics_probability_row(original)
        self.assertGreater(error, 1e-6)
        self.assertAlmostEqual(row.sum(), 1.)
        probability_summary(row[None])
        np.testing.assert_array_equal(original, before)
        for invalid in [np.zeros(20), np.full(20, np.nan), -np.ones(20), np.ones(21)]:
            with self.assertRaises(ValueError): statistics_probability_row(invalid)

    def test_group_entropy_and_movement_mass(self):
        uniform = probability_summary(np.full((3, 20), .05))
        self.assertAlmostEqual(uniform['mean_raw_entropy_nats'], np.log(20))
        self.assertAlmostEqual(uniform['mean_alias_entropy_nats'], .45*np.log(9))
        self.assertAlmostEqual(uniform['mean_movement_probability'], .4)
        self.assertAlmostEqual(uniform['mean_forward_fire_alias_probability'], .45)
        aliases = np.zeros((1, 20)); aliases[:, 9:18] = 1/9
        report = probability_summary(aliases)
        self.assertAlmostEqual(report['mean_group_entropy_nats'], 0)
        self.assertAlmostEqual(report['alias_fraction_of_entropy'], 1)
        movement = np.zeros((1, 20)); movement[0, 4] = 1
        report = probability_summary(movement)
        self.assertEqual(report['mean_movement_probability'], 1)
        self.assertEqual(report['mean_alias_entropy_nats'], 0)

    def test_rejects_malformed_probabilities_and_frames(self):
        for p in [np.zeros((1, 20)), np.ones((1, 21))/21, np.zeros((0, 20)),
                  np.full((1, 20), np.nan), np.full((1, 20), -.05), np.ones(20)/20]:
            with self.assertRaises(ValueError): probability_summary(p)
        frames = np.broadcast_to(np.arange(10, dtype=np.uint8)[:, None, None], (10, 16, 64))
        np.testing.assert_array_equal(replay_observation(frames, 2, 2)[:, 0, 0], [0, 0, 0, 2])
        np.testing.assert_array_equal(replay_observation(frames, 9, 2)[:, 0, 0], [3, 5, 7, 9])
        for index in [-1, 10, True, 1.5]:
            with self.assertRaises(ValueError): replay_observation(frames, index, 1)
        with self.assertRaises(ValueError): replay_observation(frames.astype(np.int32), 0, 1)
        with self.assertRaises(ValueError): replay_observation(frames, 0, 0)

    def test_native_feedforward_and_recurrent_replay_action_identity(self):
        import mlx.core as mx
        from rl.defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
        from rl.defense_learning import publish_best, record_game, summarize, sha256
        from rl.defense_recurrent import RecurrentPPO
        from rl.ppo import PPO
        from rl.recurrent_policy import RECURRENT_ARCHITECTURE
        for recurrent, stride in [(False, 2), (True, 1)]:
            with self.subTest(recurrent=recurrent), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = dict(game='defense', algorithm='ppo', game_sha256=GAME_SHA256,
                              environment_version=ENVIRONMENT_VERSION, action_names=list(action_names()),
                              tstates=100000, observation_stride=stride, eval_max_steps=0)
                if recurrent:
                    agent = RecurrentPPO(seed=4, action_count=20, hidden_size=8)
                    agent.model.memory_actor.weight = mx.random.normal((20, 8))*.2
                    agent.compile()
                    config.update(architecture=RECURRENT_ARCHITECTURE, recurrent_hidden=8, memory_scale=1.)
                else:
                    agent = PPO(seed=4, action_count=20)
                agent.save(root/'checkpoint', dict(config=config, steps=0))
                trace = record_game(agent.policy(), 10000, tstates=100000, max_steps=0,
                                    observation_stride=stride)
                publish_best(root/'checkpoint/model.safetensors', summarize([trace[3]]), root/'artifacts')
                bundle = (root/'artifacts/best').resolve()
                before = {p.name: sha256(p) for p in bundle.iterdir()}
                report = probe(bundle)
                self.assertEqual(report['replay_actions_reproduced'], len(trace[1]))
                self.assertEqual(report['whole_replay']['observations'], len(trace[1]))
                self.assertEqual(len(report['pre_visible_loss_windows']), 4)
                self.assertFalse(report['native_reexecution'])
                self.assertFalse(report['training_data_written'])
                self.assertEqual(before, {p.name: sha256(p) for p in bundle.iterdir()})
                # Corrupt source content cannot masquerade as verified evidence.
                (bundle/'verification.json').write_text(json.dumps({'verified': True}))
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    probe(bundle)


if __name__ == '__main__':
    unittest.main()
