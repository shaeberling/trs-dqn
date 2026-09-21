from dataclasses import replace
import multiprocessing as mp
import unittest
from unittest.mock import patch

import numpy as np

from rl.curriculum import CurriculumEnv
from rl.env import BreakdownEnv, SHAPE, validate_observation_stride
from rl.evaluate import categorical_policy, evaluate
from rl.inspect_policy import frame_stacks
from rl.snapshot import capture, restore
from rl.temporal_probe import TemporalPolicy
from rl.vector import VectorEnv


def screen_policy():
    return categorical_policy(lambda obs: np.sin(
        obs[:, :, 0, :].sum(axis=(1, 2))[:, None]+np.arange(6)[None, :]))


def snapshot_peer(pipe, saved, count):
    env = BreakdownEnv(max_steps=0, tstates=saved.tstates,
                       observation_stride=saved.observation_stride)
    try:
        env.reset(99)
        restore(env, saved)
        rows = []
        for _ in range(count):
            obs, *rest = env.step(3)
            rows.append((obs.tobytes(), *rest))
        pipe.send(rows)
    finally:
        env.close()
        pipe.close()


class ObservationStrideTests(unittest.TestCase):
    def test_invalid_stride_fails_before_emulator_or_policy(self):
        for value in (0, -1, 1.5, 2.0, True, np.bool_(True), '2', None):
            with self.subTest(value=value), patch('rl.env.TRS') as emulator:
                with self.assertRaises(ValueError):
                    BreakdownEnv(observation_stride=value)
                emulator.assert_not_called()
            with patch('rl.evaluate.BreakdownEnv') as emulator:
                with self.assertRaises(ValueError):
                    evaluate(lambda obs: self.fail('policy called'), [10000],
                             observation_stride=value)
                emulator.assert_not_called()
        self.assertEqual(validate_observation_stride(np.int64(2)), 2)

    def test_native_raw_history_spacing_preserves_actions_and_reward(self):
        env = BreakdownEnv(tstates=50000, max_steps=0)
        actions = [i % 6 for i in range(40)]
        try:
            frames = [env.reset(10015)[-1].copy()]
            expected = []
            for action in actions:
                obs, *rest = env.step(action)
                frames.append(obs[-1].copy())
                expected.append(rest)
        finally:
            env.close()
        env = BreakdownEnv(tstates=50000, max_steps=0, observation_stride=2)
        try:
            np.testing.assert_array_equal(env.reset(10015), np.stack([frames[0]]*4))
            for step, (action, original) in enumerate(zip(actions, expected), 1):
                obs, *rest = env.step(action)
                self.assertEqual(rest, original)
                self.assertEqual(obs.shape, SHAPE)
                self.assertEqual(obs.dtype, np.uint8)
                self.assertEqual(len(env.frames), 7)
                np.testing.assert_array_equal(obs,
                    np.stack([frames[max(0, step-offset)] for offset in (6, 4, 2, 0)]))
                # Returned stacks must not alias the saved history.
                obs.fill(0)
                np.testing.assert_array_equal(env.observation()[-1], frames[step])
        finally:
            env.close()

    def test_native_integrated_policy_matches_isolated_probe_and_scheduler(self):
        seeds = [10002, 10001, 10002]
        expected = evaluate(TemporalPolicy(screen_policy(), 2), seeds,
                            tstates=50000, max_steps=60)
        for count in (1, 2):
            actual = evaluate(screen_policy(), seeds, tstates=50000, max_steps=60,
                              observation_stride=2, envs=count)
            self.assertEqual(actual['games'], expected['games'])
            self.assertEqual(actual['observation_stride'], 2)

    def test_seven_frame_snapshot_full_continuation_in_peer(self):
        env = BreakdownEnv(max_steps=0, tstates=50000, observation_stride=2)
        try:
            env.reset(12)
            for _ in range(40):
                env.step(3)
            saved = capture(env)
            self.assertEqual(saved.observation_stride, 2)
            self.assertEqual(saved.frames.shape, (7, 16, 64))
            self.assertFalse(saved.frames.flags.writeable)
            expected = []
            for _ in range(4000):
                obs, *rest = env.step(3)
                expected.append((obs.tobytes(), *rest))
                if rest[1]:
                    break
            self.assertTrue(expected[-1][2])
            env.reset(99)
            rng = env.rng.bit_generator.state
            np.testing.assert_array_equal(restore(env, saved), saved.frames[::2])
            self.assertEqual(rng, env.rng.bit_generator.state)
        finally:
            env.close()
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=snapshot_peer, args=(child, saved, len(expected)))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), 'strided snapshot peer did not finish')
            self.assertEqual(parent.recv(), expected)
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()

    def test_mismatched_snapshot_and_peer_history_are_rejected_atomically(self):
        env = CurriculumEnv(observation_stride=2, tstates=50000, max_steps=0,
                            curriculum_share=True, worker_id=1, curriculum_probability=1)
        try:
            env.reset(12)
            saved = capture(env)
            for invalid in (replace(saved, observation_stride=1),
                            replace(saved, observation_stride=True),
                            replace(saved, frames=saved.frames[:4]),
                            replace(saved, frames=saved.frames.astype(float))):
                with self.assertRaises(ValueError):
                    restore(env, invalid)
                self.assertEqual(capture(env).native, saved.native)
                np.testing.assert_array_equal(np.stack(env.frames), saved.frames)
            peer = replace(saved, level=2, source_worker=0, source_action=1)
            for invalid in (replace(peer, observation_stride=1),
                            replace(peer, frames=peer.frames[:4])):
                with self.assertRaises(ValueError):
                    env.receive_archive([peer, invalid])
                self.assertEqual(env.archive, {})
                self.assertEqual(capture(env).native, saved.native)
            env.receive_archive([peer])
            restored = env.reset()
            self.assertFalse(env.full_game)
            self.assertEqual(env.episode_reward, 0)
            np.testing.assert_array_equal(restored, peer.frames[::2])
        finally:
            env.close()

    def test_replay_offsets_padding_and_batch_boundaries(self):
        frames = np.arange(12, dtype=np.uint8)[:, None, None]*np.ones((1, 16, 64), np.uint8)
        expected = np.array([[max(0, i-offset) for offset in (6, 4, 2, 0)] for i in range(11)])
        actual = frame_stacks(frames, 0, 11, observation_stride=2)
        np.testing.assert_array_equal(actual[:, :, 0, 0], expected)
        np.testing.assert_array_equal(actual[5:9], frame_stacks(frames, 5, 9, 2))
        np.testing.assert_array_equal(frame_stacks(frames, 0, 11),
                                      frame_stacks(frames, 0, 11, 1))
        with self.assertRaises(ValueError):
            frame_stacks(frames, 0, 1, True)

    def test_vector_stride_configuration_keeps_workers_cpu_only(self):
        envs = VectorEnv(2, seed=10000, tstates=50000, max_steps=5, observation_stride=2)
        try:
            self.assertEqual(envs.observations.shape, (2, *SHAPE))
            self.assertTrue(all(not row['mlx_loaded'] for row in envs.runtime()))
            for _ in range(5):
                rows = envs.step([3, 3])
            self.assertTrue(all(row[3] and row[5].shape == SHAPE for row in rows))
        finally:
            envs.close()


if __name__ == '__main__':
    unittest.main()
