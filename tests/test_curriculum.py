from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rl.curriculum import CurriculumEnv
from rl.env import BreakdownEnv, SHAPE
from rl.snapshot import Snapshot, capture, restore
from rl.vector import VectorEnv


def continuation_in_peer(pipe, saved, steps):
    env = BreakdownEnv(max_steps=0)
    try:
        env.reset(99)
        restore(env, saved)
        results = []
        for _ in range(steps):
            obs, reward, terminal, truncated, info = env.step(3)
            results.append((obs.tobytes(), reward, terminal, truncated, info))
        pipe.send(results)
    finally:
        env.close()
        pipe.close()


class SnapshotTests(unittest.TestCase):
    def test_snapshot_reproduces_full_game_in_another_process(self):
        env = BreakdownEnv(max_steps=0)
        try:
            env.reset(12)
            for _ in range(40):
                env.step(3)
            saved = capture(env)
            expected = []
            for _ in range(2000):
                obs, reward, terminal, truncated, info = env.step(3)
                expected.append((obs.tobytes(), reward, terminal, truncated, info))
                if terminal:
                    break
            self.assertTrue(expected[-1][2])
        finally:
            env.close()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=continuation_in_peer, args=(child, saved, len(expected)))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), "peer snapshot continuation did not finish")
            self.assertEqual(parent.recv(), expected)
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()

    def test_archived_complete_game_compatibility_after_native_extension(self):
        directory = Path(__file__).resolve().parents[1]/"results"/"level10"
        baseline = json.loads((directory/"validation-entropy001-20m5.json").read_text())
        data = (directory/"snapshot-native-compatibility-validation.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "aba328fb0c301ff42c15580b4dd72590d73e398f98643c42d58fbe69d44630b2")
        actual = json.loads(data)
        self.assertEqual(actual["games"], baseline["games"])
        self.assertEqual(actual["complete_games"], 20)
        self.assertEqual(actual["incomplete_games"], 0)
        self.assertEqual(actual["mean_score"], 132.6)
        self.assertEqual(actual["highest_complete_level"], 5)
        self.assertEqual(actual["checkpoint_sha256"],
                         "510ce9632f744301f3a0f2b9c100b4c5c070afa88f1dbc466324c7c411d9fb70")

    def test_full_native_continuation_and_terminal_stop_roundtrip(self):
        env = BreakdownEnv(max_steps=0)
        try:
            env.reset(12)
            for _ in range(1000):
                _, reward, done, _, _ = env.step(3)
                self.assertFalse(done)
                if reward:
                    break
            else:
                self.fail("Test fixture did not receive a score reward")
            saved = capture(env)
            self.assertGreater(saved.score, 0)
            self.assertFalse(saved.frames.flags.writeable)
            expected = []
            for _ in range(2000):
                result = env.step(3)
                expected.append(result)
                if result[2]:
                    break
            self.assertTrue(expected[-1][2])
            with self.assertRaises(ValueError):
                capture(env)
            env.reset(99)
            boot_rng = env.rng.bit_generator.state
            env.close()  # Also disable the native terminal-text stop deliberately.
            np.testing.assert_array_equal(restore(env, saved), saved.frames)
            self.assertEqual(env.rng.bit_generator.state, boot_rng)
            for original in expected:
                actual = env.step(3)
                np.testing.assert_array_equal(actual[0], original[0])
                self.assertEqual(actual[1:], original[1:])
            self.assertEqual(env.score-saved.score,
                             sum(result[1] for result in expected))
            self.assertTrue(env.done)
        finally:
            env.close()

    def test_invalid_native_snapshot_does_not_mutate_environment(self):
        env = BreakdownEnv()
        try:
            env.reset(12)
            saved = capture(env)
            for invalid in (replace(saved, native=saved.native[:-1]),
                            replace(saved, native=b"FAIL"+saved.native[4:]),
                            replace(saved, tstates=50000)):
                with self.assertRaises(ValueError):
                    restore(env, invalid)
                self.assertEqual(capture(env).native, saved.native)
                np.testing.assert_array_equal(np.stack(env.frames), saved.frames)
        finally:
            env.close()


class CurriculumTests(unittest.TestCase):
    def test_four_point_spacing_preserves_reward_and_rebases_at_level_change(self):
        env = CurriculumEnv(seed=12, curriculum_min_level=8, curriculum_score_interval=4,
                            curriculum_probability=1)
        try:
            env.reset()
            def transition(level, score):
                def step(action):
                    reward = float(score-env.score)
                    env.level, env.score = level, score
                    return np.zeros(SHAPE, np.uint8), reward, False, False, {}
                with patch.object(BreakdownEnv, 'step', side_effect=step):
                    return env.step(1)
            self.assertEqual(transition(8, 510)[4]['curriculum_archive_add']['entry_kind'], 'level_entry')
            self.assertNotIn('curriculum_archive_add', transition(8, 513)[4])
            result = transition(8, 514)
            self.assertEqual(result[1], 1.)
            self.assertEqual(result[4]['curriculum_archive_add']['progress_since_previous_archive'], 4)
            self.assertEqual(transition(9, 515)[4]['curriculum_archive_add']['entry_kind'], 'level_entry')
            self.assertNotIn('curriculum_archive_add', transition(9, 518)[4])
            self.assertEqual(transition(9, 519)[4]['curriculum_archive_add']['entry_kind'], 'score_progress')
        finally:
            env.close()

    def test_progress_entries_are_bounded_and_use_only_new_visible_score(self):
        env = CurriculumEnv(seed=12, curriculum_min_level=8, curriculum_score_interval=16,
                            curriculum_share=True, worker_id=0, curriculum_probability=1,
                            curriculum_per_level=2)
        try:
            env.reset()
            def transition(level, score, done=False):
                def step(action):
                    reward = float(score-env.score)
                    env.level, env.score, env.done = level, score, done
                    return np.zeros(SHAPE, np.uint8), reward, done, False, {}
                with patch.object(BreakdownEnv, 'step', side_effect=step):
                    return env.step(2)
            self.assertNotIn('curriculum_archive_add', transition(7, 470)[4])
            entry = transition(8, 510)[4]['curriculum_archive_add']
            self.assertEqual(entry['entry_kind'], 'level_entry')
            self.assertNotIn('curriculum_archive_add', transition(8, 525)[4])
            result = transition(8, 526)
            self.assertEqual(result[1], 1.)
            self.assertEqual(result[4]['curriculum_archive_add']['entry_kind'], 'score_progress')
            self.assertEqual(result[4]['curriculum_archive_add']['progress_since_previous_archive'], 16)
            self.assertNotIn('curriculum_archive_add', transition(8, 526)[4])
            for score in (542, 558, 574):
                self.assertIn('curriculum_archive_add', transition(8, score)[4])
            self.assertEqual(len(env.archive[8]), 2)
            self.assertNotIn('curriculum_archive_add', transition(8, 590, True)[4])
        finally:
            env.close()

    def test_progress_offset_rebases_on_own_and_peer_restores(self):
        env = CurriculumEnv(seed=12, curriculum_score_interval=16, curriculum_min_level=8,
                            curriculum_share=True, worker_id=1, curriculum_probability=1)
        try:
            env.reset()
            # Synthetic routing metadata only; no claimed game or trained weights.
            saved = replace(capture(env), score=560, level=8, source_worker=0, source_action=19)
            env.receive_archive([saved])
            env.reset()
            self.assertEqual(env.last_archive_score, 560)
            self.assertEqual(env.segment_start_score, 560)
            self.assertEqual(env.episode_reward, 0)
            self.assertFalse(env.full_game)
            self.assertEqual(env.segment_source_action, 19)
            env.reset(99)
            self.assertTrue(env.full_game)
            self.assertEqual(env.last_archive_score, 0)
        finally:
            env.close()

    def test_progress_interval_validation(self):
        with patch.object(BreakdownEnv, '__init__') as initialize:
            for value in (-1, .5, True):
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    CurriculumEnv(curriculum_score_interval=value)
            initialize.assert_not_called()

    def test_level8_threshold_filters_publications(self):
        env = CurriculumEnv(seed=12, tstates=50000, observation_stride=2,
                            curriculum_min_level=8, curriculum_share=True,
                            worker_id=1, curriculum_probability=1)
        try:
            env.reset()
            # Routing-only fixtures; no synthetic state enters a training run.
            saved = replace(capture(env), level=8, source_worker=1, source_action=2)
            def enter(level):
                def step(action):
                    env.level = level
                    return np.zeros(SHAPE, np.uint8), 1.0, False, False, {}
                return step
            with patch.object(BreakdownEnv, 'step', side_effect=enter(7)):
                self.assertNotIn('curriculum_archive_add', env.step(4)[4])
            self.assertEqual(env.archive, {})
            with patch.object(BreakdownEnv, 'step', side_effect=enter(8)), \
                    patch('rl.curriculum.capture', return_value=saved):
                info = env.step(4)[4]
            self.assertEqual(info['curriculum_archive_add']['level'], 8)
            self.assertIs(info['_curriculum_snapshot'], saved)
            self.assertEqual(set(env.archive), {8})
        finally:
            env.close()

    def test_level8_peer_filter_is_atomic_and_protected_worker_still_boots(self):
        env = CurriculumEnv(seed=14, tstates=50000, observation_stride=2,
                            curriculum_min_level=8, curriculum_share=True,
                            worker_id=1, curriculum_probability=1, curriculum_reset=False)
        try:
            env.reset()
            before = capture(env)
            # Never restored: exercise archive eligibility, not gameplay claims.
            peer = replace(before, level=8, source_worker=0, source_action=7)
            env.receive_archive([peer])
            with self.assertRaises(ValueError):
                env.receive_archive([replace(peer, level=9), replace(peer, level=7)])
            self.assertEqual(set(env.archive), {8})
            self.assertEqual(len(env.archive[8]), 1)
            self.assertEqual(capture(env).native, before.native)
            np.testing.assert_array_equal(np.stack(env.frames), before.frames)
            with patch('rl.curriculum.restore', side_effect=AssertionError('protected restore')):
                env.reset()
            self.assertTrue(env.full_game)
            self.assertEqual(env.segment_start_level, 1)
        finally:
            env.close()

    def test_reserved_boot_worker_never_restores_even_with_peer_entries(self):
        expected = []
        env = BreakdownEnv(seed=14, max_steps=12)
        try:
            for _ in range(3):
                expected.append((env.reset(), [env.step(i % 6) for i in range(12)]))
        finally:
            env.close()
        env = CurriculumEnv(seed=14, max_steps=12, curriculum_share=True,
                            worker_id=1, curriculum_probability=1, curriculum_reset=False)
        try:
            with patch("rl.curriculum.restore", side_effect=AssertionError("boot worker restored")):
                for start, game in expected:
                    np.testing.assert_array_equal(env.reset(), start)
                    if not env.archive:
                        # Routing-only fixture; this synthetic entry is never restored.
                        env.receive_archive([replace(capture(env), level=2,
                                                     source_worker=0, source_action=17)])
                    for i, original in enumerate(game):
                        actual = env.step(i % 6)
                        np.testing.assert_array_equal(actual[0], original[0])
                        self.assertEqual(actual[1:4], original[1:4])
                        self.assertEqual({k: actual[4][k] for k in original[4]}, original[4])
                        self.assertTrue(actual[4]["full_game"])
                        self.assertIsNone(actual[4]["segment_source_worker"])
            self.assertEqual(len(env.archive[2]), 1)
        finally:
            env.close()

    def test_reserved_boot_worker_still_publishes_own_discoveries(self):
        env = CurriculumEnv(seed=12, curriculum_share=True, worker_id=0,
                            curriculum_probability=1, curriculum_reset=False)
        try:
            env.reset()
            # Synthetic transition tests publication only; no learned reach is claimed.
            saved = replace(capture(env), level=2, source_worker=0, source_action=1)

            def enter_level(action):
                env.level = 2
                return np.zeros(SHAPE, np.uint8), 1.0, False, False, {}

            with patch.object(BreakdownEnv, "step", side_effect=enter_level), \
                    patch("rl.curriculum.capture", return_value=saved):
                info = env.step(4)[4]
            self.assertTrue(info["full_game"])
            self.assertTrue(info["curriculum_archive_add"]["source_full_game"])
            self.assertIs(info["_curriculum_snapshot"], saved)
            self.assertEqual(len(env.archive[2]), 1)
        finally:
            env.close()

    def test_vector_assigns_reserved_boot_roles_without_gpu_imports(self):
        envs = VectorEnv(2, seed=12, max_steps=4, curriculum=True,
                         curriculum_probability=1, curriculum_share=True,
                         curriculum_boot_envs=1)
        try:
            runtime = envs.runtime()
            self.assertEqual([row["curriculum_reset_enabled"] for row in runtime], [False, True])
            self.assertTrue(all(not row["mlx_loaded"] for row in runtime))
            for _ in range(4):
                results = envs.step([3, 3])
            self.assertTrue(all(row[4]["full_game"] for row in results))
        finally:
            envs.close()

    def test_invalid_reserved_boot_allocation_fails_before_spawning(self):
        configs = [dict(curriculum_boot_envs=n, curriculum=True, curriculum_share=True)
                   for n in (-1, 2, 3, .5)]
        configs += [dict(curriculum_boot_envs=1),
                    dict(curriculum_boot_envs=1, curriculum=True),
                    dict(curriculum_boot_envs=1, curriculum_share=True)]
        with patch("rl.vector.mp.get_context") as context:
            for config in configs:
                with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                    VectorEnv(2, **config)
            context.assert_not_called()

    def test_peer_install_changes_only_archive_and_rejects_invalid_batch(self):
        env = CurriculumEnv(seed=12, curriculum_share=True, worker_id=1)
        try:
            env.reset(12)
            before = capture(env)
            # Synthetic routing fixture, never restored or used for learning.
            peer = replace(before, level=2, source_worker=0, source_action=17)
            boot_rng = env.rng.bit_generator.state
            env.receive_archive([peer])
            self.assertEqual(len(env.archive[2]), 1)
            self.assertIs(env.archive[2][0], peer)
            self.assertEqual(capture(env).native, before.native)
            np.testing.assert_array_equal(np.stack(env.frames), before.frames)
            self.assertEqual(env.rng.bit_generator.state, boot_rng)
            self.assertEqual(env.steps, before.steps)
            for invalid in (replace(peer, tstates=50000),
                            replace(peer, source_worker=1),
                            replace(peer, source_action=None)):
                with self.assertRaises(ValueError):
                    env.receive_archive([peer, invalid])
                self.assertEqual(env.encounters, {2: 1})
            env.curriculum_share = False
            with self.assertRaises(ValueError):
                env.receive_archive([peer])
        finally:
            env.close()

    def test_shared_segment_provenance_is_not_a_complete_game(self):
        env = CurriculumEnv(seed=12, curriculum_share=True, worker_id=1,
                            curriculum_probability=1)
        try:
            env.reset(12)
            # Cheap metadata fixture as in the local reset test below. Its
            # actual native state is a boot state; no level-2 reach is claimed.
            saved = replace(capture(env), source_worker=7, source_action=123)
            env.archive = {2: [saved]}
            env.reset()
            _, _, _, _, info = env.step(3)
            self.assertFalse(info["full_game"])
            self.assertEqual(info["segment_source_worker"], 7)
            self.assertEqual(info["segment_source_action"], 123)
            self.assertEqual(info["segment_start_score"], 0)
        finally:
            env.close()

    def test_vector_routes_only_to_peers_and_removes_opaque_payload(self):
        workers = VectorEnv.__new__(VectorEnv)
        workers.pipes = [Mock(), Mock(), Mock()]
        workers.share_curriculum = True
        saved = Snapshot(b"test-only", np.zeros(SHAPE, np.uint8), 60, 2, 100,
                         60.0, 2, 17, 100000, source_worker=2, source_action=100)
        results = [(np.zeros(SHAPE, np.uint8), 0, False, False,
                    {"_curriculum_snapshot": saved, "curriculum_archive_add": {}}, None),
                   (np.zeros(SHAPE, np.uint8), 0, False, False, {}, None)]
        workers._receive = Mock(side_effect=[*results, None, None])
        actual = workers.step([1, 3], indices=[2, 0])
        self.assertTrue(all("_curriculum_snapshot" not in row[4] for row in actual))
        self.assertEqual(actual[0][4]["curriculum_archive_add"]["shared_with"], 2)
        workers.pipes[2].send.assert_called_once_with(("step", 1))
        for index in [0, 1]:
            message = workers.pipes[index].send.call_args.args[0]
            self.assertEqual(message[0], "archive")
            self.assertIs(message[1][0], saved)
        self.assertEqual(workers._receive.call_count, 4)

    def test_disabled_curriculum_matches_seeded_boots_and_actions(self):
        expected = []
        env = BreakdownEnv(seed=14, max_steps=24)
        try:
            for _ in range(3):
                start = env.reset()
                game = [env.step(i % 6) for i in range(24)]
                expected.append((start, game))
        finally:
            env.close()
        env = CurriculumEnv(seed=14, max_steps=24, curriculum_probability=0)
        try:
            for start, game in expected:
                np.testing.assert_array_equal(env.reset(), start)
                for i, original in enumerate(game):
                    actual = env.step(i % 6)
                    np.testing.assert_array_equal(actual[0], original[0])
                    self.assertEqual(actual[1:4], original[1:4])
                    self.assertEqual({k: actual[4][k] for k in original[4]}, original[4])
                    self.assertTrue(actual[4]["full_game"])
            self.assertEqual(env.archive, {})
        finally:
            env.close()

    def test_restored_segments_only_reward_new_points_and_explicit_seed_boots(self):
        env = CurriculumEnv(seed=12, max_steps=2000, curriculum_probability=1)
        try:
            env.reset(12)
            for _ in range(1000):
                _, reward, done, _, _ = env.step(3)
                self.assertFalse(done)
                if reward:
                    break
            self.assertGreater(env.score, 0)
            saved = capture(env)
            # Cheap unit fixture: inject a self-reached level-1 state into a bank.
            # Production archive insertion only accepts actual level transitions.
            env.archive = {2: [saved]}
            np.testing.assert_array_equal(env.reset(), saved.frames)
            self.assertFalse(env.full_game)
            self.assertEqual(env.steps, 0)
            self.assertEqual(env.episode_reward, 0)
            total = 0
            for _ in range(2000):
                _, reward, done, truncated, info = env.step(3)
                total += reward
                self.assertFalse(info["full_game"])
                if done or truncated:
                    break
            self.assertEqual(total, env.score-saved.score)
            self.assertEqual(total, info["episode_reward"])
            self.assertEqual(info["segment_start_score"], saved.score)
            env.reset(12)
            self.assertTrue(env.full_game)
            self.assertEqual(env.level, 1)
            self.assertEqual(env.score, 0)
        finally:
            env.close()

    def test_archive_capacity_and_action_passthrough(self):
        env = CurriculumEnv(seed=12, curriculum_per_level=3)
        try:
            env.reset(12)

            def enter_level(action):
                env.level = 2
                return np.zeros(SHAPE, np.uint8), 1.0, False, False, {}

            with patch.object(BreakdownEnv, "step", side_effect=enter_level) as step, \
                    patch("rl.curriculum.capture", return_value=Mock(level=2)) as save:
                for i in range(50):
                    env.level = 1
                    result = env.step(i % 6)
                    step.assert_called_with(i % 6)
                    self.assertEqual(result[1:4], (1.0, False, False))
                self.assertEqual(env.encounters, {2: 50})
                self.assertEqual(len(env.archive[2]), 3)
                self.assertGreaterEqual(save.call_count, 3)
        finally:
            env.close()

    def test_vector_curriculum_starts_from_boot_without_external_archive(self):
        envs = VectorEnv(2, seed=12, max_steps=12, curriculum=True,
                         curriculum_probability=.5, curriculum_min_level=2,
                         curriculum_per_level=8, curriculum_share=True)
        try:
            self.assertEqual(envs.observations.shape, (2, *SHAPE))
            runtime = envs.runtime()
            self.assertEqual(len({row["pid"] for row in runtime}), 2)
            self.assertTrue(all(not row["mlx_loaded"] for row in runtime))
            for _ in range(12):
                results = envs.step([3, 3])
            for obs, reward, terminal, truncated, info, reset in results:
                self.assertFalse(terminal)
                self.assertTrue(truncated)
                self.assertTrue(info["full_game"])
                self.assertEqual(info["curriculum_archive_counts"], {})
                self.assertEqual(reset.shape, SHAPE)
        finally:
            envs.close()

    def test_invalid_curriculum_settings(self):
        for config in (dict(curriculum_probability=-.1), dict(curriculum_probability=1.1),
                       dict(curriculum_min_level=1), dict(curriculum_per_level=0),
                       dict(curriculum_share=True),
                       dict(curriculum_share=True, worker_id=1, curriculum_probability=0)):
            with self.assertRaises(ValueError):
                CurriculumEnv(**config)


if __name__ == "__main__":
    unittest.main()
