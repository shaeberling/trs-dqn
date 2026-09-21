from collections import deque
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense import DefenseEnv
from rl.defense_curriculum import DefenseCurriculumEnv
from rl.defense_learning import game_rank, summarize
from rl.defense_snapshot import capture
from rl.vector import VectorEnv


class DefenseCurriculumTests(unittest.TestCase):
    def test_disabled_curriculum_exactly_matches_original_full_game(self):
        actions = np.random.default_rng(12).integers(20, size=3000)
        env = DefenseEnv(max_steps=0)
        try:
            initial = env.reset(12)
            expected = []
            for action in actions:
                result = env.step(int(action))
                expected.append(result)
                if result[2]:
                    break
            self.assertTrue(expected[-1][2])
        finally:
            env.close()
        env = DefenseCurriculumEnv(max_steps=0, curriculum_probability=0)
        try:
            np.testing.assert_array_equal(env.reset(12), initial)
            for action, original in zip(actions, expected):
                result = env.step(int(action))
                np.testing.assert_array_equal(result[0], original[0])
                self.assertEqual(result[1:4], original[1:4])
                self.assertEqual({k: result[4][k] for k in original[4]}, original[4])
                self.assertTrue(result[4]["full_game"])
            self.assertEqual(env.archive, {})
        finally:
            env.close()

    def test_restored_segments_reward_only_new_points_and_do_not_rank(self):
        env = DefenseCurriculumEnv(max_steps=0, curriculum_probability=1)
        try:
            env.reset(12)
            for _ in range(3000):
                _, _, done, _, _ = env.step(0)
                if done:
                    break
            self.assertTrue(done)
            self.assertTrue(env.archive)
            env.reset()
            self.assertFalse(env.full_game)
            start_score = env.score
            self.assertGreater(start_score, 0)
            self.assertEqual(env.steps, 0)
            reward_sum = 0
            for _ in range(3000):
                _, reward, done, truncated, info = env.step(0)
                reward_sum += reward
                if done:
                    break
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertEqual(reward_sum, env.score-start_score)
            self.assertEqual(info["episode_reward"], reward_sum)
            self.assertIsNone(game_rank(info))
            self.assertEqual(summarize([info])["complete_games"], 0)
            env.reset(12)
            self.assertTrue(env.full_game)
            self.assertEqual(env.score, 0)
        finally:
            env.close()

    def test_progress_is_per_stage_and_life_not_cumulative_game_score(self):
        env = DefenseCurriculumEnv(max_steps=0, curriculum_probability=1)
        try:
            env.reset(12)
            losses, entries = 0, []
            for _ in range(3000):
                _, _, done, _, info = env.step(0)
                if info["life_lost"]:
                    losses += 1
                    self.assertNotIn("curriculum_archive_add", info)
                    self.assertEqual(env.progress_start_score, info["score"])
                if "curriculum_archive_add" in info:
                    entry = info["curriculum_archive_add"]
                    entries.append(entry)
                    self.assertEqual(entry["progress_bin"], entry["progress"]//20)
                    self.assertLessEqual(entry["progress"], 80)
                if done:
                    break
            self.assertEqual(losses, 4)
            self.assertTrue(any(e["score"] > e["progress"] for e in entries))
        finally:
            env.close()

    def test_peer_archive_preserves_active_game_and_boot_worker_never_restores(self):
        producer = DefenseCurriculumEnv(max_steps=0, curriculum_probability=1,
                                        curriculum_share=True, worker_id=0)
        try:
            producer.reset(12)
            for _ in range(200):
                info = producer.step(0)[4]
                if "_curriculum_snapshot" in info:
                    saved = info["_curriculum_snapshot"]
                    break
            else:
                self.fail("No own-play score snapshot reached")
        finally:
            producer.close()
        env = DefenseCurriculumEnv(max_steps=0, curriculum_probability=1,
                                   curriculum_share=True, worker_id=1, curriculum_reset=False)
        try:
            env.reset(99)
            before = capture(env)
            env.receive_archive([saved])
            self.assertEqual(capture(env).native, before.native)
            np.testing.assert_array_equal(np.stack(env.frames), before.frames)
            with patch("rl.defense_curriculum.restore", side_effect=AssertionError("protected worker")):
                env.reset()
            self.assertTrue(env.full_game)
            env.curriculum_reset = True
            env.reset()
            self.assertFalse(env.full_game)
            self.assertEqual(env.segment_source_worker, 0)
            self.assertEqual(env.segment_source_action, saved.source_action)
            for invalid in (replace(saved, action_count=21), replace(saved, source_worker=1),
                            replace(saved, source_action=0), replace(saved, progress_start_score=saved.score+1)):
                with self.assertRaises(ValueError):
                    env.receive_archive([invalid])
        finally:
            env.close()

    def test_archive_bounds_keep_deeper_bins_without_manufacturing_states(self):
        env = DefenseCurriculumEnv(curriculum_bins=2, curriculum_per_bin=2)
        try:
            # Reservoir bookkeeping only; no fabricated state is restored.
            for stage in (1, 2, 3):
                for bucket in range(10):
                    key = stage, bucket
                    slot = env._reserve_slot(key)
                    env.archive[key].append(None)
                    self.assertEqual(slot, 0)
                self.assertIsNone(env._reserve_slot((stage, 1)))
            self.assertEqual(len(env.archive), 6)
            self.assertEqual(set(k[1] for k in env.archive), {8, 9})
        finally:
            env.close()

    def test_vector_routing_strips_opaque_snapshots_and_protects_boot_worker(self):
        workers = VectorEnv(2, seed=12, game="defense", max_steps=0, curriculum=True,
                            curriculum_probability=1, curriculum_share=True, curriculum_boot_envs=1,
                            curriculum_lookback=8)
        try:
            self.assertEqual([r["curriculum_reset_enabled"] for r in workers.runtime()], [False, True])
            for _ in range(200):
                results = workers.step([0, 0])
                self.assertTrue(all("_curriculum_snapshot" not in r[4] for r in results))
                if any("curriculum_archive_add" in r[4] for r in results):
                    for result in results:
                        if "curriculum_archive_add" in result[4]:
                            self.assertEqual(result[4]["curriculum_archive_add"]["shared_with"], 1)
                            self.assertEqual(result[4]["curriculum_archive_add"]["lookback_actions"], 8)
                    break
            else:
                self.fail("No shared archive entry")
        finally:
            workers.close()

    def test_lookback_uses_exact_earlier_own_state_and_never_crosses_ship_loss(self):
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                                   worker_id=0, curriculum_lookback=8)
        visited = deque(maxlen=9)
        entries, losses = 0, 0
        try:
            env.reset(12)
            for _ in range(3000):
                _, _, done, _, info = env.step(0)
                if info["life_lost"]:
                    losses += 1
                    visited.clear()
                    self.assertEqual(len(env.history), 0)
                    self.assertNotIn("curriculum_archive_add", info)
                elif not done:
                    visited.append(capture(env))
                self.assertLessEqual(len(env.history), 9)
                if "_curriculum_snapshot" in info:
                    entries += 1
                    saved, expected = info["_curriculum_snapshot"], visited[0]
                    self.assertEqual(saved.native, expected.native)
                    np.testing.assert_array_equal(saved.frames, expected.frames)
                    self.assertEqual(saved.source_action, env.total_actions-8)
                    self.assertEqual(saved.lives, env.lives)
                    self.assertEqual(saved.stage, env.stage)
                    self.assertEqual(saved.progress_start_score, env.progress_start_score)
                    event = info["curriculum_archive_add"]
                    self.assertEqual(event["source_action"], saved.source_action)
                    self.assertEqual(event["source_episode_steps"], saved.steps)
                    self.assertEqual(event["score"], saved.score)
                    self.assertEqual(event["progress_bin"], (saved.score-saved.progress_start_score)//20)
                    self.assertEqual(event["lookback_actions"], 8)
                    self.assertEqual(event["trigger_action"], env.total_actions)
                    self.assertEqual(event["trigger_score"], env.score)
                if done:
                    break
            self.assertEqual(losses, 4)
            self.assertGreater(entries, 4)
            env.reset(99)
            self.assertEqual(len(env.history), 0)
        finally:
            env.close()

    def test_lookback_peer_restore_uses_saved_score_not_later_trigger_reward(self):
        producer = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                                        worker_id=0, curriculum_lookback=8)
        try:
            producer.reset(12)
            for _ in range(500):
                info = producer.step(0)[4]
                saved = info.get("_curriculum_snapshot")
                if saved is not None and saved.score > 0:
                    self.assertLess(saved.score, info["score"])
                    break
            else:
                self.fail("No positive-score lookback snapshot reached")
        finally:
            producer.close()
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                                   worker_id=1, curriculum_lookback=8)
        try:
            env.reset(99)
            env.receive_archive([saved])
            self.assertIn((saved.stage, (saved.score-saved.progress_start_score)//20), env.archive)
            restored = env.reset()
            np.testing.assert_array_equal(restored, saved.frames)
            self.assertEqual(capture(env).native, saved.native)
            self.assertEqual(env.segment_start_score, saved.score)
            self.assertFalse(env.full_game)
            self.assertEqual(len(env.history), 0)
            total = 0
            for _ in range(3000):
                _, reward, done, _, result = env.step(0)
                total += reward
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(total, env.score-saved.score)
            self.assertEqual(result["episode_reward"], total)
            self.assertIsNone(game_rank(result))
        finally:
            env.close()

    def test_lookback_stage_entry_is_immediate_and_clears_previous_history(self):
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                                   worker_id=0, curriculum_lookback=8)
        try:
            env.reset(12)
            for _ in range(10):
                env.step(0)
            self.assertEqual(len(env.history), 9)
            original = DefenseEnv.step

            def bookkeeping_transition(current, action):
                result = original(current, action)
                # Test bookkeeping only: no fabricated game state is restored
                # or used as training data, and no game bytes are changed.
                current.stage = 2
                current.highest_stage = 2
                return result

            with patch.object(DefenseEnv, "step", bookkeeping_transition):
                info = env.step(0)[4]
            saved = info["_curriculum_snapshot"]
            self.assertEqual(len(env.history), 1)
            self.assertEqual(saved.source_action, env.total_actions)
            self.assertEqual(saved.native, capture(env).native)
            self.assertEqual(info["curriculum_archive_add"]["entry_kind"], "stage_entry")
            self.assertEqual(info["curriculum_archive_add"]["lookback_actions"], 0)
        finally:
            env.close()

    def test_disabled_curriculum_does_not_capture_lookback_history(self):
        env = DefenseCurriculumEnv(curriculum_probability=0, curriculum_lookback=8)
        try:
            with patch("rl.defense_curriculum.capture", side_effect=AssertionError("disabled archive")):
                env.reset(12)
                for _ in range(100):
                    env.step(0)
            self.assertFalse(env.history)
            self.assertFalse(env.archive)
        finally:
            env.close()

    def test_lookback_collection_does_not_change_gameplay_or_rewards(self):
        actions = np.random.default_rng(12).integers(20, size=3000)
        original = DefenseEnv(max_steps=0)
        expected = []
        try:
            first = original.reset(12)
            for action in actions:
                result = original.step(int(action))
                expected.append(result)
                if result[2]:
                    break
            self.assertTrue(expected[-1][2])
        finally:
            original.close()
        for cells in ("score", "screen"):
            with self.subTest(cells=cells):
                env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_lookback=8,
                                           curriculum_cells=cells)
                try:
                    np.testing.assert_array_equal(env.reset(12), first)
                    for action, reference in zip(actions, expected):
                        result = env.step(int(action))
                        np.testing.assert_array_equal(result[0], reference[0])
                        self.assertEqual(result[1:4], reference[1:4])
                        self.assertEqual({key: result[4][key] for key in reference[4]}, reference[4])
                    self.assertTrue(env.archive)
                finally:
                    env.close()

    def test_invalid_settings(self):
        for config in (dict(curriculum_probability=float("nan")), dict(curriculum_probability=2),
                       dict(curriculum_bins=0), dict(curriculum_per_bin=0),
                       dict(curriculum_score_interval=-1), dict(curriculum_lookback=-1),
                       dict(curriculum_share=True),
                       dict(curriculum_share=True, worker_id=0, curriculum_probability=0)):
            with self.assertRaises(ValueError):
                DefenseCurriculumEnv(**config)
        with self.assertRaises(TypeError):
            DefenseCurriculumEnv(curriculum_lookback=1.5)


if __name__ == "__main__":
    unittest.main()
