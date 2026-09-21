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
                            curriculum_probability=1, curriculum_share=True, curriculum_boot_envs=1)
        try:
            self.assertEqual([r["curriculum_reset_enabled"] for r in workers.runtime()], [False, True])
            for _ in range(200):
                results = workers.step([0, 0])
                self.assertTrue(all("_curriculum_snapshot" not in r[4] for r in results))
                if any("curriculum_archive_add" in r[4] for r in results):
                    for result in results:
                        if "curriculum_archive_add" in result[4]:
                            self.assertEqual(result[4]["curriculum_archive_add"]["shared_with"], 1)
                    break
            else:
                self.fail("No shared archive entry")
        finally:
            workers.close()

    def test_invalid_settings(self):
        for config in (dict(curriculum_probability=float("nan")), dict(curriculum_probability=2),
                       dict(curriculum_bins=0), dict(curriculum_per_bin=0),
                       dict(curriculum_score_interval=-1), dict(curriculum_share=True),
                       dict(curriculum_share=True, worker_id=0, curriculum_probability=0)):
            with self.assertRaises(ValueError):
                DefenseCurriculumEnv(**config)


if __name__ == "__main__":
    unittest.main()
