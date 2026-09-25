from collections import deque
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense import DefenseEnv
from rl.defense_curriculum import DefenseCurriculumEnv
from rl.defense_learning import game_rank
from rl.defense_snapshot import capture, restore
from rl.vector import VectorEnv


class DefenseAgeTests(unittest.TestCase):
    def test_frontier_resets_use_only_latest_reached_age_bins(self):
        env = DefenseCurriculumEnv(curriculum_cells="age", curriculum_age_interval=8,
                                   curriculum_frontier_bins=2, curriculum_probability=1,
                                   curriculum_bins=4, curriculum_per_bin=1)
        try:
            env.reset(12)
            for _ in range(32):
                env.step(0)
            self.assertEqual(sorted(key[1] for key in env.archive), [1, 2, 3, 4])
            sampled = set()
            for _ in range(64):
                env.reset()
                self.assertFalse(env.full_game)
                sampled.add(env.life_steps//8)
            self.assertEqual(sampled, {3, 4})
            prior_score = env.score
            _, reward, _, _, _ = env.step(0)
            self.assertEqual(reward, env.score-prior_score)
        finally:
            env.close()

    def test_frontier_mode_rejects_non_age_or_out_of_capacity_settings(self):
        for extra in ({"curriculum_cells": "score", "curriculum_frontier_bins": 1},
                      {"curriculum_cells": "screen", "curriculum_frontier_bins": 1},
                      {"curriculum_cells": "age", "curriculum_frontier_bins": 5,
                       "curriculum_bins": 4},
                      {"curriculum_cells": "age", "curriculum_frontier_bins": -1}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                DefenseCurriculumEnv(**extra)

    def test_exact_lagged_age_states_bounded_later_bins_and_visible_life_resets(self):
        env = DefenseCurriculumEnv(curriculum_cells="age", curriculum_age_interval=32,
                                   curriculum_lookback=128, curriculum_probability=1,
                                   curriculum_share=True, worker_id=0,
                                   curriculum_bins=2, curriculum_per_bin=1)
        history = deque(maxlen=129)
        observed_keys = set()
        age, losses, entries, reward_free = 0, 0, 0, False
        try:
            env.reset(12)
            for _ in range(3000):
                _, reward, done, _, info = env.step(0)
                if info["life_lost"]:
                    age = 0
                    losses += 1
                    history.clear()
                    self.assertFalse(env.history)
                    self.assertNotIn("_curriculum_snapshot", info)
                else:
                    age += 1
                    history.append(capture(env))
                self.assertEqual(env.life_steps, age)
                self.assertLessEqual(len(env.history), 129)
                if "_curriculum_snapshot" in info:
                    entries += 1
                    saved, expected = info["_curriculum_snapshot"], history[0]
                    self.assertEqual(saved.native, expected.native)
                    np.testing.assert_array_equal(saved.frames, expected.frames)
                    self.assertEqual(saved.life_steps, age-128)
                    self.assertEqual(saved.source_action, env.total_actions-128)
                    self.assertEqual(saved.lives, env.lives)
                    event = info["curriculum_archive_add"]
                    self.assertEqual(event["entry_kind"], "life_age")
                    self.assertEqual(event["life_steps"], saved.life_steps)
                    self.assertEqual(event["life_age_bin"], saved.life_steps//32)
                    self.assertEqual(event["trigger_life_steps"], age)
                    self.assertEqual(event["lookback_actions"], 128)
                    observed_keys.add((saved.stage, saved.life_steps//32))
                    self.assertEqual(sorted(env.archive), sorted(observed_keys)[-2:])
                    self.assertTrue(all(len(bank) == 1 for bank in env.archive.values()))
                    reward_free |= reward == 0
                if done:
                    break
            self.assertEqual(losses, 4)
            self.assertGreater(entries, 4)
            self.assertTrue(reward_free)
            env.reset(99)
            self.assertEqual(env.life_steps, 0)
            self.assertFalse(env.history)
        finally:
            env.close()

    def test_peer_restores_saved_age_and_score_without_counter_or_reward_bonus(self):
        producer = DefenseCurriculumEnv(curriculum_cells="age", curriculum_age_interval=8,
                                        curriculum_lookback=8, curriculum_probability=1,
                                        curriculum_share=True, worker_id=0)
        try:
            producer.reset(12)
            for _ in range(200):
                info = producer.step(0)[4]
                saved = info.get("_curriculum_snapshot")
                if saved is not None and saved.life_steps >= 64:
                    break
            else:
                self.fail("No own age snapshot")
        finally:
            producer.close()
        env = DefenseCurriculumEnv(curriculum_cells="age", curriculum_age_interval=8,
                                   curriculum_lookback=8, curriculum_probability=1,
                                   curriculum_share=True, worker_id=1)
        try:
            env.reset(99)
            before = capture(env)
            for value in (-1, 1.5, True, float("nan")):
                bad = replace(saved, life_steps=value)
                with self.assertRaises(ValueError):
                    env.receive_archive([bad])
                with self.assertRaises(ValueError):
                    restore(env, bad)
                self.assertFalse(env.archive)
                self.assertEqual(capture(env).native, before.native)
            env.receive_archive([saved])
            self.assertIn((saved.stage, saved.life_steps//8), env.archive)
            np.testing.assert_array_equal(env.reset(), saved.frames)
            self.assertEqual(capture(env).native, saved.native)
            self.assertEqual(env.life_steps, saved.life_steps)
            self.assertEqual(env.steps, 0)  # New segment, not a new life-age counter.
            self.assertEqual(env.segment_start_score, saved.score)
            self.assertFalse(env.full_game)
            total = 0
            for _ in range(3000):
                _, reward, done, _, info = env.step(0)
                total += reward
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(total, env.score-saved.score)
            self.assertEqual(info["episode_reward"], total)
            self.assertIsNone(game_rank(info))
        finally:
            env.close()

    def test_stage_bookkeeping_resets_age_and_archives_new_stage_immediately(self):
        env = DefenseCurriculumEnv(curriculum_cells="age", curriculum_lookback=8,
                                   curriculum_probability=1, curriculum_share=True, worker_id=0)
        try:
            env.reset(12)
            for _ in range(10):
                env.step(0)
            self.assertEqual(env.life_steps, 10)
            original = DefenseEnv.step

            def bookkeeping_transition(current, action):
                result = original(current, action)
                # Unit bookkeeping fixture only, never training/evaluation data;
                # no native game bytes are patched and no fake state is restored.
                current.stage = current.highest_stage = 2
                return result

            with patch.object(DefenseEnv, "step", bookkeeping_transition):
                info = env.step(0)[4]
            saved = info["_curriculum_snapshot"]
            self.assertEqual(env.life_steps, 0)
            self.assertEqual(saved.life_steps, 0)
            self.assertEqual(len(env.history), 1)
            self.assertIn((2, 0), env.archive)
            self.assertEqual(info["curriculum_archive_add"]["lookback_actions"], 0)
            self.assertEqual(info["curriculum_archive_add"]["entry_kind"], "stage_entry")
        finally:
            env.close()

    def test_age_cells_route_between_workers_without_exposing_snapshots(self):
        workers = VectorEnv(2, seed=12, game="defense", curriculum=True,
                            curriculum_probability=1, curriculum_share=True, curriculum_boot_envs=1,
                            curriculum_cells="age", curriculum_age_interval=8, curriculum_lookback=8)
        try:
            self.assertEqual([r["curriculum_reset_enabled"] for r in workers.runtime()], [False, True])
            for _ in range(100):
                results = workers.step([0, 0])
                self.assertTrue(all("_curriculum_snapshot" not in r[4] for r in results))
                entries = [r[4]["curriculum_archive_add"] for r in results if "curriculum_archive_add" in r[4]]
                if entries:
                    self.assertTrue(all(e["entry_kind"] == "life_age" and e["shared_with"] == 1
                                        and e["trigger_life_steps"]-e["life_steps"] == 8 for e in entries))
                    break
            else:
                self.fail("No shared age archive entry")
        finally:
            workers.close()

    def test_invalid_age_interval_and_age_events_ignore_score_interval(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                DefenseCurriculumEnv(curriculum_age_interval=value)
        with self.assertRaises(TypeError):
            DefenseCurriculumEnv(curriculum_age_interval=1.5)
        env = DefenseCurriculumEnv(curriculum_cells="age", curriculum_age_interval=8,
                                   curriculum_score_interval=0, curriculum_lookback=0,
                                   curriculum_probability=1, curriculum_share=True, worker_id=0)
        try:
            env.reset(12)
            for _ in range(8):
                info = env.step(0)[4]
            self.assertEqual(info["curriculum_archive_add"]["entry_kind"], "life_age")
            self.assertEqual(info["curriculum_archive_add"]["life_steps"], 8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
