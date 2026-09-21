import hashlib
from itertools import permutations
import unittest

import numpy as np

from rl.defense_cells import screen_cell
from rl.defense_curriculum import DefenseCurriculumEnv
from rl.defense_learning import game_rank
from rl.defense_snapshot import capture
from rl.vector import VectorEnv


class DefenseCellTests(unittest.TestCase):
    def test_fixed_graphics_encoding_ignores_hud_and_prior_frames(self):
        frames = np.full((4, 16, 64), 128, np.uint8)
        expected = hashlib.blake2b(bytes(144), digest_size=16,
                                   person=b"defense-cell-v1").hexdigest()
        self.assertEqual(screen_cell(frames), expected)
        frames[:3] = 191
        frames[-1, 0] = np.arange(64, dtype=np.uint8)
        self.assertEqual(screen_cell(frames), expected)
        frames[-1, 1, 0] = 191  # six white pixels in the first 40-pixel block
        coarse = np.zeros((9, 16), np.uint8)
        coarse[0, 0] = 1
        expected = hashlib.blake2b(coarse.tobytes(), digest_size=16,
                                   person=b"defense-cell-v1").hexdigest()
        self.assertEqual(screen_cell(frames), expected)
        frames[-1, 1:] = 191
        expected = hashlib.blake2b(bytes([7])*144, digest_size=16,
                                   person=b"defense-cell-v1").hexdigest()
        self.assertEqual(screen_cell(frames), expected)

    def test_distinct_layouts_are_distinct_but_subquantum_changes_aggregate(self):
        frames = np.full((4, 16, 64), 128, np.uint8)
        blank = screen_cell(frames)
        frames[-1, 1, 0] = 129  # one pixel is below the quantization threshold
        self.assertEqual(screen_cell(frames), blank)
        frames[-1, 1, :] = 191
        top = screen_cell(frames)
        frames[-1, 1:] = 128
        frames[-1, -1, :] = 191
        self.assertNotEqual(screen_cell(frames), top)

    def test_invalid_frames_and_configuration_are_rejected(self):
        for frames in (np.zeros((16, 64), np.uint8), np.zeros((4, 16, 64), np.float32)):
            with self.assertRaises(ValueError):
                screen_cell(frames)
        for config in (dict(curriculum_cells="unknown"), dict(curriculum_screen_interval=0)):
            with self.assertRaises(ValueError):
                DefenseCurriculumEnv(**config)

    def test_bounded_cell_priorities_are_order_independent_and_ignore_score(self):
        env = DefenseCurriculumEnv(curriculum_cells="screen", curriculum_bins=2,
                                   curriculum_per_bin=1)
        try:
            frames = np.full((4, 16, 64), 128, np.uint8)
            self.assertEqual(env._key(1, 0, 0, frames), env._key(1, 10000, 9000, frames))
            for order in permutations(("01", "11", "ab", "ff")):
                env.archive.clear()
                env.encounters.clear()
                for cell in order:
                    # Reservoir bookkeeping only: no placeholder is restored.
                    key = (1, cell)
                    slot = env._reserve_slot(key)
                    if slot is not None:
                        env.archive[key].append(None)
                self.assertEqual(set(env.archive), {(1, "01"), (1, "11")})
                self.assertEqual(set(env.encounters), set(env.archive))
                self.assertIsNone(env._reserve_slot((1, "ff")))
        finally:
            env.close()

    def test_screen_events_can_be_reward_free_and_restore_own_earlier_state(self):
        producer = DefenseCurriculumEnv(curriculum_cells="screen", curriculum_screen_interval=8,
                                        curriculum_lookback=8, curriculum_probability=1,
                                        curriculum_share=True, worker_id=0)
        reward_free, candidate = False, None
        try:
            producer.reset(12)
            for _ in range(300):
                _, reward, done, _, info = producer.step(0)
                if "_curriculum_snapshot" in info:
                    saved = info["_curriculum_snapshot"]
                    entry = info["curriculum_archive_add"]
                    self.assertEqual(entry["entry_kind"], "screen_cell")
                    self.assertEqual(entry["screen_cell"], screen_cell(saved.frames))
                    self.assertEqual(entry["source_action"], producer.total_actions-8)
                    self.assertEqual(entry["score"], saved.score)
                    self.assertEqual(saved.lives, producer.lives)
                    reward_free |= reward == 0
                    if saved.score > 0:
                        candidate = saved
                if reward_free and candidate is not None:
                    break
                if done:
                    break
            self.assertTrue(reward_free)
            self.assertIsNotNone(candidate)
        finally:
            producer.close()
        env = DefenseCurriculumEnv(curriculum_cells="screen", curriculum_probability=1,
                                   curriculum_share=True, worker_id=1, curriculum_lookback=8)
        try:
            env.reset(99)
            env.receive_archive([candidate])
            self.assertIn((candidate.stage, screen_cell(candidate.frames)), env.archive)
            np.testing.assert_array_equal(env.reset(), candidate.frames)
            self.assertEqual(capture(env).native, candidate.native)
            self.assertEqual(env.segment_start_score, candidate.score)
            total = 0
            for _ in range(3000):
                _, reward, done, _, info = env.step(0)
                total += reward
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(total, env.score-candidate.score)
            self.assertEqual(info["episode_reward"], total)
            self.assertIsNone(game_rank(info))
        finally:
            env.close()

    def test_screen_cells_route_between_workers_without_exposing_snapshots(self):
        workers = VectorEnv(2, seed=12, game="defense", curriculum=True,
                            curriculum_probability=1, curriculum_share=True, curriculum_boot_envs=1,
                            curriculum_cells="screen", curriculum_screen_interval=8, curriculum_lookback=8)
        try:
            self.assertEqual([r["curriculum_reset_enabled"] for r in workers.runtime()], [False, True])
            for _ in range(200):
                results = workers.step([0, 0])
                self.assertTrue(all("_curriculum_snapshot" not in r[4] for r in results))
                entries = [r[4]["curriculum_archive_add"] for r in results if "curriculum_archive_add" in r[4]]
                if entries:
                    self.assertTrue(all(e["entry_kind"] == "screen_cell" and e["shared_with"] == 1 for e in entries))
                    break
            else:
                self.fail("No shared screen-cell archive entry")
        finally:
            workers.close()


if __name__ == "__main__":
    unittest.main()
