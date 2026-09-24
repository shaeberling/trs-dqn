import unittest

import numpy as np

from rl.defense_frontier_search import AgeFrontierArchive, FrontierArchive, trace_path


class DefenseFrontierSearchTests(unittest.TestCase):
    def test_age_frontier_retains_later_own_survival_without_changing_reward(self):
        rng = np.random.default_rng(17)
        archive = AgeFrontierArchive(2, 16)
        key_a, key_b, key_c = (character*32 for character in "abc")
        self.assertTrue(archive.add(key_a, 0, 32, "a0", rng))
        self.assertFalse(archive.add(key_a, 1, 32, "a1", rng))
        self.assertTrue(archive.add(key_a, 2, 40, "a2", rng))
        self.assertTrue(archive.add(key_a, 3, 48, "a3", rng))
        self.assertEqual(archive.cells[(key_a, 2)], (2, 40, "a2"))
        self.assertEqual(archive.cells[(key_a, 3)], (3, 48, "a3"))
        archive.add(key_b, 4, 64, "b4", rng)
        archive.add(key_c, 5, 80, "c5", rng)
        self.assertLessEqual(len(archive.cells), 2)
        self.assertEqual(len(archive.seen), 4)
        for _ in range(20):
            self.assertIn(archive.choose(rng), archive.cells.values())
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                AgeFrontierArchive(invalid, 16)
            with self.assertRaises(ValueError):
                AgeFrontierArchive(2, invalid)
        for key, age in (("bad", 10), (key_a, -1), (key_a, True)):
            with self.assertRaises(ValueError):
                archive.add(key, 6, age, "bad", rng)

    def test_archive_is_bounded_and_uses_visible_score_for_same_cell(self):
        rng = np.random.default_rng(13)
        archive = FrontierArchive(2)
        key_a, key_b, key_c = (character*32 for character in "abc")
        self.assertTrue(archive.add(key_a, 0, 20, "a0", rng))
        self.assertFalse(archive.add(key_a, 1, 20, "a1", rng))
        self.assertEqual(archive.cells[key_a], (0, 20, "a0"))
        self.assertTrue(archive.add(key_a, 2, 40, "a2", rng))
        self.assertTrue(archive.add(key_b, 3, 10, "b3", rng))
        archive.add(key_c, 4, 30, "c4", rng)
        self.assertLessEqual(len(archive.cells), 2)
        self.assertEqual(len(archive.seen), 3)
        for _ in range(20):
            node, score, state = archive.choose(rng)
            self.assertIn((node, score, state), archive.cells.values())
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                FrontierArchive(invalid)
        with self.assertRaises(ValueError):
            archive.add("bad", 5, 1, "bad", rng)
        with self.assertRaises(ValueError):
            archive.add(key_a, 5, -1, "bad", rng)

    def test_path_retains_own_source_prefix_and_all_random_macros(self):
        records = [dict(kind="own_prefix", source="seed-75000-life-1.npz", prefix=48),
                   dict(kind="random_hold", parent=0, action=4, length=8),
                   dict(kind="random_hold", parent=1, action=0, length=16)]
        self.assertEqual(trace_path(records, 2, 3, 4),
                         ("seed-75000-life-1.npz", 48, [(4, 8), (0, 16), (3, 4)]))
        self.assertEqual(trace_path(records, 0, 9, 2),
                         ("seed-75000-life-1.npz", 48, [(9, 2)]))


if __name__ == "__main__":
    unittest.main()
