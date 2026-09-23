from pathlib import Path
import unittest

from rl.defense_barrier_atlas import collect


ARCHIVE = (Path(__file__).resolve().parents[1]
           /'results/defense/training/ars-score-gated-69/run/own-loss-states')


class BarrierAtlasTests(unittest.TestCase):
    def test_own_visible_sources_and_score_band(self):
        index, entries = collect(ARCHIVE)
        self.assertEqual((len(index['games']), len(entries)), (12, 48))
        self.assertEqual(sum(2500 <= entry['displayed_life_score'] <= 2650
                             for entry in entries), 45)
        self.assertEqual(sum(entry['last_64_pure_right_actions'] for entry in entries), 894)
        self.assertEqual(sum(entry['last_64_pure_left_actions'] for entry in entries), 503)
        for entry in entries:
            self.assertEqual([panel['screen_index'] for panel in entry['panels']],
                             [63, 95, 119, 126])


if __name__ == '__main__':
    unittest.main()
