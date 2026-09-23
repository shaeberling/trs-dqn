import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_ars_aggregate import archived_proposal, score_gradient


class AggregateScoreSearchTests(unittest.TestCase):
    def test_paired_finite_difference_respects_score_sign(self):
        directions = np.zeros((20, 20, 257), np.float32)
        directions[0, 4, 10] = 1
        scales = np.ones(20, np.float32)
        means = np.zeros(40, np.float64)
        means[0], means[1] = 100, 20
        direction, differences = score_gradient(directions, scales, means)
        self.assertEqual(float(direction[4, 10]), 40)
        self.assertEqual(float(differences[0]), 80)
        self.assertEqual(int(np.count_nonzero(direction)), 1)
        with self.assertRaises(ValueError):
            score_gradient(directions, np.zeros(20), means)

    def test_archived_proposal_requires_unchanged_incumbent(self):
        center = np.zeros((20, 257), np.float32)
        directions = np.zeros((20, 20, 257), np.float32)
        for direction in range(20):
            directions[direction, 4, 10+direction] = 1
        scales = np.ones(20, np.float32)
        heads = np.stack([np.stack((center+direction, center-direction))
                          for direction in directions]).reshape(40, 20, 257)
        games = [dict(candidate=candidate, repetition=repetition,
                      score=100 if candidate == 0 else 20 if candidate == 1 else 60)
                 for candidate in range(40) for repetition in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'population-000001').mkdir()
            (run/'metrics.jsonl').write_text(json.dumps(dict(event='generation',
                generation=1, accepted=False))+'\n')
            np.savez_compressed(run/'population-000001/plan.npz',
                                center=center, directions=directions,
                                scales=scales, heads=heads)
            (run/'population-000001/screen-games.json').write_text(json.dumps(games))
            proposals, record = archived_proposal(run, center, 1)
            self.assertEqual(proposals.shape, (8, 20, 257))
            self.assertGreater(float(proposals[0, 4, 10]), 0)
            np.testing.assert_allclose(proposals[0], -proposals[1])
            self.assertEqual(record['paired_directions'], 20)
            with self.assertRaises(ValueError):
                archived_proposal(run, center+1, 1)
            (run/'metrics.jsonl').write_text(json.dumps(dict(event='generation',
                generation=1, accepted=True))+'\n')
            with self.assertRaises(ValueError):
                archived_proposal(run, center, 1)


if __name__ == '__main__':
    unittest.main()
