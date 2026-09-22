import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_spr_probe import path_starts, probe, summarize
from rl.defense_learning import sha256


class SprProbeTests(unittest.TestCase):
    def test_constant_and_nonconstant_features(self):
        constant = np.ones((8, 4))
        report = summarize(constant, constant, constant, constant)
        self.assertEqual(report['unit_variance_sum'], 0)
        self.assertEqual(report['constant_target_distance'], 0)
        self.assertEqual(report['prediction_action_sensitivity'], 0)
        varied = np.tile(np.eye(4), (2, 1))
        report = summarize(varied, varied, np.roll(varied, 1, axis=1), varied)
        self.assertEqual(report['recorded_action_distance'], 0)
        self.assertEqual(report['rotated_action_distance'], 1)
        self.assertEqual(report['constant_target_distance'], .5)
        self.assertEqual(report['unchanged_current_target_distance'], 0)
        self.assertGreater(report['unit_variance_sum'], 0)
        self.assertEqual(summarize(np.zeros((8, 4)), constant, constant, constant)['zero_target_vectors'], 8)

    def test_boundaries_and_invalid_inputs(self):
        self.assertEqual(path_starts(20, [10, 20], 5, 4), [0, 4, 10, 14])
        self.assertEqual(path_starts(20, [10, 20], 5, 5), [0, 5, 10, 15])
        for bounds in ([10], [10, 10, 20], [20, 10], [0, 20]):
            with self.assertRaises(ValueError):
                path_starts(20, bounds, 5)
        for values in (np.ones((1, 4)), np.full((8, 4), np.nan), np.ones(8)):
            with self.assertRaises(ValueError):
                summarize(values, values, values, values)
        with self.assertRaises(ValueError):
            summarize(np.ones((8, 4)), np.ones((8, 3)), np.ones((8, 4)), np.ones((8, 4)))

    def test_frozen_own_replay_determinism_and_provenance(self):
        root = Path('results/defense/training/spr-split-calibration-01').resolve()
        def hashes():
            return {str(p): sha256(p) for folder in ('checkpoint', 'replay')
                    for p in (root/folder).iterdir() if p.is_file()}
        before = hashes()
        first = probe(root/'checkpoint', root/'replay')
        second = probe(root/'checkpoint', root/'replay')
        self.assertEqual(first, second)
        self.assertEqual(first['replay_actions_reproduced'], 2567)
        self.assertEqual(first['paths'], 320)
        self.assertEqual(first['statistics']['vectors'], 1600)
        self.assertEqual(first['parameter_updates'], 0)
        self.assertFalse(first['training_data_written'])
        self.assertEqual(before, hashes())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for p in (root/'checkpoint').iterdir():
                if p.name != 'state.json': (directory/p.name).symlink_to(p)
            state = json.loads((root/'checkpoint/state.json').read_text())
            state['spr_auxiliary_hashes']['spr-ema.safetensors'] = 'bad'
            (directory/'state.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'matching full SPR'):
                probe(directory, root/'replay')
        self.assertEqual(before, hashes())


if __name__ == '__main__':
    unittest.main()
