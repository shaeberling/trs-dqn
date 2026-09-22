import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_inverse_probe import classification_metrics, probe, summarize
from rl.defense_learning import sha256


class InverseProbeTests(unittest.TestCase):
    def test_frozen_calibration_determinism_and_provenance(self):
        root=Path('results/defense/training/inverse-split-calibration-01').resolve()
        def hashes():
            return {str(p):sha256(p) for folder in ('checkpoint','replay')
                    for p in (root/folder).iterdir() if p.is_file()}
        before=hashes()
        first=probe(root/'checkpoint',root/'replay')
        second=probe(root/'checkpoint',root/'replay')
        self.assertEqual(first,second)
        verification=json.loads((root/'replay/verification.json').read_text())
        self.assertEqual(first['replay_actions_reproduced'],verification['verified_actions'])
        self.assertEqual(first['parameter_updates'],0)
        self.assertFalse(first['training_data_written'])
        self.assertEqual(before,hashes())
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for p in (root/'checkpoint').iterdir():
                if p.name != 'inverse-head.safetensors':(directory/p.name).symlink_to(p)
            (directory/'inverse-head.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'matching full inverse'):
                probe(directory,root/'replay')
        self.assertEqual(before,hashes())

    def test_uniform_and_exact_predictions(self):
        labels=np.array([0,1,2,3]*2)
        uniform=np.zeros((8,20))
        report=classification_metrics(uniform,labels)
        self.assertAlmostEqual(report['cross_entropy'],np.log(20))
        self.assertEqual(report['accuracy'],.25)
        self.assertEqual(np.sum(report['confusion_true_rows_predicted_columns']),8)
        perfect=np.full((8,20),-1000.);perfect[np.arange(8),labels]=1000.
        reports=summarize(labels,perfect,uniform,np.roll(perfect,1,axis=1))
        self.assertEqual(reports['all_examples']['paired']['accuracy'],1)
        self.assertEqual(reports['all_examples']['paired']['cross_entropy'],0)
        self.assertEqual(reports['paired_vs_permuted_disagreement'],1)
        self.assertEqual(reports['second_half']['first_half_frequency_prior']['accuracy'],.25)

    def test_invalid_inputs_and_no_mutation(self):
        labels=np.array([0,19,0,19]);logits=np.ones((4,20));copy=logits.copy()
        summarize(labels,logits,logits,logits)
        np.testing.assert_array_equal(logits,copy)
        for values,actions in [(np.full((4,20),np.nan),labels),(logits,np.array([0,20,0,19])),
                                (logits,labels.astype(float)),(np.ones((4,19)),labels),
                                (np.empty((0,20)),np.array([],int))]:
            with self.assertRaises(ValueError):classification_metrics(values,actions)
        with self.assertRaises(ValueError):summarize(np.array([0]),np.zeros((1,20)),np.zeros((1,20)),np.zeros((1,20)))


if __name__ == '__main__':
    unittest.main()
