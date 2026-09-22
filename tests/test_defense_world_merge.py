import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense import GAME_SHA256, ENVIRONMENT_VERSION
from rl.defense_learning import sha256
from rl.defense_world_data import SCHEMA, Sequences
from rl.defense_world_merge import merge


class WorldMergeTests(unittest.TestCase):
    def source(self, root, name, seed):
        directory = root/name; directory.mkdir()
        rows = []
        for i, split in enumerate(('train', 'heldout')):
            filename = f'episode-{i}.npz'
            np.savez_compressed(directory/filename,
                frames=np.full((5,16,64), seed+i, np.uint8),
                actions=np.arange(4,dtype=np.int32), rewards=np.zeros(4,np.float32),
                continuation=np.array([1,1,1,0],np.float32))
            rows.append(dict(file=filename, sha256=sha256(directory/filename), seed=seed+i,
                             split=split, steps=4, result={'score':0}))
        manifest = dict(schema=SCHEMA,purpose='new-own-experience-for-world-model',
            game_sha256=GAME_SHA256,environment_version=ENVIRONMENT_VERSION,tstates=100000,
            observation_stride=1,parent_hashes={'model.safetensors':'own-parent'},
            existing_evaluation_data=False,episodes=rows,epsilon=.25)
        (directory/'manifest.json').write_text(json.dumps(manifest))
        return directory

    def test_preserves_bytes_labels_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            a,b=self.source(root,'a',10),self.source(root,'b',20)
            before=[sha256(p/'manifest.json') for p in (a,b)]
            result=merge([a,b],root/'union')
            self.assertEqual(result['games'],4);self.assertEqual(result['heldout_games'],2)
            self.assertEqual(before,[sha256(p/'manifest.json') for p in (a,b)])
            for row in result['episodes']:
                origin=(a,b)[row['source_index']]/row['source_file']
                self.assertEqual(sha256(origin),sha256(root/'union'/row['file']))
            train=Sequences(root/'union','train',2)
            held=Sequences(root/'union','heldout',2)
            self.assertFalse(set(train.files)&set(held.files))
            values=train.sample(100,np.random.default_rng(4))[0][:,0,0,0]
            self.assertEqual(set(values),{10,20})
            self.assertEqual(set(held.sample(100,np.random.default_rng(4))[0][:,0,0,0]),{11,21})

    def test_rejects_duplicates_overlap_and_incompatible_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            a,b=self.source(root,'a',10),self.source(root,'b',20)
            for sources in ([a],[a,a]):
                with self.assertRaises(ValueError):merge(sources,root/'bad')
                self.assertFalse((root/'bad').exists())
            original=json.loads((b/'manifest.json').read_text())
            changed=dict(original,tstates=50000)
            (b/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'incompatible'):merge([a,b],root/'bad')
            changed=original;changed['episodes'][0]['seed']=10
            (b/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'overlapping'):merge([a,b],root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_policy_mixture_requires_opt_in_and_preserves_parent_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = self.source(root, 'a', 10), self.source(root, 'b', 20)
            actor = json.loads((b/'manifest.json').read_text())
            actor['parent_hashes'] = {'model.safetensors': 'another-own-policy'}
            actor['algorithm'] = 'gaussian-rssm-imagined-reinforce'
            (b/'manifest.json').write_text(json.dumps(actor))
            with self.assertRaisesRegex(ValueError, 'policy parents'):
                merge([a, b], root/'bad')
            self.assertFalse((root/'bad').exists())
            merged = merge([a, b], root/'union', allow_policy_mixture=True)
            self.assertIsNone(merged['parent_hashes'])
            self.assertTrue(merged['policy_mixture'])
            self.assertEqual(merged['source_manifests'][1]['parent_hashes'], actor['parent_hashes'])
            self.assertEqual(merged['source_manifests'][1]['algorithm'], actor['algorithm'])
            for row in merged['episodes']:
                original = (a, b)[row['source_index']]/row['source_file']
                self.assertEqual(sha256(original), row['sha256'])
            self.assertEqual([r['split'] for r in merged['episodes']], ['train', 'heldout']*2)


if __name__=='__main__':
    unittest.main()
