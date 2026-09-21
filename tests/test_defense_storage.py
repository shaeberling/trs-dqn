import json
from pathlib import Path
import sys
import tempfile
import unittest

from rl.defense_storage import clone_duplicate, digest, duplicates


class DefenseStorageTests(unittest.TestCase):
    def artifact(self, directory, data=b'own-model'):
        directory.mkdir(parents=True)
        (directory/'state.json').write_text(json.dumps(dict(config=dict(game='defense'))))
        path=directory/'model.safetensors';path.write_bytes(data)
        return path

    def test_inventory_excludes_live_paths_other_games_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp).resolve()
            a=self.artifact(repo/'runs/defense-example/step-000000000001')
            b=self.artifact(repo/'results/defense/training/example/checkpoint')
            self.artifact(repo/'runs/defense-example/latest')
            self.artifact(repo/'runs/other-game/step-000000000001')
            wrong=self.artifact(repo/'runs/defense-wrong/step-000000000001')
            (wrong.parent/'state.json').write_text(json.dumps(dict(config=dict(game='breakdown'))))
            link=repo/'runs/defense-example/best';link.symlink_to(a.parent)
            groups=duplicates(repo)
            self.assertEqual(len(groups),1)
            self.assertEqual(set(next(iter(groups.values()))),{a,b})

    @unittest.skipUnless(sys.platform=='darwin','APFS clone test')
    def test_clone_keeps_bytes_and_paths_and_modifications_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();a=root/'a';b=root/'b'
            a.write_bytes(b'same immutable artifact'*100);b.write_bytes(a.read_bytes())
            before=b.stat();checksum=digest(a)
            clone_duplicate(a,b,checksum)
            self.assertTrue(a.is_file() and b.is_file())
            self.assertEqual(digest(b),checksum)
            self.assertEqual(b.stat().st_mtime_ns,before.st_mtime_ns)
            self.assertEqual(b.stat().st_mode,before.st_mode)
            self.assertNotEqual(a.stat().st_ino,b.stat().st_ino)
            b.write_bytes(b'changed later')
            self.assertEqual(digest(a),checksum)
            self.assertFalse(list(root.glob('.defense-cow-*')))

    @unittest.skipUnless(sys.platform=='darwin','APFS clone test')
    def test_mismatch_or_symlink_leaves_destination_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();a=root/'a';b=root/'b';link=root/'link'
            a.write_bytes(b'one');b.write_bytes(b'two');link.symlink_to(a)
            with self.assertRaises(ValueError):clone_duplicate(a,b,digest(a))
            self.assertEqual(b.read_bytes(),b'two')
            with self.assertRaises(ValueError):clone_duplicate(link,b,digest(a))
            self.assertEqual(b.read_bytes(),b'two')


if __name__=='__main__':unittest.main()
