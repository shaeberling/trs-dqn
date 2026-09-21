import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.artifacts import FORMAT, digest, publish_record, record_rank, effort_rank
from rl.outcome import OUTCOME_VERSION
from rl.evaluate import summary


class ArtifactTests(unittest.TestCase):
    def test_verified_win_outranks_higher_scoring_nonwinner(self):
        winner = dict(level=8, score=500, steps=100, terminated=True, truncated=False,
                      game_over=True, game_won=True, outcome_version=OUTCOME_VERSION,
                      win_status='verified_win', reserve_balls_visible=1)
        loser = dict(level=8, score=999, steps=100, terminated=True)
        self.assertGreater(effort_rank(winner), effort_rank(loser))
        self.assertLess(effort_rank({**winner, 'full_game': False}), effort_rank(loser))

    def fixture(self, root, name='candidate', score=500):
        checkpoint = root/name
        checkpoint.mkdir()
        for filename in ('model.safetensors', 'optimizer.npz', 'state.json', 'evaluation.json'):
            (checkpoint/filename).write_text('synthetic never-loaded fixture '+name)
        replay, inspection = checkpoint/'source.html', checkpoint/'inspection.json'
        replay.write_text('synthetic replay '+name)
        inspection.write_text('{}')
        game = dict(score=score, level=8, steps=100, seed=10000, terminated=True, truncated=False)
        primary = {'games': [{**game, 'seed': s} for s in range(10000, 10020)]}
        secondary = {'games': [{**game, 'seed': s} for s in range(10100, 10150)]}
        return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint/'model.safetensors'),
                    replay=str(replay), inspection=str(inspection), replay_game=game,
                    primary=primary, secondary=secondary,
                    combined=summary(primary['games']+secondary['games']))

    def test_immutable_bundles_stable_replay_and_no_regression(self):
        with tempfile.TemporaryDirectory() as tmp, patch('rl.supervise.verify_replay'):
            root = Path(tmp); archive = root/'archive'
            first = self.fixture(root)
            a = publish_record(first, archive)
            original = (archive/'versions'/a['version']/'replay.html').read_bytes()
            second = self.fixture(root, 'better', 600)
            b = publish_record(second, archive)
            self.assertEqual((archive/'replay.html').read_bytes(), Path(second['replay']).read_bytes())
            self.assertEqual((archive/'versions'/a['version']/'replay.html').read_bytes(), original)
            self.assertEqual(publish_record(first, archive), b)
            for name, value in b['files'].items():
                self.assertEqual(digest(archive/'versions'/b['version']/name), value)
            self.assertTrue((archive/'versions'/b['version']/'selection.json').exists())

    def test_failed_verification_does_not_replace_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'archive'
            first = self.fixture(root)
            with patch('rl.supervise.verify_replay'):
                publish_record(first, archive)
            before = {n: (archive/n).read_bytes() for n in ('current.json', 'replay.html')}
            with patch('rl.supervise.verify_replay', side_effect=ValueError('mismatched action')):
                with self.assertRaises(ValueError):
                    publish_record(self.fixture(root, 'bad', 600), archive)
            self.assertEqual(before, {n: (archive/n).read_bytes() for n in before})

    def test_unmanaged_directory_and_wrong_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch('rl.supervise.verify_replay'):
            root = Path(tmp); archive = root/'archive'; archive.mkdir()
            record = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'unmanaged'):
                publish_record(record, archive)
            record['checkpoint_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'identity'):
                publish_record(record, root/'new')
            self.assertFalse((root/'new').exists())

    def test_selection_rejects_wrong_seeds_and_partial_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.fixture(Path(tmp))
            changed = copy.deepcopy(record)
            changed['secondary'] = changed['primary']
            with self.assertRaisesRegex(ValueError, 'seeds'):
                record_rank(changed, 'validation_selected')
            changed = copy.deepcopy(record)
            changed['primary']['games'][0]['truncated'] = True
            with self.assertRaisesRegex(ValueError, 'complete'):
                record_rank(changed, 'validation_selected')

    def test_single_effort_is_separate_from_selected_rank(self):
        with tempfile.TemporaryDirectory() as tmp, patch('rl.supervise.verify_replay'):
            root = Path(tmp); record = self.fixture(root)
            result = publish_record(record, root/'effort', 'single_game_best_effort')
            self.assertEqual(result['kind'], 'single_game_best_effort')
            with self.assertRaisesRegex(ValueError, 'separate'):
                publish_record(record, root/'effort', 'validation_selected')


if __name__ == '__main__':
    unittest.main()
