import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_learning import sha256
from rl.defense_loss_probe import analyze, loss_windows, render_sheet


class LossProbeTests(unittest.TestCase):
    def fixture(self):
        frames = np.full((321, 16, 64), 128, np.uint8)
        actions = np.full(320, 4, np.uint8)
        rewards = np.zeros(320, np.float32)
        events = []
        for i in range(4):
            stop = (i+1)*80
            rewards[stop-21] = 100
            if i < 3:
                frames[stop-10, 1:] = 191
            events.append(dict(frame=stop, life_lost=True, score=(i+1)*100, lives=3-i))
        metadata = dict(events=events, result=dict(terminated=True, truncated=False,
                        missions_completed=0, highest_stage=1, score=400))
        return frames, actions, rewards, metadata

    def test_score_windows_flash_alignment_and_no_input_mutation(self):
        frames, actions, rewards, metadata = self.fixture()
        before = frames.copy(), actions.copy(), rewards.copy(), copy.deepcopy(metadata)
        windows = loss_windows(frames, actions, rewards, metadata)
        self.assertEqual([w['score_gained_this_life'] for w in windows], [100]*4)
        self.assertEqual([w['first_major_white_flash_in_last_128_frames'] for w in windows],
                         [70, 150, 230, None])
        self.assertEqual(windows[-1]['panel_frames'], [256, 288, 312, 319])
        self.assertEqual(windows[0]['action_counts']['RIGHT'], 64)
        self.assertEqual(windows[0]['movement_action_fraction'], 1)
        self.assertEqual(windows[0]['recent_visible_score_increments'], [dict(frame=60, points=100.)])
        for old, new in zip(before[:3], (frames, actions, rewards)):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(metadata, before[3])

    def test_rejects_invalid_or_incomplete_evidence(self):
        for problem in ['frame_type', 'bad_action', 'bad_reward', 'score_mismatch', 'later_stage',
                        'missing_loss', 'reordered_losses', 'truncated']:
            frames, actions, rewards, metadata = self.fixture()
            if problem == 'frame_type': frames = frames.astype(np.float32)
            if problem == 'bad_action': actions[0] = 20
            if problem == 'bad_reward': rewards[0] = np.nan
            if problem == 'score_mismatch': rewards[0] = 1
            if problem == 'later_stage': metadata['result']['highest_stage'] = 2
            if problem == 'missing_loss': metadata['events'].pop()
            if problem == 'reordered_losses': metadata['events'].reverse()
            if problem == 'truncated': metadata['result']['truncated'] = True
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                loss_windows(frames, actions, rewards, metadata)

    def test_preserved_verified_trace_and_checksum_rejection(self):
        source = Path('results/defense/learned/versions/step-000008342272-125346536cb1-seed-10004').resolve()
        before = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
        report, frames, actions = analyze(source)
        self.assertEqual([w['score_gained_this_life'] for w in report['lives']], [2620]*4)
        self.assertFalse(report['native_reexecution'])
        self.assertFalse(report['training_data_written'])
        self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir() if p.is_file()})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_sheet(root/'screens.png', 'Test', report, frames, actions)
            from PIL import Image
            with Image.open(root/'screens.png') as image:
                self.assertEqual(image.size, (2048, 1896))
            (root/'manifest.json').write_text(json.dumps(dict(hashes={'model.safetensors': 'bad'})))
            (root/'model.safetensors').write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                analyze(root)


if __name__ == '__main__':
    unittest.main()
