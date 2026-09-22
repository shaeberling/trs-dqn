from collections import deque
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense import DefenseEnv
from rl.defense_curriculum import DefenseCurriculumEnv
from rl.defense_learning import game_rank
from rl.defense_snapshot import capture


class DefenseLossCurriculumTests(unittest.TestCase):
    def test_exact_own_states_at_all_four_losses_and_unchanged_gameplay(self):
        actions = np.random.default_rng(12).integers(20, size=3000)
        ordinary = DefenseEnv(max_steps=0)
        expected = []
        try:
            initial = ordinary.reset(12)
            for action in actions:
                result = ordinary.step(int(action))
                expected.append(result)
                if result[2]:
                    break
            self.assertTrue(expected[-1][2])
        finally:
            ordinary.close()
        # Both cell encodings must key the actual earlier snapshot, not the
        # visible post-loss state or its newly reset per-life score baseline.
        for cells in ('score', 'screen', 'age'):
            env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                worker_id=0, curriculum_trigger='life-loss', curriculum_lookback=64,
                curriculum_cells=cells)
            visited = deque(maxlen=65)
            losses = 0
            try:
                np.testing.assert_array_equal(env.reset(12), initial)
                for action, original in zip(actions, expected):
                    result = env.step(int(action))
                    np.testing.assert_array_equal(result[0], original[0])
                    self.assertEqual(result[1:4], original[1:4])
                    info = result[4]
                    self.assertEqual({k: info[k] for k in original[4]}, original[4])
                    if info['life_lost']:
                        losses += 1
                        saved, prior = info['_curriculum_snapshot'], visited[-64]
                        self.assertEqual(saved.native, prior.native)
                        np.testing.assert_array_equal(saved.frames, prior.frames)
                        self.assertEqual(saved.source_action, env.total_actions-64)
                        self.assertEqual(saved.lives, info['lives']+1)
                        self.assertEqual(saved.stage, 1)
                        self.assertEqual(saved.progress_start_score, prior.progress_start_score)
                        event = info['curriculum_archive_add']
                        self.assertEqual(event['entry_kind'], 'life_loss_lookback')
                        self.assertEqual(event['trigger_progress'], info['score']-prior.progress_start_score)
                        self.assertEqual(event['progress'], saved.score-saved.progress_start_score)
                        self.assertEqual(event['trigger_terminal'], result[2])
                        self.assertEqual(event['trigger_life_steps']-event['source_life_steps'], 64)
                        self.assertFalse(env.history)
                        visited.clear()
                    else:
                        self.assertNotIn('curriculum_archive_add', info)
                        if not result[2]:
                            visited.append(capture(env))
                    self.assertLessEqual(len(env.history), 65)
                self.assertEqual(losses, 4)
                self.assertTrue(env.archive)
                env.reset(99)
                self.assertFalse(env.history)
                self.assertTrue(env.full_game)
            finally:
                env.close()

    def test_short_history_is_not_borrowed_from_previous_life_or_reset(self):
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_trigger='life-loss',
                                   curriculum_lookback=64)
        try:
            env.reset(12)
            # Clear only the host history after each native step. The game is
            # unmodified; even an actual loss must not invent a missing source.
            losses = 0
            for _ in range(3000):
                _, _, done, _, info = env.step(0)
                self.assertNotIn('curriculum_archive_add', info)
                losses += int(info['life_lost'])
                env.history.clear()
                if done:
                    break
            self.assertEqual(losses, 4)
            self.assertFalse(env.archive)
        finally:
            env.close()

    def test_peer_restores_exact_state_and_only_new_score_counts(self):
        producer = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
            worker_id=0, curriculum_trigger='life-loss', curriculum_lookback=64)
        try:
            producer.reset(12)
            for _ in range(3000):
                info = producer.step(0)[4]
                if '_curriculum_snapshot' in info:
                    saved = info['_curriculum_snapshot']
                    break
            else:
                self.fail('No actual visible loss')
        finally:
            producer.close()
        peer = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
            worker_id=1, curriculum_reset=False, curriculum_trigger='life-loss', curriculum_lookback=64)
        try:
            peer.reset(99)
            before = capture(peer)
            peer.receive_archive([saved])
            self.assertEqual(capture(peer).native, before.native)
            with patch('rl.defense_curriculum.restore', side_effect=AssertionError('boot worker')):
                peer.reset()
            self.assertTrue(peer.full_game)
            peer.curriculum_reset = True
            obs = peer.reset()
            self.assertEqual(capture(peer).native, saved.native)
            np.testing.assert_array_equal(obs, saved.frames)
            self.assertEqual(peer.segment_source_worker, 0)
            self.assertEqual(peer.segment_source_action, saved.source_action)
            self.assertEqual(peer.life_steps, saved.life_steps)
            self.assertFalse(peer.history)
            reward_sum = 0
            for _ in range(3000):
                _, reward, done, _, info = peer.step(0)
                reward_sum += reward
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(reward_sum, peer.score-saved.score)
            self.assertEqual(info['episode_reward'], reward_sum)
            self.assertIsNone(game_rank(info))
        finally:
            peer.close()

    def test_stage_entry_is_immediate_and_does_not_reuse_old_life_history(self):
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
            worker_id=0, curriculum_trigger='life-loss', curriculum_lookback=8)
        try:
            env.reset(12)
            for _ in range(10):
                env.step(0)
            original = DefenseEnv.step
            def bookkeeping_transition(current, action):
                result = original(current, action)
                # Host bookkeeping fixture only, never replayed or trained on.
                current.stage = current.highest_stage = 2
                return result
            with patch.object(DefenseEnv, 'step', bookkeeping_transition):
                info = env.step(0)[4]
            saved = info['_curriculum_snapshot']
            self.assertEqual(saved.native, capture(env).native)
            self.assertEqual(saved.source_action, env.total_actions)
            self.assertEqual(info['curriculum_archive_add']['entry_kind'], 'stage_entry')
            self.assertEqual(len(env.history), 1)
        finally:
            env.close()

    def test_disabled_mode_never_captures_and_invalid_settings_fail(self):
        env = DefenseCurriculumEnv(curriculum_probability=0,
            curriculum_trigger='life-loss', curriculum_lookback=64)
        try:
            with patch('rl.defense_curriculum.capture', side_effect=AssertionError('disabled')):
                env.reset(12)
                for _ in range(100):
                    env.step(0)
            self.assertFalse(env.history)
            self.assertFalse(env.archive)
        finally:
            env.close()
        for kwargs in (dict(curriculum_trigger='invalid'),
                       dict(curriculum_trigger='life-loss', curriculum_lookback=0)):
            with self.assertRaises(ValueError):
                DefenseCurriculumEnv(**kwargs)
        from rl import defense_dqn
        for extra in (['--curriculum-trigger', 'invalid'],
                      ['--curriculum-trigger', 'life-loss'],
                      ['--curriculum-trigger', 'life-loss', '--curriculum-lookback', '64'],
                      ['--curriculum-trigger', 'life-loss', '--curriculum-probability', '.5']):
            with patch.object(sys, 'argv', ['dqn', '--run', '/nonexistent/loss-trigger',
                      '--artifacts', '/nonexistent/loss-artifacts']+extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_truncation_without_visible_loss_does_not_trigger_archive(self):
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_trigger='life-loss',
                                   curriculum_lookback=4, max_steps=8, observation_stride=3)
        try:
            env.reset(12)
            for _ in range(8):
                _, _, terminal, truncated, info = env.step(0)
                self.assertFalse(info['life_lost'])
                self.assertNotIn('curriculum_archive_add', info)
            self.assertFalse(terminal)
            self.assertTrue(truncated)
            self.assertFalse(env.archive)
            self.assertFalse(env.history)
        finally:
            env.close()

    def test_native_dqn_routing_learning_and_resume_without_saved_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            args = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/'run'),
                '--artifacts', str(p/'artifacts'), '--envs', '2', '--capacity', '4096',
                '--compact-replay', '--warmup', '32', '--batch-size', '4', '--train-every', '256',
                '--steps', '4096', '--eval-every', '10000', '--max-episode-steps', '512',
                '--mlx-cache-mb', '64', '--exploration-max-repeat', '64', '--epsilon-final', '1',
                '--curriculum-probability', '1', '--curriculum-share', '--curriculum-boot-envs', '1',
                '--curriculum-trigger', 'life-loss', '--curriculum-lookback', '16']
            result = subprocess.run(args, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = [json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            archives = [r for r in rows if r['event'] == 'curriculum_archive']
            self.assertTrue(archives)
            self.assertTrue(all(r['entry_kind'] == 'life_loss_lookback' for r in archives))
            self.assertTrue(all(r['trigger_action']-r['source_action'] == 16 for r in archives))
            self.assertTrue(all(r['source_lives'] == r['trigger_lives']+1 for r in archives))
            self.assertTrue(all(r['shared_with'] == 1 for r in archives))
            episodes = [r for r in rows if r['event'] == 'episode']
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker'] == 0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker'] == 1))
            runtime = next(r['workers'] for r in rows if r['event'] == 'workers_started')
            self.assertTrue(all(not w['mlx_loaded'] for w in runtime))
            state = json.loads((p/'run/latest/state.json').read_text())
            self.assertGreater(state['updates'], 0)
            self.assertEqual(state['config']['curriculum_trigger'], 'life-loss')
            self.assertFalse(state['config']['curriculum_archive_saved'])
            result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/'resume'),
                '--artifacts', str(p/'resume-artifacts'), '--resume', str(p/'run/latest'),
                '--steps', '4112', '--warmup', '4', '--batch-size', '4', '--train-every', '4'],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            after = json.loads((p/'resume/latest/state.json').read_text())
            self.assertEqual(after['config']['curriculum_trigger'], 'life-loss')
            self.assertEqual(after['config']['curriculum_lookback'], 16)
            self.assertGreater(after['updates'], state['updates'])
            self.assertEqual(after['restored_segments'], state['restored_segments'])
            self.assertEqual({f.name for f in (p/'resume/latest').iterdir()},
                {'model.safetensors', 'target.safetensors', 'optimizer.npz', 'state.json'})


if __name__ == '__main__':
    unittest.main()
