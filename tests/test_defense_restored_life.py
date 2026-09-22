import contextlib
import ctypes
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
from rl.replay import NStep, Replay
from rl.snapshot import _configure
from rl.vector import VectorEnv
from trs.native import wrapper


def native_bytes():
    # Opaque equality check only. Never decode or manufacture a native state.
    _configure()
    data = ctypes.create_string_buffer(wrapper.z80_snapshot_size())
    if not wrapper.z80_save_snapshot(data, len(data)):
        raise AssertionError('native snapshot failed')
    return data.raw


def own_sources(stride=1):
    env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
        worker_id=0, curriculum_trigger='life-loss', curriculum_lookback=16,
        observation_stride=stride)
    sources = []
    try:
        env.reset(12)
        for _ in range(3000):
            _, _, done, _, info = env.step(0)
            if '_curriculum_snapshot' in info:
                sources.append(info['_curriculum_snapshot'])
            if done:
                break
        assert done and [s.lives for s in sources] == [4, 3, 2, 1]
        return sources
    finally:
        env.close()


class DefenseRestoredLifeTests(unittest.TestCase):
    def test_boot_games_unchanged_with_cut_option_enabled(self):
        actions = np.random.default_rng(12).integers(20, size=3000)
        original = DefenseEnv()
        expected = []
        try:
            initial = original.reset(12)
            for action in actions:
                result = original.step(int(action)); expected.append(result)
                if result[2]:
                    break
            self.assertTrue(expected[-1][2])
        finally:
            original.close()
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_trigger='life-loss',
            curriculum_lookback=16, curriculum_restored_life_only=True)
        try:
            np.testing.assert_array_equal(env.reset(12), initial)
            losses = 0
            for action, reference in zip(actions, expected):
                result = env.step(int(action))
                np.testing.assert_array_equal(result[0], reference[0])
                self.assertEqual(result[1:4], reference[1:4])
                self.assertEqual({k: result[4][k] for k in reference[4]}, reference[4])
                self.assertTrue(result[4]['full_game'])
                self.assertNotIn('curriculum_life_cut', result[4])
                losses += int(result[4]['life_lost'])
            self.assertEqual(losses, 4)
        finally:
            env.close()

    def test_restored_prefix_native_state_rewards_and_td_targets_are_exact(self):
        for stride in (1, 3):
            saved = own_sources(stride)[0]
            variants = []
            for enabled in (False, True):
                env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
                    worker_id=1, curriculum_trigger='life-loss', curriculum_lookback=16,
                    curriculum_restored_life_only=enabled, observation_stride=stride)
                replay = Replay(1024)
                buffer = NStep(replay, n=5, gamma=.997)
                trajectory = []
                try:
                    env.reset(99); env.receive_archive([saved]); obs = env.reset()
                    self.assertFalse(env.full_game)
                    self.assertEqual(native_bytes(), saved.native)
                    for _ in range(1000):
                        following, reward, terminal, truncated, info = env.step(0)
                        buffer.append(obs, 0, reward*.01, following, terminal or info['life_lost'], truncated)
                        trajectory.append((following.copy(), reward, terminal, truncated, dict(info)))
                        obs = following
                        if info['life_lost']:
                            break
                    self.assertTrue(info['life_lost'])
                    self.assertFalse(terminal)
                    self.assertEqual(truncated, enabled)
                    self.assertEqual(env.done, enabled)
                    self.assertEqual(info.get('curriculum_life_cut', False), enabled)
                    self.assertFalse(info['game_over'])
                    self.assertEqual(info['lives'], saved.lives-1)
                    self.assertEqual(sum(row[1] for row in trajectory), env.score-saved.score)
                    self.assertIsNone(game_rank(info))
                    self.assertFalse(buffer.queue)
                    variants.append((trajectory, native_bytes(), replay))
                    if enabled:
                        with self.assertRaises(RuntimeError):
                            env.step(0)
                        # Explicit seeded boot still starts a whole game, not
                        # another prematurely truncated restored segment.
                        env.reset(99)
                        self.assertTrue(env.full_game)
                        self.assertFalse(env.done)
                finally:
                    env.close()
            baseline, focused = variants
            self.assertEqual(baseline[1], focused[1])
            self.assertEqual(len(baseline[0]), len(focused[0]))
            for a, b in zip(baseline[0], focused[0], strict=True):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:3], b[1:3])
                excluded = {'truncated', 'curriculum_life_cut', 'curriculum_archive_counts'}
                self.assertEqual({k:v for k,v in a[4].items() if k not in excluded and k!='_curriculum_snapshot'},
                                 {k:v for k,v in b[4].items() if k not in excluded and k!='_curriculum_snapshot'})
            a, b = baseline[2], focused[2]
            self.assertEqual(a.size, b.size)
            for name in ('obs', 'next_obs', 'actions', 'returns', 'discounts'):
                np.testing.assert_array_equal(getattr(a, name)[:a.size], getattr(b, name)[:b.size])

    def test_restored_last_life_remains_native_game_over_not_cutoff(self):
        saved = own_sources()[-1]
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
            worker_id=1, curriculum_trigger='life-loss', curriculum_lookback=16,
            curriculum_restored_life_only=True)
        try:
            env.reset(99); env.receive_archive([saved]); env.reset()
            for _ in range(1000):
                _, _, terminal, truncated, info = env.step(0)
                if terminal or truncated:
                    break
            self.assertTrue(terminal)
            self.assertFalse(truncated)
            self.assertTrue(info['game_over'])
            self.assertEqual(info['lives'], 0)
            self.assertNotIn('curriculum_life_cut', info)
            self.assertFalse(info['full_game'])
            self.assertIsNone(game_rank(info))
        finally:
            env.close()

    def test_vector_autoresets_cut_segments_and_preserves_actual_loss_screen(self):
        workers = VectorEnv(2, seed=12, game='defense', curriculum=True,
            curriculum_probability=1, curriculum_share=True, curriculum_boot_envs=1,
            curriculum_trigger='life-loss', curriculum_lookback=16,
            curriculum_restored_life_only=True)
        try:
            cuts = 0
            for _ in range(3000):
                results = workers.step([0, 0])
                self.assertTrue(results[0][4]['full_game'])
                self.assertFalse(results[0][3])
                self.assertNotIn('curriculum_life_cut', results[0][4])
                for obs, _, terminal, truncated, info, reset in results:
                    self.assertNotIn('_curriculum_snapshot', info)
                    if info.get('curriculum_life_cut'):
                        cuts += 1
                        self.assertFalse(terminal)
                        self.assertTrue(truncated)
                        self.assertTrue(info['life_lost'])
                        self.assertFalse(info['full_game'])
                        self.assertGreater(info['lives'], 0)
                        self.assertIsNotNone(reset)
                        self.assertFalse(np.array_equal(obs, reset))
                        self.assertIn('curriculum_archive_counts', info)
                if cuts >= 2:
                    break
            self.assertGreaterEqual(cuts, 2)
            self.assertTrue(all(not w['mlx_loaded'] for w in workers.runtime()))
        finally:
            workers.close()

    def test_invalid_options_fail_before_launch(self):
        with self.assertRaises(ValueError):
            DefenseCurriculumEnv(curriculum_restored_life_only='yes')
        from rl import defense_dqn
        for extra in (['--curriculum-restored-life-only'],
                      ['--curriculum-restored-life-only', '--curriculum-probability', '.5', '--no-life-terminal']):
            with patch.object(sys, 'argv', ['dqn', '--run', '/nonexistent/restored-life',
                    '--artifacts', '/nonexistent/restored-artifacts']+extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_stage_entry_without_life_loss_does_not_cut_restored_segment(self):
        saved = own_sources()[0]
        env = DefenseCurriculumEnv(curriculum_probability=1, curriculum_share=True,
            worker_id=1, curriculum_trigger='life-loss', curriculum_lookback=16,
            curriculum_restored_life_only=True)
        try:
            env.reset(99); env.receive_archive([saved]); env.reset()
            original = DefenseEnv.step
            def bookkeeping_transition(current, action):
                result = original(current, action)
                self.assertFalse(result[4]['life_lost'])
                # Host-only fixture, never restored or used as training data.
                current.stage = current.highest_stage = 2
                result[4].update(stage=2, highest_stage=2)
                return result
            with patch.object(DefenseEnv, 'step', bookkeeping_transition):
                _, _, terminal, truncated, info = env.step(0)
            self.assertFalse(terminal or truncated or env.done)
            self.assertFalse(info['full_game'])
            self.assertNotIn('curriculum_life_cut', info)
            self.assertEqual(info['curriculum_archive_add']['entry_kind'], 'stage_entry')
        finally:
            env.close()

    def test_native_learning_and_resume_count_cuts_as_restored_not_boot_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            args = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/'run'),
                '--artifacts', str(p/'artifacts'), '--envs', '2', '--capacity', '4096',
                '--compact-replay', '--warmup', '32', '--batch-size', '4', '--train-every', '256',
                '--steps', '4096', '--eval-every', '10000', '--mlx-cache-mb', '64',
                '--exploration-max-repeat', '64', '--epsilon-final', '1',
                '--curriculum-probability', '1', '--curriculum-share', '--curriculum-boot-envs', '1',
                '--curriculum-trigger', 'life-loss', '--curriculum-lookback', '16',
                '--curriculum-restored-life-only']
            result = subprocess.run(args, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = [json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            episodes = [r for r in rows if r['event']=='episode']
            cuts = [r for r in episodes if r.get('curriculum_life_cut')]
            self.assertTrue(cuts)
            self.assertTrue(all(not r['full_game'] and r['truncated'] and not r['terminated'] for r in cuts))
            self.assertTrue(all(r['full_game'] and r['terminated'] and not r['truncated'] for r in episodes if r['worker']==0))
            state = json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['boot_episodes'], sum(r['full_game'] for r in episodes))
            self.assertEqual(state['restored_segments'], sum(not r['full_game'] for r in episodes))
            self.assertGreater(state['updates'], 0)
            result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/'resume'),
                '--artifacts', str(p/'resume-artifacts'), '--resume', str(p/'run/latest'),
                '--steps', '4112', '--warmup', '4', '--batch-size', '4', '--train-every', '4'],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            after = json.loads((p/'resume/latest/state.json').read_text())
            self.assertTrue(after['config']['curriculum_restored_life_only'])
            self.assertTrue(after['config']['life_terminal'])
            self.assertEqual(after['restored_segments'], state['restored_segments'])
            self.assertGreater(after['updates'], state['updates'])
            self.assertEqual({f.name for f in (p/'resume/latest').iterdir()},
                {'model.safetensors', 'target.safetensors', 'optimizer.npz', 'state.json'})


if __name__ == '__main__':
    unittest.main()
