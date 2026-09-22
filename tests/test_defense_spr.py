import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_spr import AUX_FILES, SprLearner, cosine_distance
from rl.defense_spr_replay import SprReplay
from rl.defense_trace_replay import TraceNStep
from rl.replay import NStep, Replay


def arrays(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def screens(seed, shape):
    return np.random.default_rng(seed).integers(128, 192, shape, dtype=np.uint8)


class SprTests(unittest.TestCase):
    def test_replay_preserves_exact_td_fields_sampling_and_boundaries(self):
        base, extra = Replay(8, compact=True), SprReplay(8, 3)
        a, b = NStep(base, 3, .997), TraceNStep(extra, 3, .997)
        for i in range(30):
            obs, following = screens(i, (4, 16, 64)), screens(i+1, (4, 16, 64))
            args = (obs, i % 20, i/7, following, i % 7 == 6, i % 11 == 10)
            a.append(*args); b.append(*args)
            if not base.size: continue
            r1, r2 = np.random.default_rng(i), np.random.default_rng(i)
            ia, ba = base.sample(4, r1, .7); ib, bb = extra.sample(4, r2, .7)
            np.testing.assert_array_equal(ia, ib)
            for x, y in zip(ba, bb[:6], strict=True): np.testing.assert_array_equal(x, y)
            self.assertEqual(r1.bit_generator.state, r2.bit_generator.state)
            self.assertEqual(bb[6].shape, (4, 3))
            self.assertEqual(bb[7].shape, (4, 3, 4, 16, 64))
            for n in range(4):
                size = int((bb[6][n] >= 0).sum())
                np.testing.assert_array_equal(bb[3][n], bb[7][n, size-1])
            base.priorities(ia, np.arange(4)); extra.priorities(ib, np.arange(4))
        self.assertEqual(len(a.queue), len(b.queue))

    def test_auxiliary_gradients_masks_and_target_stop_gradient(self):
        agent = SprLearner(seed=13, horizon=3)
        obs = mx.array(screens(1, (2, 4, 16, 64)))
        actions = mx.array([[1, 2, 3], [4, -1, -1]])
        future = screens(2, (2, 3, 4, 16, 64))
        key = mx.random.key(9)
        def compute(data):
            def objective(model):
                return agent.auxiliary_loss(model, obs, actions, mx.array(data), key)[0].sum()
            return nn.value_and_grad(agent.training_model, objective)(agent.training_model)
        loss, grads = compute(future)
        flat = {k: np.array(v) for k, v in tree_flatten(grads)}
        self.assertTrue(np.isfinite(loss.item()))
        self.assertGreater(np.abs(flat['q.conv.0.weight']).sum(), 0)
        self.assertGreater(np.abs(flat['q.hidden.weight']).sum(), 0)
        self.assertGreater(np.abs(flat['aux.conv1.weight']).sum(), 0)
        self.assertGreater(np.abs(flat['aux.predictor.weight']).sum(), 0)
        for k in ('q.value.weight', 'q.value.bias', 'q.advantage.weight', 'q.advantage.bias'):
            np.testing.assert_array_equal(flat[k], 0)
        changed = future.copy(); changed[1, 1:] = 255
        second, other = compute(changed)
        self.assertEqual(loss.item(), second.item())
        for k, v in tree_flatten(other): np.testing.assert_array_equal(np.array(v), flat[k])
        gradient = mx.grad(lambda target: cosine_distance(mx.ones((2, 4)), target).sum())
        np.testing.assert_array_equal(np.array(gradient(mx.arange(8).reshape(2, 4).astype(mx.float32))), 0)

    def test_joint_update_ema_rng_and_unchanged_acting_path(self):
        agent = SprLearner(seed=3, horizon=3)
        replay = SprReplay(8, 3)
        replay.add_sequence([(screens(i, (4,16,64)), i, .1, screens(i+1, (4,16,64)), False)
                             for i in range(3)], .997)
        _, batch = replay.sample(2, np.random.default_rng(4), .4)
        target_before, ema_before = arrays(agent.target), arrays(agent.ema)
        key_before = np.array(agent.aux_rng['key'])
        loss, errors, q = agent.train(batch)
        self.assertTrue(np.isfinite([loss, q]).all() and np.isfinite(errors).all())
        self.assertEqual(agent.spr_stats()['valid_future_predictions'], 6)
        self.assertFalse(np.array_equal(key_before, np.array(agent.aux_rng['key'])))
        for k, value in arrays(agent.target).items(): np.testing.assert_array_equal(value, target_before[k])
        for k, value in arrays(agent.ema).items():
            np.testing.assert_allclose(value, .99*ema_before[k]+.01*arrays(agent.online)[k], atol=1e-7)
        ema_before_sync = arrays(agent.ema)
        agent.sync_target()
        for k, value in arrays(agent.target).items(): np.testing.assert_array_equal(value, arrays(agent.online)[k])
        for k, value in arrays(agent.ema).items(): np.testing.assert_array_equal(value, ema_before_sync[k])
        obs = screens(77, (2,4,16,64))
        expected = np.array(mx.argmax(agent.online(mx.array(obs)), axis=1))
        with patch.object(agent.aux, 'transition', side_effect=AssertionError('auxiliary acting')):
            np.testing.assert_array_equal(agent.actions(obs), expected)

    def test_invalid_cli_and_auxiliary_checkpoint(self):
        from rl import defense_dqn
        for extra in (['--spr-weight', '-1'], ['--spr-weight', 'nan'], ['--spr-weight', '11'],
                      ['--spr-weight', '.1'], ['--spr-weight', '.1', '--compact-replay', '--n-step', '6'],
                      ['--spr-weight', '.1', '--compact-replay', '--greedy-trace-cut'],
                      ['--spr-weight', '.1', '--compact-replay', '--quantiles', '32'],
                      ['--spr-weight', '.1', '--compact-replay', '--bootstrap-heads', '5']):
            with patch.object(sys, 'argv', ['dqn','--run','runs/invalid-spr',
                                           '--artifacts','runs/invalid-spr-artifacts']+extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                defense_dqn.main()
            self.assertEqual(result.exception.code, 2)
        agent = SprLearner()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): agent.restore_auxiliary(tmp, {})
            hashes = agent.save_auxiliary(tmp)
            (Path(tmp)/AUX_FILES[0]).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                agent.restore_auxiliary(tmp, {'spr_auxiliary_hashes': hashes})

    def test_native_default_conversion_resume_and_frozen_loader(self):
        from rl.defense_learning import evaluate, load_policy, policy_description, publish_best
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/name),
                    '--artifacts', str(root/(name+'-artifacts'))]+extra, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            common = ['--envs','2','--capacity','32','--compact-replay','--warmup','4',
                '--batch-size','4','--n-step','2','--train-every','4','--target-every','2',
                '--steps','64','--eval-every','1000','--max-episode-steps','3','--mlx-cache-mb','64',
                '--exploration-max-repeat','64','--epsilon-final','.9','--curriculum-probability','.5',
                '--curriculum-share','--curriculum-boot-envs','1','--curriculum-boot-epsilon','.05',
                '--curriculum-lookback','4']
            default = run('default', common)
            disabled = run('disabled', common+['--spr-weight','0'])
            for key in ('steps','updates','episodes','rng','persistent_exploration_rng'):
                self.assertEqual(default[key], disabled[key])
            converted = run('converted', ['--resume',str(root/'default/latest'),'--steps','64','--spr-weight','.1'])
            for key in ('steps','updates','episodes','rng','persistent_exploration_rng'):
                self.assertEqual(default[key], converted[key])
            self.assertEqual(converted['spr']['local_updates'], 0)
            parent = mx.load(str(root/'default/latest/model.safetensors'))
            ema = mx.load(str(root/'converted/latest/spr-ema.safetensors'))
            for key in parent: np.testing.assert_array_equal(np.array(parent[key]), np.array(ema[key]))
            enabled = run('enabled', common+['--spr-weight','.1'])
            self.assertEqual(enabled['spr']['local_updates'], enabled['updates'])
            unchanged = run('unchanged', ['--resume',str(root/'enabled/latest'),'--steps','64'])
            self.assertEqual(unchanged['spr']['local_updates'], 0)
            for key in ('steps','updates','episodes','rng','persistent_exploration_rng'):
                self.assertEqual(enabled[key], unchanged[key])
            first = run('resumed-a', ['--resume',str(root/'enabled/latest'),'--steps','96'])
            second = run('resumed-b', ['--resume',str(root/'enabled/latest'),'--steps','96'])
            self.assertGreater(first['updates'], enabled['updates'])
            self.assertEqual(first['spr']['local_updates'], first['updates']-enabled['updates'])
            for left,right,extra in [('default','disabled',()),('default','converted',()),
                                     ('enabled','unchanged',AUX_FILES),('resumed-a','resumed-b',AUX_FILES)]:
                for name in ('model.safetensors','target.safetensors','optimizer.npz',*extra):
                    a=mx.load(str(root/left/'latest'/name));b=mx.load(str(root/right/'latest'/name))
                    self.assertEqual(set(a),set(b))
                    for key in a: np.testing.assert_array_equal(np.array(a[key]),np.array(b[key]))
            policy, config = load_policy(root/'enabled/latest/model.safetensors')
            self.assertEqual(policy_description(config), 'learned Q-values, greedy')
            self.assertEqual(policy(screens(7,(2,4,16,64))).shape, (2,))
            # One complete native game tests frozen loader/publisher plumbing,
            # not model quality; this temporary trace never enters learning.
            result = evaluate(policy, [17123], envs=1)
            self.assertEqual(result['complete_games'], 1)
            published = publish_best(root/'enabled/latest/model.safetensors', result, root/'verified')
            self.assertIsNotNone(published)
            verification = json.loads((root/'verified/best/verification.json').read_text())
            self.assertTrue(verification['verified'])
            self.assertEqual(verification['verified_actions'], result['games'][0]['steps'])
            rows=[json.loads(l) for l in (root/'enabled/metrics.jsonl').read_text().splitlines()]
            workers=next(r['workers'] for r in rows if r['event']=='workers_started')
            self.assertTrue(all(not w['mlx_loaded'] for w in workers))

    def test_native_auxiliary_with_actual_own_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            command=[sys.executable,'-m','rl.defense_dqn','--run',str(root/'run'),
                '--artifacts',str(root/'artifacts'),'--envs','2','--capacity','4096','--compact-replay',
                '--warmup','32','--batch-size','4','--train-every','256','--steps','4096',
                '--eval-every','10000','--max-episode-steps','512','--mlx-cache-mb','64',
                '--n-step','5','--spr-weight','.1','--exploration-max-repeat','64','--epsilon-final','1',
                '--curriculum-probability','1','--curriculum-share','--curriculum-boot-envs','1',
                '--curriculum-boot-epsilon','.05','--curriculum-lookback','4']
            result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(l) for l in (root/'run/metrics.jsonl').read_text().splitlines()]
            archives=[r for r in rows if r['event']=='curriculum_archive']
            episodes=[r for r in rows if r['event']=='episode']
            self.assertTrue(archives)
            self.assertTrue(all(r['trigger_action']-r['source_action']==4 for r in archives))
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker']==0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker']==1))
            state=json.loads((root/'run/latest/state.json').read_text())
            self.assertGreater(state['spr']['valid_future_predictions'],0)
            self.assertGreater(state['restored_segments'],0)


if __name__ == '__main__':
    unittest.main()
