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

from rl.defense_inverse import ARCHITECTURE, AUX_FILES, InverseLearner
from rl.defense_inverse_replay import InverseReplay
from rl.defense_trace_replay import TraceNStep
from rl.model import Learner
from rl.replay import NStep, Replay


def screens(seed, shape=(4, 16, 64)):
    return np.random.default_rng(seed).integers(128, 192, shape, dtype=np.uint8)


def arrays(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


class InverseTests(unittest.TestCase):
    def test_exact_td_fields_sampler_and_immediate_not_nstep_endpoint(self):
        base, extra = Replay(8, compact=True), InverseReplay(8, 3)
        first, second = NStep(base, 3, .997), TraceNStep(extra, 3, .997)
        for i in range(30):
            transition = screens(i), i % 20, i/7, screens(i+1), i % 7 == 6, i % 11 == 10
            first.append(*transition); second.append(*transition)
            if not base.size: continue
            r1, r2 = np.random.default_rng(i), np.random.default_rng(i)
            ia, a = base.sample(4, r1, .7); ib, b = extra.sample(4, r2, .7)
            np.testing.assert_array_equal(ia, ib)
            self.assertEqual(r1.bit_generator.state, r2.bit_generator.state)
            for x, y in zip(a, b[:6], strict=True): np.testing.assert_array_equal(x, y)
            for root, following in zip(b[0], b[6]):
                matches = [j for j in range(i+1) if np.array_equal(root, screens(j))]
                self.assertEqual(len(matches), 1)
                np.testing.assert_array_equal(following, screens(matches[0]+1))
            base.priorities(ia, np.arange(4)); extra.priorities(ib, np.arange(4))
        self.assertEqual(len(first.queue), len(second.queue))

    def test_auxiliary_gradients_and_td_only_priorities(self):
        agent = InverseLearner(seed=13)
        obs, immediate = mx.array(screens(1, (2,4,16,64))), mx.array(screens(2, (2,4,16,64)))
        actions = mx.array([1, 4])
        def loss(model): return agent.auxiliary_loss(model, obs, actions, immediate)[0].sum()
        value, grads = nn.value_and_grad(agent.training_model, loss)(agent.training_model)
        flat = {k: np.array(v) for k,v in tree_flatten(grads)}
        self.assertTrue(np.isfinite(value.item()))
        for name in ('q.conv.0.weight','q.hidden.weight','aux.hidden.weight','aux.output.weight'):
            self.assertGreater(np.abs(flat[name]).sum(), 0)
        for name in ('q.value.weight','q.value.bias','q.advantage.weight','q.advantage.bias'):
            np.testing.assert_array_equal(flat[name], 0)
        batch=(obs,actions,mx.array([.1,.2]),immediate,mx.array([.99,0.]),mx.array([.3,1.]))
        td, (errors, q)=Learner._loss(agent,agent.online,*batch)
        combined, (new_errors,new_q,inverse,accuracy)=agent._joint_loss(agent.training_model,*batch,immediate)
        np.testing.assert_array_equal(np.array(errors),np.array(new_errors))
        self.assertEqual(q.item(),new_q.item())
        self.assertAlmostEqual(combined.item(),(td+agent.weight*inverse).item(),places=6)
        self.assertTrue(0 <= accuracy.item() <= 1)

    def test_joint_update_and_unchanged_acting_path(self):
        agent=InverseLearner(seed=3);replay=InverseReplay(8,3)
        replay.add_sequence([(screens(i),i,.1,screens(i+1),False) for i in range(3)],.997)
        _,batch=replay.sample(2,np.random.default_rng(4),.4)
        target=arrays(agent.target);before=arrays(agent.aux)
        loss,errors,q=agent.train(batch)
        self.assertTrue(np.isfinite([loss,q]).all() and np.isfinite(errors).all())
        self.assertEqual(agent.inverse_stats()['own_pairs'],2)
        self.assertTrue(any(not np.array_equal(v,before[k]) for k,v in arrays(agent.aux).items()))
        for k,v in arrays(agent.target).items():np.testing.assert_array_equal(v,target[k])
        auxiliary=arrays(agent.aux);agent.sync_target()
        for k,v in arrays(agent.target).items():np.testing.assert_array_equal(v,arrays(agent.online)[k])
        for k,v in arrays(agent.aux).items():np.testing.assert_array_equal(v,auxiliary[k])
        obs=screens(77,(2,4,16,64));expected=np.array(mx.argmax(agent.online(mx.array(obs)),axis=1))
        with patch.object(type(agent.aux),'__call__',side_effect=AssertionError('auxiliary acting')):
            np.testing.assert_array_equal(agent.actions(obs),expected)

    def test_invalid_cli_and_checkpoint(self):
        from rl import defense_dqn
        for extra in (['--inverse-weight','-1'],['--inverse-weight','nan'],['--inverse-weight','11'],
                      ['--inverse-weight','.01'],['--inverse-weight','.01','--compact-replay','--n-step','33'],
                      ['--inverse-weight','.01','--compact-replay','--spr-weight','.1'],
                      ['--inverse-weight','.01','--compact-replay','--greedy-trace-cut'],
                      ['--inverse-weight','.01','--compact-replay','--quantiles','32'],
                      ['--inverse-weight','.01','--compact-replay','--bootstrap-heads','5']):
            with patch.object(sys,'argv',['dqn','--run','runs/invalid-inverse','--artifacts','runs/invalid-inverse-artifacts']+extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                defense_dqn.main()
            self.assertEqual(result.exception.code,2)
        agent=InverseLearner()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):agent.restore_auxiliary(tmp,{})
            hashes=agent.save_auxiliary(tmp)
            saved=dict(config=dict(inverse_architecture=ARCHITECTURE),inverse_auxiliary_hashes=hashes)
            agent.restore_auxiliary(tmp,saved)
            with self.assertRaises(ValueError):agent.restore_auxiliary(tmp,dict(saved,config={}))
            (Path(tmp)/AUX_FILES[0]).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):agent.restore_auxiliary(tmp,saved)

    def test_native_parity_conversion_resume_and_verified_game(self):
        from rl.defense_learning import evaluate,load_policy,publish_best
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def run(name,extra):
                result=subprocess.run([sys.executable,'-m','rl.defense_dqn','--run',str(root/name),
                    '--artifacts',str(root/(name+'-artifacts'))]+extra,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            common=['--envs','2','--capacity','32','--compact-replay','--warmup','4','--batch-size','4',
                '--n-step','2','--train-every','4','--target-every','2','--steps','64','--eval-every','1000',
                '--max-episode-steps','3','--mlx-cache-mb','64','--exploration-max-repeat','64',
                '--epsilon-final','.9','--curriculum-probability','.5','--curriculum-share',
                '--curriculum-boot-envs','1','--curriculum-boot-epsilon','.05','--curriculum-lookback','4']
            default=run('default',common);disabled=run('disabled',common+['--inverse-weight','0'])
            converted=run('converted',['--resume',str(root/'default/latest'),'--steps','64','--inverse-weight','.01'])
            for other in (disabled,converted):
                for key in ('steps','updates','episodes','rng','persistent_exploration_rng'):
                    self.assertEqual(default[key],other[key])
            self.assertEqual(converted['inverse']['local_updates'],0)
            enabled=run('enabled',common+['--inverse-weight','.01'])
            self.assertEqual(enabled['inverse']['local_updates'],enabled['updates'])
            unchanged=run('unchanged',['--resume',str(root/'enabled/latest'),'--steps','64'])
            self.assertEqual(unchanged['inverse']['local_updates'],0)
            first=run('resumed-a',['--resume',str(root/'enabled/latest'),'--steps','96'])
            second=run('resumed-b',['--resume',str(root/'enabled/latest'),'--steps','96'])
            self.assertGreater(first['updates'],enabled['updates'])
            for left,right,extra in [('default','disabled',()),('default','converted',()),
                                      ('enabled','unchanged',AUX_FILES),('resumed-a','resumed-b',AUX_FILES)]:
                for name in ('model.safetensors','target.safetensors','optimizer.npz',*extra):
                    a=mx.load(str(root/left/'latest'/name));b=mx.load(str(root/right/'latest'/name))
                    self.assertEqual(set(a),set(b))
                    for key in a:np.testing.assert_array_equal(np.array(a[key]),np.array(b[key]))
            for key in ('steps','updates','episodes','rng','persistent_exploration_rng'):
                self.assertEqual(first[key],second[key])
            policy,_=load_policy(root/'enabled/latest/model.safetensors')
            evaluation=evaluate(policy,[18123],envs=1)
            self.assertEqual(evaluation['complete_games'],1)
            self.assertIsNotNone(publish_best(root/'enabled/latest/model.safetensors',evaluation,root/'verified'))
            verification=json.loads((root/'verified/best/verification.json').read_text())
            self.assertTrue(verification['verified'])
            self.assertEqual(verification['verified_actions'],evaluation['games'][0]['steps'])
            rows=[json.loads(l) for l in (root/'enabled/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(not w['mlx_loaded'] for w in next(r['workers'] for r in rows if r['event']=='workers_started')))

    def test_native_actual_own_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            command=[sys.executable,'-m','rl.defense_dqn','--run',str(root/'run'),'--artifacts',str(root/'artifacts'),
                '--envs','2','--capacity','4096','--compact-replay','--warmup','32','--batch-size','4',
                '--train-every','256','--steps','4096','--eval-every','10000','--max-episode-steps','512',
                '--mlx-cache-mb','64','--n-step','5','--inverse-weight','.01','--exploration-max-repeat','64',
                '--epsilon-final','1','--curriculum-probability','1','--curriculum-share',
                '--curriculum-boot-envs','1','--curriculum-boot-epsilon','.05','--curriculum-lookback','4']
            result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(l) for l in (root/'run/metrics.jsonl').read_text().splitlines()]
            archives=[r for r in rows if r['event']=='curriculum_archive'];episodes=[r for r in rows if r['event']=='episode']
            self.assertTrue(archives)
            self.assertTrue(all(r['trigger_action']-r['source_action']==4 for r in archives))
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker']==0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker']==1))
            state=json.loads((root/'run/latest/state.json').read_text())
            self.assertGreater(state['inverse']['own_pairs'],0)
            self.assertGreater(state['restored_segments'],0)


if __name__ == '__main__':
    unittest.main()
