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
from mlx.utils import tree_flatten
import numpy as np

from rl.defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_initialization import initialize_policy
from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace
from rl.defense_recurrent import RecurrentPPO, ResidualRecurrentNetwork
from rl.ppo import PPO
from rl.recurrent_policy import RECURRENT_ARCHITECTURE, RecurrentPolicy, sequence_batches


def config():
    return dict(game='defense', algorithm='ppo', game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION, action_names=list(action_names()),
                tstates=100000, observation_stride=1, eval_max_steps=0)


class RecurrentTests(unittest.TestCase):
    def test_sequence_packing_never_interleaves_workers(self):
        x = np.arange(8*3*2).reshape(8, 3, 2)
        y = sequence_batches(x, 4)
        self.assertEqual(y.shape, (6, 4, 2))
        for chunk in range(2):
            for worker in range(3):
                np.testing.assert_array_equal(y[chunk*3+worker], x[chunk*4:chunk*4+4, worker])
        for length in [0, True, 3, 1.5]:
            with self.assertRaises(ValueError):sequence_batches(x, length)

    def test_zero_residual_and_sequence_resets_match_step_execution(self):
        mx.random.seed(4)
        model = ResidualRecurrentNetwork(hidden_size=8)
        obs = np.random.default_rng(4).integers(128, 192, (2, 4, 4, 16, 64), dtype=np.uint8)
        flat = mx.array(obs.reshape(-1, 4, 16, 64))
        baseline = model.base.policy_value(flat)
        current = model.step(flat, mx.ones((8, 8)))
        for a, b in zip(baseline, current[:2], strict=True):
            np.testing.assert_array_equal(np.array(a), np.array(b))
        model.memory_actor.weight = mx.ones_like(model.memory_actor.weight)*.1
        model.memory_value.weight = mx.ones_like(model.memory_value.weight)*.2
        initial = mx.ones((2, 8))
        starts = np.array([[False, False, True, False], [True, False, False, True]])
        logits, values, final = model.sequence(mx.array(obs), initial, mx.array(starts))
        h = initial; ls=[];vs=[]
        for t in range(4):
            h = h*(1-mx.array(starts[:, t, None]))
            l,v,h = model.step(mx.array(obs[:, t]), h);ls.append(l);vs.append(v)
        np.testing.assert_allclose(np.array(logits), np.array(mx.stack(ls, axis=1)), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.array(values), np.array(mx.stack(vs, axis=1)), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(np.array(final), np.array(h), atol=2e-6, rtol=2e-6)

    def test_parallel_policy_memory_follows_game_identity_and_resets(self):
        def infer(obs, hidden):
            next_hidden = hidden+np.asarray(obs)[:, :1]
            logits = np.concatenate([next_hidden, -next_hidden, next_hidden*.2], axis=1)
            return logits, next_hidden
        parallel = RecurrentPolicy(infer, 1)
        rngs = [np.random.default_rng(s) for s in [4,5,6]]
        serial = [RecurrentPolicy(infer, 1, seed=s) for s in [4,5,6]]
        for ids, numbers in [([0,1],[1.,2.]), ([1,0],[3.,4.]), ([2,1],[1.,-1.]), ([0],[2.])]:
            obs = np.asarray(numbers, np.float32)[:,None]
            actual = parallel.sample_with_rngs(obs, [rngs[i] for i in ids])
            expected = [serial[i](obs[j:j+1])[0] for j,i in enumerate(ids)]
            np.testing.assert_array_equal(actual, expected)
            for i in ids:
                np.testing.assert_array_equal(parallel.memories[rngs[i]], serial[i].serial_memory[0])
        with self.assertRaises(ValueError):parallel.sample_with_rngs(np.ones((2,1)),[rngs[0]]*2)
        parallel.reset_seed(4)
        self.assertFalse(parallel.memories)
        self.assertIsNone(parallel.serial_memory)
        np.testing.assert_array_equal(parallel(np.ones((1,1))), RecurrentPolicy(infer,1,seed=4)(np.ones((1,1))))
        with self.assertRaises(ValueError):parallel(np.ones((2,1)))

    def test_update_learns_memory_and_bootstrap_does_not_consume_history(self):
        agent = RecurrentPPO(seed=8, hidden_size=8)
        rng=np.random.default_rng(5)
        obs=rng.integers(128,192,(2,4,16,64),dtype=np.uint8)
        agent.reset_memory(2)
        agent.act(obs,rng)
        before=agent.memory.copy();agent.bootstrap_value(obs);agent.bootstrap_value(obs[:1],[0])
        np.testing.assert_array_equal(agent.memory,before)
        evaluation_policy=agent.policy(seed=7)
        evaluation_policy(obs)
        evaluation_policy.reset_seed(8)
        evaluation_policy(obs)
        np.testing.assert_array_equal(agent.memory,before)
        agent.reset_done(np.array([True,False]))
        np.testing.assert_array_equal(agent.memory[0],np.zeros(8))
        np.testing.assert_array_equal(agent.memory[1],before[1])
        with self.assertRaises(ValueError):agent.reset_done(np.array([1,0]))
        sequence=np.tile(obs[:,None],(1,4,1,1,1))
        initial=np.zeros((2,8),np.float32);starts=np.zeros((2,4),bool)
        logits,values,_=agent.model.sequence(mx.array(sequence),mx.array(initial),mx.array(starts))
        lp=np.array(logits-mx.logsumexp(logits,axis=-1,keepdims=True))
        data=(sequence,np.zeros((2,4),np.int32),lp[:,:,0],np.ones((2,4),np.float32),
              np.array(values)+1.,initial,starts)
        original=np.array(agent.model.memory.Wx)
        for _ in range(2):
            loss,aux=agent.update(*(mx.array(a) for a in data));mx.eval(loss,aux,agent.state)
            self.assertTrue(np.isfinite(loss.item()))
        self.assertTrue(np.any(np.array(agent.model.memory_actor.weight)!=0))
        self.assertFalse(np.array_equal(original,np.array(agent.model.memory.Wx)))

    def test_disabled_memory_has_no_output_or_gradient_contribution(self):
        import mlx.nn as nn
        model=ResidualRecurrentNetwork(hidden_size=8,memory_scale=0.)
        model.memory_actor.weight=mx.ones_like(model.memory_actor.weight)
        model.memory_value.weight=mx.ones_like(model.memory_value.weight)
        obs=mx.array(np.random.default_rng(9).integers(128,192,(2,4,16,64),dtype=np.uint8))
        logits,values,_=model.step(obs,mx.ones((2,8)))
        expected=model.base.policy_value(obs)
        for a,b in zip((logits,values),expected,strict=True):
            np.testing.assert_array_equal(np.array(a),np.array(b))
        def loss(network):
            logits,values,_=network.step(obs,mx.ones((2,8)))
            return mx.mean(logits)+mx.mean(values)
        _,gradients=nn.value_and_grad(model,loss)(model)
        memory_gradients=[g for name,g in tree_flatten(gradients) if name.startswith('memory')]
        self.assertTrue(memory_gradients)
        for gradient in memory_gradients:
            np.testing.assert_array_equal(np.array(gradient),np.zeros(gradient.shape))

    def test_full_initialization_loader_parallel_games_and_verified_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=PPO(seed=4,action_count=20)
            base.save(root/'base',dict(config=config(),steps=99))
            agent=RecurrentPPO(seed=7,hidden_size=8)
            metadata=initialize_policy(agent.model,root/'base')
            self.assertEqual(metadata['source_training_steps'],99)
            self.assertFalse(metadata['trajectories_loaded'])
            for (name,a),(other,b) in zip(tree_flatten(agent.model.base.parameters()),tree_flatten(base.model.parameters()),strict=True):
                self.assertEqual(name,other);np.testing.assert_array_equal(np.array(a),np.array(b))
            with self.assertRaises(ValueError):initialize_policy(agent.model,root/'base',tstates=50000)
            cfg=dict(config(),architecture=RECURRENT_ARCHITECTURE,recurrent_hidden=8,memory_scale=1.)
            # Exercise nonzero learned residuals in the replay/parallel checks.
            agent.model.memory_actor.weight=mx.random.normal(agent.model.memory_actor.weight.shape)*.2
            agent.model.memory_actor.bias=mx.arange(20,dtype=mx.float32)*.04
            agent.compile();agent.save(root/'rnn',dict(config=cfg,steps=0))
            policy,_=load_policy(root/'rnn/model.safetensors')
            result=evaluate(policy,[10000,10001],envs=2,max_steps=0)
            self.assertEqual(result['complete_games'],2)
            for game in result['games']:
                single,_=load_policy(root/'rnn/model.safetensors')
                trace=record_game(single,game['seed'],tstates=100000,max_steps=0)
                self.assertEqual(trace[3],game)
            verification=verify_policy_trace(root/'rnn/model.safetensors',*trace[:4])
            self.assertTrue(verification['verified'])
            self.assertEqual(verification['verified_actions'],len(trace[1]))

    def test_cli_invalid_modes_and_real_train_resume(self):
        from rl import defense_train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for extra in [['--recurrent-hidden=-1'],['--recurrent-hidden=8','--sequence-length=3'],
                          ['--recurrent-hidden=8','--sil-updates=1'],['--recurrent-hidden=8','--policy-bias-noise=1'],
                          ['--memory-scale=0'],['--initialize-policy=missing']]:
                with patch.object(sys,'argv',['train','--run',str(root/'invalid'),*extra]), \
                        contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                    defense_train.main()
                self.assertEqual(error.exception.code,2)
            base=[sys.executable,'-m','rl.defense_train','--envs','2','--rollout','8','--batch-size','16',
                  '--epochs','1','--recurrent-hidden','8','--sequence-length','4','--mlx-cache-mb','128',
                  '--max-episode-steps','3','--curriculum-probability','.5','--curriculum-share',
                  '--curriculum-boot-envs','1']
            def run(name,extra):
                result=subprocess.run(base+['--run',str(root/name),'--artifacts',str(root/(name+'-artifacts'))]+extra,
                                      capture_output=True,text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            first=run('first',['--steps','32'])
            self.assertEqual(first['steps'],32)
            self.assertEqual(first['config']['architecture'],RECURRENT_ARCHITECTURE)
            clone=run('clone',['--resume',str(root/'first/latest'),'--steps','32'])
            for f in ['model.safetensors','optimizer.npz']:
                a,b=[mx.load(str(root/p/'latest'/f)) for p in ['first','clone']]
                for k in a:np.testing.assert_array_equal(np.array(a[k]),np.array(b[k]))
            later=run('later',['--resume',str(root/'first/latest'),'--steps','64'])
            self.assertEqual(later['steps'],64)
            self.assertGreater(later['episodes'],first['episodes'])
            self.assertEqual(first['rng'],clone['rng'])
            rows=[json.loads(l) for l in (root/'first/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(not w['mlx_loaded'] for r in rows if r['event']=='workers_started' for w in r['workers']))


if __name__=='__main__':
    unittest.main()
