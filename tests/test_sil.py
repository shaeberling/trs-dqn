from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from rl.env import SHAPE
from rl.ppo import PPO
from rl.sil import SILReplay, TrainingSuffixes, SelfImitation, sil_terms


class SILTests(unittest.TestCase):
    def screen(self, value=129):
        return np.full(SHAPE,value,np.uint8)

    def test_terminal_returns_and_input_immutability(self):
        replay = SILReplay(8)
        collector = TrainingSuffixes(replay,1,gamma=.5)
        obs = self.screen()
        self.assertIsNone(collector.append(0,obs,0,1,False,False,True))
        obs.fill(130)
        collector.append(0,obs,1,2,False,False,True)
        event = collector.append(0,self.screen(131),2,3,True,False,True)
        self.assertEqual(replay.size,3)
        np.testing.assert_array_equal(replay.returns[:3],[2.75,3.5,3])
        np.testing.assert_array_equal(replay.obs[0],self.screen())
        np.testing.assert_array_equal(replay.actions[:3],[0,1,2])
        self.assertEqual(event['score_reward_sum'],6)
        self.assertTrue(event['committed'])
        self.assertEqual(collector.metrics()['committed_transitions'],{'from_boot':3,'restored':0})

    def test_suffix_bound_keeps_exact_returns_without_artificial_terminal(self):
        replay = SILReplay(8)
        collector = TrainingSuffixes(replay,1,gamma=.5,suffix_steps=2)
        for value in range(1,5):
            collector.append(0,self.screen(128+value),value, value,False,False,True)
        self.assertEqual(replay.size,0)
        self.assertEqual(collector.metrics()['pending_transitions'],2)
        self.assertEqual(collector.metrics()['pending_screen_bytes'],2*4096)
        event = collector.append(0,self.screen(133),5,5,True,False,True)
        np.testing.assert_array_equal(replay.returns[:2],[6.5,5])
        self.assertEqual(event['dropped_prefix_steps'],3)
        self.assertEqual(collector.metrics()['pending_transitions'],0)

    def test_truncation_discards_suffix_including_terminal_coincidence(self):
        for terminal in (False,True):
            replay = SILReplay(8)
            collector = TrainingSuffixes(replay,1)
            collector.append(0,self.screen(),0,7,False,False,True)
            event = collector.append(0,self.screen(),0,4,terminal,True,True)
            self.assertEqual(replay.size,0)
            self.assertFalse(event['committed'])
            self.assertEqual(collector.metrics()['discarded_truncated_segments'],1)
            self.assertEqual(collector.metrics()['pending_transitions'],0)

    def test_worker_life_and_restored_origin_boundaries(self):
        replay = SILReplay(16)
        collector = TrainingSuffixes(replay,2,gamma=1)
        collector.append(0,self.screen(),0,2,False,False,True)
        collector.append(1,self.screen(),0,3,False,False,False)
        collector.append(0,self.screen(),0,4,True,False,True)
        collector.append(1,self.screen(),0,5,True,False,False)
        collector.append(0,self.screen(),0,9,True,False,True)
        np.testing.assert_array_equal(replay.returns[:5],[6,4,8,5,9])
        np.testing.assert_array_equal(replay.from_boot[:5],[True,True,False,False,True])
        self.assertEqual(collector.metrics()['completed_learning_segments'],3)
        collector.append(0,self.screen(),0,1,False,False,True)
        with self.assertRaises(ValueError):
            collector.append(0,self.screen(),0,1,False,False,False)

    def test_restored_screen_score_is_metadata_not_reward(self):
        replay = SILReplay(2)
        collector = TrainingSuffixes(replay,1)
        obs = np.full(SHAPE,32,np.uint8)
        obs[:,0,6:11] = np.frombuffer(b'00286',np.uint8)
        event = collector.append(0,obs,1,4,True,False,False)
        self.assertEqual(event['initial_score'],286)
        self.assertEqual(event['score_reward_sum'],4)
        self.assertEqual(replay.returns[0],4)
        self.assertFalse(replay.from_boot[0])

    def test_replay_wrap_sampling_and_priorities(self):
        replay = SILReplay(3,alpha=1)
        for i in range(5):
            replay.add(self.screen(128+i),i,i,True)
        self.assertEqual(replay.size,3)
        np.testing.assert_array_equal(replay.returns,[3,4,2])
        replay.priorities(np.array([0,1,2]),np.array([100.,0.,0.]))
        indices,batch = replay.sample(1000,np.random.default_rng(7),.1)
        self.assertGreater(np.mean(indices==0),.99)
        self.assertTrue(np.all(indices<3))
        self.assertTrue(np.isfinite(batch[-1]).all())
        self.assertTrue(np.all((batch[-1]>0)&(batch[-1]<=1)))
        indices2,_ = replay.sample(1000,np.random.default_rng(7),.1)
        np.testing.assert_array_equal(indices,indices2)
        with self.assertRaises(ValueError):
            replay.priorities([0],[float('nan')])

    def test_actor_advantage_is_detached_and_critic_is_one_sided(self):
        logits = mx.log(mx.array([[.2,.8],[.2,.8]]))
        values, returns = mx.array([.5,2.]),mx.array([1.,1.])
        actions, weights = mx.array([0,0]),mx.ones(2)
        actor,critic,positive = sil_terms(logits,values,actions,returns,weights)
        self.assertAlmostEqual(float(actor.item()),-np.log(.2)*.5/2,places=6)
        self.assertAlmostEqual(float(critic.item()),.0625,places=6)
        np.testing.assert_array_equal(np.array(positive),[.5,0])
        actor_grad = mx.grad(lambda v:sil_terms(logits,v,actions,returns,weights)[0])(values)
        critic_grad = mx.grad(lambda v:sil_terms(logits,v,actions,returns,weights)[1])(values)
        np.testing.assert_array_equal(np.array(actor_grad),[0,0])
        np.testing.assert_array_equal(np.array(critic_grad),[-.25,0])

    def test_gradient_update_and_zero_batch_skip_adam_momentum(self):
        agent = PPO(seed=13,learning_rate=1e-4,entropy=.001)
        sil = SelfImitation(agent)
        obs = np.stack([self.screen()]*8)
        logits,values = agent.predict(mx.array(obs))
        before = np.array(mx.softmax(logits))[:,0].mean()
        batch = (obs,np.zeros(8,np.int32),np.array(values)+2,np.ones(8,np.float32))
        positive,result = sil.train(batch)
        self.assertTrue(result['applied'])
        self.assertEqual(sil.updates,1)
        self.assertTrue(np.all(positive>0))
        logits,_ = agent.predict(mx.array(obs))
        self.assertGreater(np.array(mx.softmax(logits))[:,0].mean(),before)
        snapshot = [(name,np.array(value).copy()) for name,value in tree_flatten(agent.state)]
        _,result = sil.train((obs,np.zeros(8,np.int32),np.full(8,-100,np.float32),np.ones(8,np.float32)))
        self.assertFalse(result['applied'])
        self.assertEqual(sil.updates,1)
        for (name,expected),(actual_name,actual) in zip(snapshot,tree_flatten(agent.state),strict=True):
            self.assertEqual(name,actual_name)
            np.testing.assert_array_equal(np.array(actual),expected)

    def test_invalid_cli_settings_fail_before_run_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'must-not-exist'
            for flags in (['--sil-updates','-1'],['--sil-capacity','0'],
                          ['--sil-suffix-steps','0'],['--sil-loss-weight','nan'],
                          ['--sil-priority-alpha','2'],['--sil-value-weight','-1']):
                result = subprocess.run([sys.executable,'-m','rl.ppo','--run',str(output),*flags],
                                        capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('invalid SIL settings',result.stderr)
                self.assertFalse(output.exists())

    def test_interleaved_ppo_sil_optimizer_resume(self):
        agent = PPO(seed=23,learning_rate=1e-4,entropy=.001)
        sil = SelfImitation(agent)
        obs = np.stack([self.screen()]*8)
        actions,logp,values = agent.act(obs,np.random.default_rng(4))
        ppo_batch = tuple(mx.array(x) for x in
            (obs,actions,logp,np.ones(8,np.float32),values+1))
        sil_batch = (obs,actions,values+2,np.ones(8,np.float32))
        loss,aux = agent.update(*ppo_batch)
        mx.eval(loss,aux,agent.state)
        sil.train(sil_batch)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent.save(root,{'config':{'algorithm':'ppo'}})
            resumed = PPO(seed=29,learning_rate=1e-4,entropy=.001)
            resumed.model.load_weights(str(root/'model.safetensors'))
            resumed.optimizer.state = tree_unflatten(list(mx.load(str(root/'optimizer.npz')).items()))
            resumed.compile()
            resumed_sil = SelfImitation(resumed)
            for learner,auxiliary in ((agent,sil),(resumed,resumed_sil)):
                loss,aux = learner.update(*ppo_batch)
                mx.eval(loss,aux,learner.state)
                auxiliary.train(sil_batch)
                loss,aux = learner.update(*ppo_batch)
                mx.eval(loss,aux,learner.state)
            expected,actual = dict(tree_flatten(agent.state)),dict(tree_flatten(resumed.state))
            self.assertEqual(expected.keys(),actual.keys())
            for name in expected:
                np.testing.assert_allclose(np.array(actual[name]),np.array(expected[name]),
                                           rtol=1e-6,atol=1e-7,err_msg=name)

    def test_cpu_only_import_and_invalid_collection(self):
        subprocess.run([sys.executable,'-c',"import sys; import rl.sil; assert 'mlx.core' not in sys.modules"],
                       check=True,capture_output=True,text=True,timeout=30)
        collector = TrainingSuffixes(SILReplay(2),1)
        with self.assertRaises(ValueError):
            collector.append(0,self.screen(),0,float('nan'),False,False,True)
        self.assertEqual(collector.metrics()['pending_transitions'],0)


if __name__ == '__main__':
    unittest.main()
