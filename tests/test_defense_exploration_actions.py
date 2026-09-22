import contextlib
import copy
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
from rl.defense_exploration_actions import action_probabilities
from rl.persistent_exploration import PersistentExploration


class DefenseExplorationActionsTests(unittest.TestCase):
    def test_distribution_group_masses_and_invalid_inputs(self):
        self.assertIsNone(action_probabilities('uniform', 20))
        p = action_probabilities('stage1-balanced', 20)
        self.assertTrue(np.all(p > 0))
        self.assertAlmostEqual(p.sum(), 1.)
        np.testing.assert_allclose(p[:9], np.full(9, 1/12))
        np.testing.assert_allclose(p[9:18], np.full(9, 1/108))
        np.testing.assert_allclose(p[18:], [1/12]*2)
        self.assertAlmostEqual(p[9:18].sum(), 1/12)
        for mode,count in [('bad',20),('stage1-balanced',21),('stage1-balanced',6)]:
            with self.assertRaises(ValueError): action_probabilities(mode,count)
        for probabilities in ([.05]*19, [0.]*20, [.1]*20, [np.nan]*20, [-.05]+[1.05/19]*19):
            with self.assertRaises(ValueError):
                PersistentExploration(2,20,64,1.5,np.random.default_rng(9),probabilities)

    def test_default_exact_rng_and_balanced_empirical_sampling(self):
        a = PersistentExploration(8,20,64,1.5,np.random.default_rng(7))
        b = PersistentExploration(8,20,64,1.5,np.random.default_rng(7),action_probabilities('uniform',20))
        for step in range(400):
            np.testing.assert_array_equal(a.select(np.arange(8),.9),b.select(np.arange(8),.9))
            if step%17==0:
                boundary=np.arange(8)%2==0;a.reset(boundary);b.reset(boundary)
        self.assertEqual(a.stats(),b.stats())
        self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
        p=action_probabilities('stage1-balanced',20)
        c=PersistentExploration(100000,20,1,1.5,np.random.default_rng(9),p)
        p[:]=0  # The explorer owns its validated copy.
        actions=c.select(np.zeros(100000,np.int32),1.)
        observed=np.bincount(actions,minlength=20)/len(actions)
        np.testing.assert_allclose(observed,action_probabilities('stage1-balanced',20),atol=.0025)
        self.assertTrue(np.all(np.bincount(actions,minlength=20)>0))

    def test_greedy_choices_boundaries_and_rng_restore(self):
        p=action_probabilities('stage1-balanced',20)
        a=PersistentExploration(8,20,64,1.5,np.random.default_rng(7),p)
        greedy=np.arange(8)
        for _ in range(20): np.testing.assert_array_equal(a.select(greedy,0.),greedy)
        a.select(greedy,1.)
        a.reset(np.ones(8,dtype=bool))
        np.testing.assert_array_equal(a.select(greedy,0.),greedy)
        saved=copy.deepcopy(a.rng.bit_generator.state)
        b=PersistentExploration(8,20,64,1.5,np.random.default_rng(),p)
        b.rng.bit_generator.state=saved
        for _ in range(100):
            np.testing.assert_array_equal(a.select(greedy,.9),b.select(greedy,.9))
        self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)

    def test_real_stage_one_forward_fire_aliases_preserve_whole_visible_game(self):
        # Random diagnostic actions only, never demonstrations or training data.
        actions=np.random.default_rng(12).integers(20,size=3000)
        reference=[]
        for alias in [None,9,17]:
            env=DefenseEnv()
            try:
                initial=env.reset(12)
                if alias is None: initial_reference=initial.copy()
                else: np.testing.assert_array_equal(initial,initial_reference)
                for i,action in enumerate(actions):
                    chosen=int(alias if alias is not None and 9<=action<=17 else action)
                    result=env.step(chosen)
                    self.assertEqual(result[4]['highest_stage'],1)
                    if alias is None: reference.append(result)
                    else:
                        np.testing.assert_array_equal(result[0],reference[i][0])
                        self.assertEqual(result[1:],reference[i][1:])
                    if result[2]: break
                self.assertTrue(result[2]);self.assertFalse(result[3])
                self.assertEqual(i+1,len(reference))
            finally:env.close()

    def test_invalid_cli_before_launch(self):
        from rl import defense_dqn
        common=['dqn','--run','/nonexistent/balanced','--artifacts','/nonexistent/balanced-artifacts',
                '--exploration-actions','stage1-balanced']
        for extra in [[],['--exploration-max-repeat','64','--allow-enter']]:
            with patch.object(sys,'argv',common+extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code,2)

    def test_native_learning_resume_and_original_policy_loader(self):
        from rl.defense_learning import load_policy
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def run(name,extra):
                args=[sys.executable,'-m','rl.defense_dqn','--run',str(p/name),
                      '--artifacts',str(p/(name+'-artifacts'))]+extra
                result=subprocess.run(args,capture_output=True,text=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return json.loads((p/name/'latest/state.json').read_text())
            a=run('first',['--envs','2','--capacity','64','--batch-size','4','--warmup','4',
                '--steps','64','--eval-every','1000','--max-episode-steps','3','--train-every','4',
                '--target-every','2','--mlx-cache-mb','64','--epsilon-final','1',
                '--exploration-max-repeat','64','--exploration-actions','stage1-balanced'])
            self.assertGreater(a['updates'],0)
            np.testing.assert_allclose(a['config']['exploration_action_probabilities'],action_probabilities('stage1-balanced',20))
            policy,config=load_policy(p/'first/latest/model.safetensors')
            self.assertEqual(config,a['config'])
            self.assertEqual(config['evaluation_policy'],'learned Q-values, greedy')
            self.assertEqual(a['persistent_exploration']['decisions'],58)
            b=run('resumed',['--resume',str(p/'first/latest'),'--steps','96'])
            self.assertEqual(b['config']['exploration_actions'],'stage1-balanced')
            self.assertGreater(b['updates'],a['updates'])
            self.assertEqual({f.name for f in (p/'resumed/latest').iterdir()},
                {'model.safetensors','target.safetensors','optimizer.npz','state.json'})


if __name__=='__main__':unittest.main()
