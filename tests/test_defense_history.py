from dataclasses import replace
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
from rl.defense_learning import evaluate, record_game, verify_policy_trace
from rl.defense_snapshot import capture, restore
from rl.ppo import categorical_policy
from rl.temporal_probe import TemporalPolicy
from rl.vector import VectorEnv


class DefenseHistoryTests(unittest.TestCase):
    def test_stride_two_is_exact_probe_history_without_changing_gameplay(self):
        actions = np.random.default_rng(91).integers(20, size=120)
        baseline = []
        env = DefenseEnv(tstates=50000, max_steps=0)
        try:
            start = env.reset(9)
            for action in actions:
                obs, reward, done, truncated, info = env.step(int(action))
                baseline.append((obs[-1].copy(), reward, done, truncated, info, capture(env).native))
        finally:
            env.close()
        received = []
        policy = TemporalPolicy(categorical_policy(lambda x: received.append(x.copy()) or
                                                   np.zeros((len(x),20),np.float32)), stride=2)
        rng = np.random.default_rng(17)
        env = DefenseEnv(tstates=50000, max_steps=0, observation_stride=2)
        try:
            obs = env.reset(9)
            np.testing.assert_array_equal(obs, start)
            raw = [start[-1]]
            for step, (action, row) in enumerate(zip(actions, baseline, strict=True)):
                # Feed the old four-adjacent-frame observation to the diagnostic.
                narrow = np.stack([raw[max(0,step+j)] for j in [-3,-2,-1,0]])
                policy.sample_with_rngs(narrow[None], [rng])
                np.testing.assert_array_equal(obs, received[-1][0])
                obs, reward, done, truncated, info = env.step(int(action))
                self.assertEqual((reward,done,truncated,info),row[1:5])
                np.testing.assert_array_equal(obs[-1],row[0])
                self.assertEqual(capture(env).native,row[5])
                self.assertEqual(len(env.frames),7)
                raw.append(row[0])
        finally:
            env.close()

    def test_snapshot_keeps_intermediate_frames_and_rejects_mismatches_atomically(self):
        env = DefenseEnv(tstates=50000,max_steps=0,observation_stride=2)
        actions=np.random.default_rng(4).integers(20,size=140)
        try:
            env.reset(11)
            for a in actions[:100]:env.step(int(a))
            saved=capture(env)
            self.assertEqual(saved.frames.shape,(7,16,64))
            self.assertEqual(saved.observation_stride,2)
            expected=[]
            for a in actions[100:]:
                obs,*rest=env.step(int(a));expected.append((obs.copy(),rest))
            np.testing.assert_array_equal(restore(env,saved),saved.frames[::2])
            for invalid in [replace(saved,observation_stride=1),replace(saved,observation_stride=2.),
                            replace(saved,observation_stride=True),replace(saved,frames=saved.frames[::2])]:
                with self.assertRaises(ValueError):restore(env,invalid)
                self.assertEqual(capture(env).native,saved.native)
                np.testing.assert_array_equal(capture(env).frames,saved.frames)
            for a,(want,rest) in zip(actions[100:],expected,strict=True):
                obs,*actual=env.step(int(a));np.testing.assert_array_equal(obs,want)
                self.assertEqual(actual,rest)
        finally:
            env.close()

    def test_all_archive_modes_share_exact_spaced_history_and_reject_other_stride(self):
        for mode in ['score','screen','age']:
            config=dict(tstates=50000,max_steps=0,observation_stride=2,curriculum_probability=1,
                        curriculum_share=True,curriculum_cells=mode,curriculum_lookback=4,
                        curriculum_screen_interval=4,curriculum_age_interval=8)
            env=DefenseCurriculumEnv(worker_id=0,**config)
            saved=None;seen={}
            try:
                env.reset(11)
                for a in np.random.default_rng(4).integers(20,size=320):
                    obs,_,done,truncated,info=env.step(int(a));seen[env.total_actions]=obs.copy()
                    if '_curriculum_snapshot' in info:
                        saved=info['_curriculum_snapshot']
                        np.testing.assert_array_equal(saved.frames[::2],seen[saved.source_action])
                    if done or truncated:break
                self.assertIsNotNone(saved,mode)
            finally:
                env.close()
            peer=DefenseCurriculumEnv(worker_id=1,**config)
            try:
                peer.reset(19);before=capture(peer)
                for invalid in [replace(saved,observation_stride=1),replace(saved,frames=saved.frames[::2])]:
                    with self.assertRaises(ValueError):peer.receive_archive([invalid])
                    self.assertFalse(peer.archive)
                    self.assertEqual(capture(peer).native,before.native)
                peer.receive_archive([saved])
                self.assertEqual(capture(peer).native,before.native)
                np.testing.assert_array_equal(peer.reset(),saved.frames[::2])
                self.assertFalse(peer.full_game)
                self.assertEqual(peer.segment_start_score,saved.score)
                self.assertEqual(capture(peer).native,saved.native)
            finally:
                peer.close()

    def test_vector_spaced_history_matches_serial_and_workers_have_no_gpu(self):
        vector=VectorEnv(2,71,game='defense',tstates=50000,max_steps=0,observation_stride=2)
        try:
            self.assertTrue(all(not w['mlx_loaded'] for w in vector.runtime()))
            initial=vector.observations.copy();rows=[]
            for _ in range(12):rows.append(vector.step([1,9]))
        finally:vector.close()
        for worker in range(2):
            env=DefenseEnv(tstates=50000,max_steps=0,observation_stride=2)
            try:
                np.testing.assert_array_equal(env.reset(71+worker),initial[worker])
                for r in rows:
                    obs,*actual=env.step([1,9][worker]);np.testing.assert_array_equal(obs,r[worker][0])
                    self.assertEqual(tuple(actual),r[worker][1:5])
            finally:env.close()

    def test_cli_invalid_history_and_both_trainers_resume_the_saved_stride(self):
        from rl import defense_train,defense_dqn
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for module in [defense_train,defense_dqn]:
                with patch.object(sys,'argv',['train','--run',str(p/'bad'),'--artifacts',str(p/'artifacts'),
                                              '--observation-stride','0']), \
                        contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code,2)
            for kind in ['ppo','dqn']:
                module='rl.defense_train' if kind=='ppo' else 'rl.defense_dqn'
                args=[sys.executable,'-m',module,'--run',str(p/kind),'--artifacts',str(p/(kind+'-artifacts')),
                      '--envs','2','--batch-size','4','--steps','32','--eval-every','1000',
                      '--max-episode-steps','3','--mlx-cache-mb','64','--observation-stride','2']
                args+=['--rollout','8'] if kind=='ppo' else ['--capacity','32','--warmup','4','--train-every','4','--n-step','2']
                result=subprocess.run(args,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                s=json.loads((p/kind/'latest/state.json').read_text());self.assertEqual(s['config']['observation_stride'],2)
                result=subprocess.run([sys.executable,'-m',module,'--run',str(p/(kind+'-resume')),
                                       '--artifacts',str(p/(kind+'-resume-artifacts')),'--resume',str(p/kind/'latest'),
                                       '--steps','64'],capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                s=json.loads((p/(kind+'-resume')/'latest/state.json').read_text())
                self.assertEqual(s['steps'],64);self.assertEqual(s['config']['observation_stride'],2)

    def test_invalid_environment_history_and_verifier_forwards_saved_stride(self):
        for stride in [0,-1,True,2.,'2']:
            with self.assertRaises(ValueError):DefenseEnv(observation_stride=stride)
        cfg=dict(tstates=50000,eval_max_steps=0,observation_stride=2)
        frames=np.zeros((2,16,64),np.uint8);actions=np.zeros(1,np.uint8);rewards=np.zeros(1,np.float32)
        result=dict(seed=10000,terminated=True,truncated=False,missions_completed=0,highest_stage=1,score=0)
        with patch('rl.defense_learning.load_policy',return_value=(None,cfg)), \
                patch('rl.defense_learning.record_game',return_value=(frames,actions,rewards,result,[])) as record, \
                patch('rl.defense_learning.sha256',return_value='hash'):
            v=verify_policy_trace('fake',frames,actions,rewards,result)
        self.assertEqual(record.call_args.kwargs['observation_stride'],2)
        self.assertEqual(v['observation_stride'],2)


if __name__=='__main__':unittest.main()
