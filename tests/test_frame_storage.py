from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_bootstrap_replay import BootstrapReplay
from rl.frame_storage import FrameStorage
from rl.replay import NStep, Replay


def frame(number):
    image=np.zeros((16,64),np.uint8)
    image.flat[:8]=np.frombuffer(int(number).to_bytes(8,'little'),np.uint8)
    return image


class FrameStorageTests(unittest.TestCase):
    def audit(self, storage):
        pool=storage.pool
        ids=np.concatenate([storage.obs.indices.ravel(),storage.next_obs.indices.ravel()])
        counts=Counter(int(i) for i in ids if i>=0)
        self.assertEqual(counts,{i:n for i,n in enumerate(pool.references) if n})
        self.assertEqual(len(pool.free),len(set(pool.free)))
        self.assertEqual(set(pool.free),{i for i,n in enumerate(pool.references) if not n})
        self.assertEqual(pool.lookup,{f:i for i,f in enumerate(pool.frames) if f is not None})
        self.assertLessEqual(len(pool.frames),8*storage.capacity+8)

    def test_ring_releases_repeated_frames_and_samples_never_alias(self):
        storage=FrameStorage(3);expected={}
        for step in range(1000):
            slot=step%3
            obs=np.stack([frame(step),frame(step),frame(step+1),frame(step+2)])
            following=np.stack([frame(step+1),frame(step+2),frame(step+6),frame(step+6)])
            storage.obs[slot]=obs;storage.next_obs[slot]=following
            expected[slot]=(obs.copy(),following.copy())
            obs[:]=255;following[:]=255
            for i,(a,b) in expected.items():
                np.testing.assert_array_equal(storage.obs[i],a)
                np.testing.assert_array_equal(storage.next_obs[i],b)
            sample=storage.obs[slot];sample[:]=77
            np.testing.assert_array_equal(storage.obs[slot],expected[slot][0])
            self.audit(storage)

    def test_invalid_assignment_preserves_live_references(self):
        for size in [0,-1,True,2.5,2**31]:
            with self.assertRaises((ValueError,TypeError)):FrameStorage(size)
        storage=FrameStorage(2)
        obs=np.stack([frame(1)]*4);storage.obs[0]=obs
        before=(dict(storage.pool.lookup),list(storage.pool.references))
        for bad in [obs.astype(np.float32),obs[:3],obs[0]]:
            with self.assertRaises(ValueError):storage.obs[0]=bad
        with self.assertRaises(IndexError):storage.obs[2]=obs
        with self.assertRaises(IndexError):storage.obs[1]
        self.assertEqual(before,(storage.pool.lookup,storage.pool.references))
        np.testing.assert_array_equal(storage.obs[0],obs)
        self.audit(storage)

    def test_priorities_nstep_masks_and_rng_match_dense_after_many_wraps(self):
        for heads in [0,3]:
            make=lambda compact: (BootstrapReplay(17,heads,.5,np.random.default_rng(11),compact=compact)
                                  if heads else Replay(17,compact=compact))
            dense,compact=make(False),make(True)
            buffers=[[NStep(r,n=5,gamma=.99) for _ in range(3)] for r in [dense,compact]]
            sampling=[np.random.default_rng(99),np.random.default_rng(99)]
            rng=np.random.default_rng(5)
            for step in range(400):
                worker=step%3
                # Arbitrary frame histories and variable next-state distances
                # deliberately break any adjacency/reconstruction assumption.
                obs=np.stack([frame(int(i)) for i in rng.integers(30,size=4)])
                nxt=np.stack([frame(int(i)) for i in rng.integers(30,70,size=4)])
                terminal=step%19==0;truncated=step%31==0
                for bank in buffers:bank[worker].append(obs,step%20,float(step%7),nxt,terminal,truncated)
                if dense.size and step%7==0:
                    left=dense.sample(13,sampling[0],.7);right=compact.sample(13,sampling[1],.7)
                    np.testing.assert_array_equal(left[0],right[0])
                    for a,b in zip(left[1],right[1],strict=True):np.testing.assert_array_equal(a,b)
                    errors=rng.random(13)
                    dense.priorities(left[0],errors);compact.priorities(right[0],errors)
                    np.testing.assert_array_equal(dense.tree,compact.tree)
                    self.assertEqual(sampling[0].bit_generator.state,sampling[1].bit_generator.state)
                    self.audit(compact.frame_storage)

    def test_actual_spaced_own_state_resets_have_exact_dense_batches(self):
        from rl.defense_curriculum import DefenseCurriculumEnv
        env=DefenseCurriculumEnv(seed=17,max_steps=128,observation_stride=2,
                                 curriculum_probability=1,curriculum_lookback=4)
        dense,compact=Replay(97),Replay(97,compact=True)
        a,b=NStep(dense,5,.997),NStep(compact,5,.997)
        rng=np.random.default_rng(9);restores=0
        try:
            obs=env.reset(17)
            for _ in range(768):
                action=int(rng.integers(20));following,reward,done,truncated,info=env.step(action)
                for nstep in [a,b]:nstep.append(obs,action,reward*.01,following,done or info['life_lost'],truncated)
                obs=env.reset() if done or truncated else following
                if done or truncated:restores+=not env.full_game
            self.assertGreater(restores,0)
            for x,y in zip(dense.sample(64,np.random.default_rng(1),1)[1],
                           compact.sample(64,np.random.default_rng(1),1)[1],strict=True):
                np.testing.assert_array_equal(x,y)
            self.audit(compact.frame_storage)
        finally:env.close()

    def test_temporal_frame_reuse_reduces_payload_with_bounded_slots(self):
        replay=Replay(512,compact=True)
        for step in range(1200):
            obs=np.stack([frame(step+i) for i in range(4)])
            nxt=np.stack([frame(step+5+i) for i in range(4)])
            replay.add(obs,0,0,nxt,.99)
        stats=replay.frame_storage.stats()
        self.assertLess(stats['screen_payload_bytes']+stats['screen_index_bytes'],
                        stats['dense_equivalent_screen_bytes']/6)
        self.audit(replay.frame_storage)

    def test_real_dense_compact_training_and_resume_are_parameter_exact(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for heads in [0,2]:
                directories=[]
                for compact in [False,True]:
                    p=root/f'h{heads}-compact{compact}';directories.append(p)
                    args=[sys.executable,'-m','rl.defense_dqn','--run',str(p),
                          '--artifacts',str(p)+'-artifacts','--envs','2','--capacity','32',
                          '--warmup','8','--steps','256','--eval-every','1000',
                          '--max-episode-steps','64','--batch-size','4','--n-step','2',
                          '--train-every','4','--target-every','8','--mlx-cache-mb','64']
                    args+=['--bootstrap-heads',str(heads)] if heads else [
                        '--curriculum-probability','1','--curriculum-share',
                        '--curriculum-boot-envs','1','--curriculum-lookback','4']
                    if compact:args+=['--compact-replay']
                    result=subprocess.run(args,capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                states=[json.loads((p/'latest/state.json').read_text()) for p in directories]
                for key in ['steps','episodes','boot_episodes','restored_segments','updates','rng']:
                    self.assertEqual(states[0][key],states[1][key])
                if heads:self.assertEqual(states[0]['bootstrap_rng'],states[1]['bootstrap_rng'])
                for filename in ['model.safetensors','target.safetensors','optimizer.npz']:
                    a,b=[mx.load(str(p/'latest'/filename)) for p in directories]
                    self.assertEqual(set(a),set(b))
                    for key in a:np.testing.assert_array_equal(np.array(a[key]),np.array(b[key]))
                self.assertTrue(states[1]['config']['compact_replay'])
                resumed=root/f'resumed-{heads}'
                result=subprocess.run([sys.executable,'-m','rl.defense_dqn','--run',str(resumed),
                                       '--artifacts',str(resumed)+'-artifacts',
                                       '--resume',str(directories[1]/'latest'),'--steps','272'],
                                      capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                state=json.loads((resumed/'latest/state.json').read_text())
                self.assertTrue(state['config']['compact_replay'])
                self.assertGreater(state['updates'],states[1]['updates'])


if __name__=='__main__':unittest.main()
