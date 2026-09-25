import contextlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl import defense_dqn
from rl.replay import Replay


class DefenseDQNCurriculumTests(unittest.TestCase):
    def test_invalid_reset_configuration_is_rejected_before_starting(self):
        base=['dqn','--run','/nonexistent/dqn-curriculum','--artifacts','/nonexistent/artifacts']
        for extra in [['--curriculum-probability','nan'],['--curriculum-probability','1.1'],
                      ['--curriculum-lookback','-1'],['--curriculum-share'],
                      ['--curriculum-bins','0'],['--curriculum-per-bin','0'],
                      ['--curriculum-score-interval','-1'],['--curriculum-screen-interval','0'],
                      ['--curriculum-age-interval','0'],['--curriculum-frontier-bins','-1'],
                      ['--curriculum-frontier-bins','2'],
                      ['--curriculum-probability','.5','--curriculum-frontier-bins','2'],
                      ['--curriculum-probability','.5','--curriculum-cells','age',
                       '--curriculum-bins','1','--curriculum-frontier-bins','2'],
                      ['--curriculum-cells','invalid'],
                      ['--curriculum-boot-envs','1'],
                      ['--curriculum-probability','.5','--curriculum-share','--curriculum-boot-envs','8'],
                      ['--bootstrap-heads','5','--curriculum-probability','.5']]:
            with self.subTest(extra=extra),patch.object(sys,'argv',base+extra), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code,2)

    def test_replay_uses_new_rewards_terminal_screens_and_separate_segment_counts(self):
        # Bookkeeping fixtures only, never evidence of actual game progression.
        replays=[];worker_configs=[]
        class RecordingReplay(Replay):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);replays.append(self)
        class Workers:
            def __init__(self,count,seed,**kwargs):
                worker_configs.append(kwargs);self.turn=0
                self.observations=np.zeros((2,4,16,64),np.uint8)
            def runtime(self):return []
            def close(self):pass
            def step(self,actions):
                self.turn+=1;terminal=self.turn==2;rows=[]
                for i in range(2):
                    reward=10*(2*(self.turn-1)+i+1)
                    info=dict(life_lost=terminal,full_game=i==0,episode_reward=40+20*i,
                              score=40 if i==0 else 1060,highest_stage=1,missions_completed=0,
                              terminated=terminal,truncated=False)
                    if self.turn==1:
                        info['curriculum_archive_add']=dict(source_action=1,source_full_game=True)
                    rows.append((np.full((4,16,64),9 if terminal else 1,np.uint8),
                                 reward,terminal,False,info,
                                 np.full((4,16,64),99,np.uint8) if terminal else None))
                return rows
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            args=['dqn','--run',str(p/'run'),'--artifacts',str(p/'artifacts'),
                  '--envs','2','--capacity','32','--warmup','16','--steps','4',
                  '--eval-every','4','--mlx-cache-mb','64',
                  '--curriculum-probability','.5','--curriculum-share',
                  '--curriculum-boot-envs','1','--curriculum-lookback','4']
            fake_eval=dict(complete_games=0,incomplete_games=0,games=[])
            with patch.object(sys,'argv',args),patch.object(defense_dqn,'VectorEnv',Workers), \
                    patch.object(defense_dqn,'Replay',RecordingReplay), \
                    patch.object(defense_dqn,'evaluate',return_value=fake_eval) as evaluate, \
                    patch.object(defense_dqn,'publish_best'), \
                    patch.object(defense_dqn.time,'monotonic',side_effect=itertools.count(0,20)), \
                    contextlib.redirect_stdout(io.StringIO()):
                defense_dqn.main()
            replay=replays[0]
            np.testing.assert_allclose(replay.returns[:4],[.1+.997*.3,.3,.2+.997*.4,.4],rtol=1e-6)
            np.testing.assert_array_equal(replay.discounts[:4],0)
            np.testing.assert_array_equal(replay.next_obs[:4],9) # Not reset frame 99.
            self.assertTrue(worker_configs[0]['curriculum'])
            self.assertEqual(worker_configs[0]['curriculum_boot_envs'],1)
            self.assertEqual(worker_configs[0]['curriculum_cells'],'score')
            self.assertEqual(worker_configs[0]['curriculum_bins'],16)
            self.assertEqual(worker_configs[0]['curriculum_per_bin'],4)
            self.assertFalse(any('curriculum' in key for key in evaluate.call_args.kwargs))
            state=json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual((state['episodes'],state['boot_episodes'],state['restored_segments']),(2,1,1))
            rows=[json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['event']=='curriculum_archive' for r in rows),2)
            progress=[r for r in rows if r['event']=='progress'][-1]
            self.assertEqual(progress['recent']['mean_score'],40)
            self.assertEqual(progress['recent_restored'],dict(segments=1,mean_new_score=60))

    def test_real_own_archives_reserved_boot_worker_and_resume_refill(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            args=[sys.executable,'-m','rl.defense_dqn','--run',str(p/'run'),
                  '--artifacts',str(p/'artifacts'),'--envs','2','--capacity','4096',
                  '--warmup','4096','--steps','4096','--eval-every','10000',
                  '--max-episode-steps','512','--mlx-cache-mb','64',
                  '--curriculum-probability','1','--curriculum-share',
                  '--curriculum-boot-envs','1','--curriculum-lookback','4']
            result=subprocess.run(args,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            archives=[r for r in rows if r['event']=='curriculum_archive']
            self.assertTrue(archives)
            self.assertTrue(all(r['trigger_action']-r['source_action']==4 for r in archives))
            episodes=[r for r in rows if r['event']=='episode']
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker']==0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker']==1))
            workers=next(r['workers'] for r in rows if r['event']=='workers_started')
            self.assertEqual([w['curriculum_reset_enabled'] for w in workers],[False,True])
            self.assertTrue(all(not w['mlx_loaded'] for w in workers))
            state=json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['boot_episodes']+state['restored_segments'],state['episodes'])
            self.assertFalse(state['config']['curriculum_archive_saved'])
            resumed=subprocess.run([sys.executable,'-m','rl.defense_dqn','--run',str(p/'resume'),
                                    '--artifacts',str(p/'resume-artifacts'),'--resume',str(p/'run/latest'),
                                    '--steps','4112','--warmup','4','--train-every','4','--batch-size','4'],
                                   capture_output=True,text=True,timeout=60)
            self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            after=json.loads((p/'resume/latest/state.json').read_text())
            self.assertGreater(after['updates'],state['updates'])
            for key in ['curriculum_probability','curriculum_share','curriculum_boot_envs','curriculum_lookback']:
                self.assertEqual(after['config'][key],state['config'][key])
            self.assertEqual(after['restored_segments'],state['restored_segments'])
            self.assertEqual({f.name for f in (p/'resume/latest').iterdir()},
                             {'model.safetensors','target.safetensors','optimizer.npz','state.json'})

    def test_persistent_screen_archive_controls_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            args=[sys.executable,'-m','rl.defense_dqn','--run',str(p/'run'),
                  '--artifacts',str(p/'artifacts'),'--envs','2','--capacity','4096',
                  '--compact-replay','--warmup','32','--batch-size','4','--train-every','256',
                  '--steps','4096','--eval-every','10000','--max-episode-steps','512',
                  '--mlx-cache-mb','64','--exploration-max-repeat','64','--epsilon-final','1',
                  '--curriculum-probability','1','--curriculum-share','--curriculum-boot-envs','1',
                  '--curriculum-lookback','4','--curriculum-cells','screen',
                  '--curriculum-bins','8','--curriculum-per-bin','1',
                  '--curriculum-screen-interval','8']
            result=subprocess.run(args,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            archives=[r for r in rows if r['event']=='curriculum_archive']
            self.assertTrue(archives)
            self.assertTrue(all(r['entry_kind']=='screen_cell' for r in archives))
            self.assertTrue(all(len(r['screen_cell'])==32 for r in archives))
            self.assertTrue(all(r['trigger_action']-r['source_action']==4 for r in archives))
            episodes=[r for r in rows if r['event']=='episode']
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker']==0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker']==1))
            for episode in episodes:
                counts=episode['curriculum_archive_counts']
                self.assertLessEqual(len(counts),8)
                self.assertTrue(all(v==1 for v in counts.values()))
            state=json.loads((p/'run/latest/state.json').read_text())
            self.assertGreater(state['updates'],0)
            self.assertGreater(state['persistent_exploration']['exploratory_steps'],0)
            self.assertEqual(state['config']['curriculum_cell_encoding'],'graphics-9x16x8-v1')
            self.assertFalse(state['config']['curriculum_archive_saved'])
            resumed=subprocess.run([sys.executable,'-m','rl.defense_dqn','--run',str(p/'resume'),
                                    '--artifacts',str(p/'resume-artifacts'),'--resume',str(p/'run/latest'),
                                    '--steps','4112','--warmup','4','--train-every','4','--batch-size','4'],
                                   capture_output=True,text=True,timeout=90)
            self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            after=json.loads((p/'resume/latest/state.json').read_text())
            self.assertGreater(after['updates'],state['updates'])
            for key in ['curriculum_cells','curriculum_bins','curriculum_per_bin',
                        'curriculum_score_interval','curriculum_screen_interval']:
                self.assertEqual(after['config'][key],state['config'][key])
            self.assertEqual(after['restored_segments'],state['restored_segments'])

    def test_fine_cadence_age_frontier_integrates_with_score_only_dqn(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            args=[sys.executable,'-m','rl.defense_dqn','--run',str(p/'run'),
                  '--artifacts',str(p/'artifacts'),'--envs','2','--capacity','4096',
                  '--compact-replay','--warmup','32','--batch-size','4','--train-every','256',
                  '--steps','4096','--eval-every','10000','--max-episode-steps','512',
                  '--mlx-cache-mb','64','--exploration-max-repeat','64','--epsilon-final','1',
                  '--tstates','50000','--observation-stride','2',
                  '--curriculum-probability','1','--curriculum-share','--curriculum-boot-envs','1',
                  '--curriculum-lookback','8','--curriculum-cells','age',
                  '--curriculum-bins','8','--curriculum-per-bin','1',
                  '--curriculum-age-interval','16','--curriculum-frontier-bins','2']
            result=subprocess.run(args,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(x) for x in (p/'run/metrics.jsonl').read_text().splitlines()]
            archives=[row for row in rows if row['event']=='curriculum_archive']
            self.assertTrue(archives)
            self.assertTrue(all(row['entry_kind']=='life_age' for row in archives))
            self.assertTrue(all(row['lookback_actions']==8 for row in archives))
            self.assertTrue(all(row['life_age_bin']==row['life_steps']//16 for row in archives))
            state=json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['steps'],4096)
            self.assertGreater(state['updates'],0)
            self.assertEqual(state['config']['curriculum_cells'],'age')
            self.assertEqual(state['config']['curriculum_frontier_bins'],2)
            self.assertEqual(state['config']['tstates'],50000)
            self.assertEqual(state['config']['observation_stride'],2)
            self.assertEqual(state['config']['reward'],
                             'visible score difference only, constant scale for optimizer')
            self.assertFalse(state['config']['curriculum_archive_saved'])


if __name__=='__main__':unittest.main()
