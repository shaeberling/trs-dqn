import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl import defense_temporal_probe
from rl.defense import DefenseEnv
from rl.ppo import categorical_policy
from rl.temporal_probe import TemporalPolicy


class DefenseTemporalProbeTests(unittest.TestCase):
    def test_rejects_fresh_seeds_and_existing_output_before_loading(self):
        for extra in [['--seed', '20000'], ['--games', '11'], ['--envs', '0']]:
            args = ['probe', '/nonexistent/model.safetensors', '--stride', '2',
                    '--output', '/nonexistent/out.json', *extra]
            with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_temporal_probe.main()
            self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'out.json'; p.write_text('{}')
            with patch.object(sys, 'argv', ['probe', '/nonexistent/model.safetensors',
                                           '--stride', '2', '--output', str(p)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                defense_temporal_probe.main()
            self.assertEqual(error.exception.code, 2)

    def test_real_defense_history_and_sampling_use_only_spaced_visible_frames(self):
        received = []
        def logits(obs):
            received.append(obs.copy())
            return np.zeros((len(obs), 20), np.float32)
        wrapped = TemporalPolicy(categorical_policy(logits), stride=2)
        rng = np.random.default_rng(10000)
        expected_rng = np.random.default_rng(10000)
        env = DefenseEnv(tstates=50_000, max_steps=0)
        try:
            obs = env.reset(10000)
            history = [obs[-1].copy()]
            for step in range(12):
                action = wrapped.sample_with_rngs(obs[None], [rng])
                # Same one categorical draw per action; history never chooses actions.
                expected = int((expected_rng.random() > np.cumsum(np.full(20, .05))).sum())
                self.assertEqual(int(action[0]), expected)
                wanted = np.stack([history[max(0, step+offset)] for offset in [-6, -4, -2, 0]])
                np.testing.assert_array_equal(received[-1][0], wanted)
                obs, _, done, truncated, _ = env.step(int(action[0]))
                self.assertFalse(done or truncated)
                history.append(obs[-1].copy())
        finally:
            env.close()

    def test_probe_is_uncapped_and_cannot_promote_or_change_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); checkpoint=p/'model.safetensors'; checkpoint.write_bytes(b'frozen')
            cfg=dict(algorithm='ppo', game_sha256='game', environment_version='version',tstates=100000)
            (p/'state.json').write_text(json.dumps(dict(config=cfg)))
            policy=categorical_policy(lambda obs: np.zeros((len(obs),20),np.float32))
            evaluation=dict(complete_games=10,incomplete_games=0,mean_score=280,median_score=280,
                            best_score=280,highest_stage=1,mission_games=0,games=[])
            args=['probe',str(checkpoint),'--stride','2','--output',str(p/'out.json')]
            with patch.object(sys,'argv',args), patch.object(defense_temporal_probe,'load_policy',return_value=(policy,cfg)), \
                    patch.object(defense_temporal_probe,'evaluate',return_value=evaluation) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()):
                defense_temporal_probe.main()
            self.assertEqual(evaluate.call_args.kwargs['max_steps'],0)
            self.assertEqual(list(evaluate.call_args.args[1]),list(range(10000,10010)))
            saved=json.loads((p/'out.json').read_text())
            self.assertTrue(saved['evaluation_only']);self.assertFalse(saved['promotion_eligible'])
            self.assertEqual(saved['nominal_history_span_tstates'],300000)
            self.assertEqual(checkpoint.read_bytes(),b'frozen')


if __name__ == '__main__':
    unittest.main()
