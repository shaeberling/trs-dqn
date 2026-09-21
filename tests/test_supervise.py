import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from rl.env import ENVIRONMENT_VERSION
from rl.outcome import OUTCOME_VERSION
from rl.evaluate import level_rank, summary
from rl.supervise import (EventTail, Supervisor, checked_suite, digest,
                          latest_event, learner_arguments, new_primary_record, write_status,
                          verify_replay)


CONFIG = dict(environment_version=ENVIRONMENT_VERSION, tstates=50000, observation_stride=2,
              eval_seed=10000, eval_games=20, eval_max_steps=200000, target_level=10,
              target_clears=1, eval_every=500000, envs=32, curriculum_boot_envs=16,
              curriculum_min_level=8)


def suite(seeds, level=8, reserves=None):
    games = [dict(seed=s, score=500, level=level, steps=100, terminated=True,
                  truncated=False, game_over=True, episode_reward=500.,
                  start_tstates=int(np.random.default_rng(s).integers(0, 200001))) for s in seeds]
    if reserves is not None:
        for game in games:
            won = level == 8 and reserves > 0
            game.update(outcome_version=OUTCOME_VERSION, reserve_balls_visible=reserves,
                        game_won=won, win_status='verified_win' if won else
                        'unverified_final_level' if level == 8 else 'loss')
    return {**summary(games), 'games': games, 'tstates': 50000, 'observation_stride': 2,
            'environment_version': ENVIRONMENT_VERSION, 'max_steps': 200000,
            'epsilon': 0, 'policy': 'learned categorical, sampled'}


class FakeProcess:
    pid = 123456789
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0


class SupervisorTests(unittest.TestCase):
    def test_progress_override_is_only_explicit_learner_change(self):
        self.assertEqual(learner_arguments('run', 'source', 16),
                         ['--run', 'run', '--resume', 'source', '--steps', 0,
                          '--curriculum-score-interval', 16])
        self.assertNotIn('--curriculum-score-interval', learner_arguments('run', 'source'))

    def test_best_effort_is_recorded_without_selecting_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self.setup_supervisor(root)
            supervisor.root.mkdir()
            supervisor.effort_artifacts = root/'effort'
            prior = supervisor.selected.copy()
            with patch.object(supervisor, 'job') as jobs, patch('rl.supervise.publish_record') as publish:
                supervisor.consider_effort(supervisor.resume, suite(range(10000, 10020)))
                self.assertEqual(publish.call_args.args[2], 'single_game_best_effort')
                self.assertEqual(jobs.call_count, 2)
                supervisor.consider_effort(supervisor.resume, suite(range(10000, 10020)))
                self.assertEqual(jobs.call_count, 2)
            self.assertEqual(supervisor.selected, prior)

    def setup_supervisor(self, directory, game_win=False):
        base = directory/'source'
        base.mkdir()
        (base/'model.safetensors').write_bytes(b'synthetic test weights; never loaded')
        (base/'state.json').write_text(json.dumps({'config': CONFIG}))
        reserves = 0 if game_win else None
        (base/'evaluation.json').write_text(json.dumps(suite(range(10000, 10020), reserves=reserves)))
        secondary = directory/'secondary.json'
        secondary.write_text(json.dumps(suite(range(10100, 10150), reserves=reserves)))
        return Supervisor(directory/'supervised', base, base, secondary, 1, game_win=game_win)

    def test_all_eight_mode_requires_outcome_baseline_and_keeps_old_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp), game_win=True)
            self.assertEqual(supervisor.config['target_level'], 8)
            self.assertTrue(supervisor.config['target_game_win'])
            self.assertFalse(supervisor.config['stop_on_target'])
            self.assertFalse(supervisor.meets_goal(supervisor.selected['combined']))
            with self.assertRaisesRegex(ValueError, 'all-eight winning'):
                supervisor.final_test()
        with tempfile.TemporaryDirectory() as tmp:
            old = self.setup_supervisor(Path(tmp))
            with self.assertRaisesRegex(ValueError, '70 outcome-assessed'):
                Supervisor(Path(tmp)/'new', old.resume, old.baseline,
                           Path(tmp)/'secondary.json', game_win=True)

    def test_all_eight_fresh_gate_requires_actual_winning_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp), game_win=True)
            supervisor.root.mkdir()
            (supervisor.resume/'optimizer.npz').write_bytes(b'synthetic optimizer')
            winner = suite(range(10000, 10020), reserves=1)
            supervisor.selected.update(promoted=True, replay='fixture', inspection='fixture',
                                       replay_game=winner['games'][0], combined=winner)
            with patch('rl.supervise.verify_replay', return_value=suite([10000], reserves=0)['games'][0]):
                with self.assertRaisesRegex(ValueError, 'winning replay'):
                    supervisor.final_test()
                self.assertFalse((supervisor.root/'final-test-started.json').exists())
            with patch('rl.supervise.verify_replay', return_value=winner['games'][0]):
                with patch.object(supervisor, 'job') as job:
                    with patch('rl.supervise.checked_suite', return_value=dict(
                            games=[], complete_games=100, incomplete_games=0,
                            outcome_assessed_games=100, verified_wins=0)):
                        supervisor.final_test()
            self.assertTrue((supervisor.root/'frozen-winner/model.safetensors').exists())
            self.assertEqual(supervisor.phase, 'target_verified_and_tested')
            self.assertEqual(job.call_args.args[1][job.call_args.args[1].index('--seed')+1], 40000)

    def test_already_verified_baseline_runs_final_gate_without_new_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp), game_win=True)
            supervisor.selected['promoted'] = True
            supervisor.selected['combined']['verified_wins'] = 1
            with patch.object(supervisor, 'final_test') as final:
                with patch.object(supervisor, 'command') as launch:
                    supervisor.run()
            final.assert_called_once_with()
            launch.assert_not_called()
            self.assertIsNone(supervisor.learner)

    def test_outcome_claims_and_replay_metadata_cannot_invent_a_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self.setup_supervisor(root, game_win=True)
            evaluation = suite(range(10000, 10020), reserves=0)
            evaluation['games'][0]['game_won'] = True
            p = root/'forged.json'; p.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, 'outcome evidence'):
                checked_suite(p, range(10000, 10020), CONFIG, 'identity')
            game = suite([10000], reserves=1)['games'][0]
            frame = np.full((16, 64), 128, np.uint8)
            frame[0, 6:11] = list(b'00500'); frame[0, 59:64] = list(b'00008')
            frame[10, 28:37] = list(b'GAME OVER')
            replay = root/'fixture.html'; replay.write_text('synthetic fixture')
            report = root/'inspection.json'
            model_hash = digest(supervisor.resume/'model.safetensors')
            report.write_text(json.dumps(dict(checkpoint_sha256=model_hash,
                replay_sha256=digest(replay), seeded_action_mismatches=0, all={'actions':100})))
            metadata = dict(game, checkpoint_sha256=model_hash, tstates=50000,
                            observation_stride=2, environment_version=ENVIRONMENT_VERSION)
            with patch('rl.supervise.decode_replay', return_value=(metadata, [frame]*101, [0]*100)):
                with self.assertRaisesRegex(ValueError, 'screen outcome differs'):
                    verify_replay(supervisor.resume, replay, report, game)

    def test_incremental_partial_line_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'metrics.jsonl'
            tail = EventTail(p)
            self.assertEqual(tail.read(), [])
            p.write_bytes(b'{"steps":1}\n{"steps":')
            self.assertEqual(tail.read(), [{'steps': 1}])
            self.assertEqual(latest_event(p), {'steps': 1})
            with p.open('ab') as out:
                out.write(b'2}\n')
            self.assertEqual(tail.read(), [{'steps': 2}])
            self.assertEqual(tail.read(), [])
            self.assertEqual(latest_event(p), {'steps': 2})
            p.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                tail.read()

    def test_validation_checks_full_seed_set_summary_hash_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'evaluation.json'
            original = suite(range(10000, 10020))
            p.write_text(json.dumps(original))
            self.assertEqual(checked_suite(p, range(10000, 10020), CONFIG, 'identity'), original)
            # A duplicated primary is never accepted as a50-game secondary.
            with self.assertRaisesRegex(ValueError, 'seed set'):
                checked_suite(p, range(10100, 10150), CONFIG, 'identity')
            for key, value in [('mean_score', 0), ('checkpoint_sha256', 'changed'),
                               ('max_steps', 0), ('epsilon', .01), ('observation_stride', 1)]:
                changed = {**original, key: value}
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    checked_suite(p, range(10000, 10020), CONFIG, 'identity')
            changed = copy.deepcopy(original)
            changed['games'][0]['start_tstates'] += 1
            p.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'boot'):
                checked_suite(p, range(10000, 10020), CONFIG, 'identity')

    def test_incomplete_suite_never_qualifies(self):
        previous = level_rank(suite(range(10000, 10020), 8), 10)
        good = suite(range(10000, 10020), 9)
        self.assertTrue(new_primary_record(good, previous))
        good['incomplete_games'] = 1
        self.assertFalse(new_primary_record(good, previous))
        self.assertFalse(new_primary_record(good, None))

    def test_fresh_tests_rejected_before_any_output_or_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            with patch.object(supervisor, 'job') as job:
                with self.assertRaisesRegex(ValueError, 'verified, promoted Level10'):
                    supervisor.final_test()
                job.assert_not_called()
                self.assertFalse(supervisor.root.exists())

    def test_operator_stop_has_no_budget_cap_and_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            process = FakeProcess()
            def stop_after_first_poll(seconds):
                (supervisor.root/'STOP').touch()
            with patch.object(supervisor, 'command', return_value=process) as command:
                with patch('rl.supervise.time.sleep', side_effect=stop_after_first_poll):
                    with self.assertRaises(InterruptedError):
                        supervisor.run()
                arguments = command.call_args.args[1]
                self.assertEqual(arguments[arguments.index('--steps')+1], 0)
                self.assertNotIn('--additional-steps', arguments)
            self.assertEqual(process.poll(), 0)
            status = json.loads((supervisor.root/'status.json').read_text())
            self.assertEqual(status['phase'], 'operator_stopped')
            self.assertFalse(status['learner_running'])
            self.assertFalse((supervisor.root/'final-test-started.json').exists())

    def test_atomic_heartbeat_and_runtime_integrity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            supervisor.root.mkdir()
            supervisor.heartbeat()
            self.assertFalse((supervisor.root/'status.tmp').exists())
            supervisor.expected_code['rl/ppo.py'] = 'not-the-current-hash'
            with self.assertRaisesRegex(RuntimeError, 'runtime code changed'):
                supervisor.safeguards()

    def test_automatic_secondary_and_replay_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = self.setup_supervisor(root)
            supervisor.root.mkdir()
            candidate = root/'step-test'
            candidate.mkdir()
            (candidate/'model.safetensors').write_bytes(b'synthetic candidate')
            (candidate/'state.json').write_text(json.dumps({'config': CONFIG}))
            primary = suite(range(10000, 10020), 9)
            def job(module, arguments, output_name):
                if module == 'rl.evaluate':
                    p = arguments[arguments.index('--output')+1]
                    Path(p).write_text(json.dumps(suite(range(10100, 10150), 9)))
            with patch.object(supervisor, 'job', side_effect=job) as calls:
                with patch('rl.supervise.verify_replay', return_value={'level': 9}):
                    result = supervisor.audit(candidate, primary)
            self.assertTrue(result['promoted'])
            self.assertEqual(result['combined']['complete_games'], 70)
            self.assertEqual(supervisor.selected['checkpoint'], str(candidate))
            self.assertEqual([c.args[0] for c in calls.call_args_list],
                             ['rl.evaluate', 'rl.record', 'rl.inspect_policy'])
            self.assertFalse((supervisor.root/'final-test-started.json').exists())

    def test_final_test_only_after_verified10_and_frozen_identity(self):
        # Synthetic lifecycle fixture only: no emulator or fresh game is run.
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            supervisor.root.mkdir()
            (supervisor.resume/'optimizer.npz').write_bytes(b'synthetic optimizer')
            supervisor.selected.update(promoted=True, replay='synthetic', inspection='synthetic',
                                       replay_game={'level': 10})
            supervisor.selected['combined']['level_reach_counts']['10'] = 1
            with patch('rl.supervise.verify_replay', return_value={'level': 10}):
                with patch.object(supervisor, 'job') as job:
                    with patch('rl.supervise.checked_suite', return_value={
                            'games': [], 'complete_games': 100, 'incomplete_games': 0}):
                        supervisor.final_test()
            self.assertEqual(supervisor.phase, 'target_verified_and_tested')
            self.assertEqual(digest(supervisor.root/'frozen-level10/model.safetensors'),
                             supervisor.selected['checkpoint_sha256'])
            arguments = job.call_args.args[1]
            self.assertEqual(arguments[arguments.index('--seed')+1], 40000)
            self.assertEqual(arguments[arguments.index('--games')+1], 100)
            self.assertEqual(arguments[arguments.index('--max-steps')+1], 0)
            self.assertTrue((supervisor.root/'final-test-started.json').exists())

    def test_sustained_regression_requests_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            learner = supervisor.root/'learner'
            learner.mkdir(parents=True)
            rows = []
            for step in range(1, 6):
                checkpoint = learner/f'step-{step:09d}'
                checkpoint.mkdir()
                (checkpoint/'model.safetensors').write_bytes(b'synthetic')
                (checkpoint/'evaluation.json').write_text(json.dumps(suite(range(10000, 10020), 4)))
                rows.append(json.dumps({'event': 'validation', 'steps': step}))
            (learner/'metrics.jsonl').write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(RuntimeError, 'five consecutive'):
                supervisor.process_events()
            self.assertEqual(supervisor.misses, 5)

    def test_real_readonly_helper_closes_output_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = self.setup_supervisor(Path(tmp))
            supervisor.root.mkdir()
            logs = Path(tmp)/'empty-log'
            logs.mkdir()
            (logs/'metrics.jsonl').write_text('')
            supervisor.job('rl.status', [logs], 'helper.log')
            self.assertIsNone(supervisor.helper)
            self.assertEqual(supervisor.handles, [])
            self.assertIn('no complete log records', (supervisor.root/'helper.log').read_text())

    def test_real_ppo_parser_accepts_supervisor_budget_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'evaluation-only'
            source.mkdir()
            (source/'state.json').write_text(json.dumps({'evaluation_only': True}))
            output = Path(tmp)/'must-not-exist'
            result = subprocess.run([sys.executable, '-m', 'rl.ppo',
                *map(str, learner_arguments(output, source))], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            # Reaching the semantic resume check proves argparse accepted the
            # budget flags. No GPU, emulator, or output directory is created.
            self.assertIn('evaluation-only checkpoint cannot resume', result.stderr)
            self.assertNotIn('not allowed with argument', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
