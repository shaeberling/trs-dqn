"""Local, single-learner supervision with automatic validation and replay gates.

This runs ordinary local training, not a Codex/API agent. It cannot resume a
usage-limited chat. No demonstrations, policy overrides, or automatic tuning.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from .artifacts import effort_rank, publish_record
from .env import ENVIRONMENT_VERSION, screen_info
from .evaluate import level_rank, summary
from .inspect_policy import decode_replay
from .outcome import OUTCOME_VERSION, screen_outcome, verified_win


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def checked_suite(path, seeds, config, model_hash, max_steps=200000):
    data = json.loads(Path(path).read_text())
    games = data['games']
    if [g['seed'] for g in games] != list(seeds):
        raise ValueError('evaluation seed set differs from declared protocol')
    for key in ('tstates', 'observation_stride', 'environment_version'):
        if data[key] != config[key]:
            raise ValueError('evaluation environment differs: '+key)
    if data.get('checkpoint_sha256', model_hash) != model_hash:
        raise ValueError('evaluation checkpoint hash differs')
    if (data['max_steps'] != max_steps or data['epsilon'] != 0
            or data.get('deterministic_override', False)
            or data['policy'] != 'learned categorical, sampled'):
        raise ValueError('evaluation guard or policy differs')
    for game in games:
        if (game['start_tstates'] != int(np.random.default_rng(game['seed']).integers(0, 200001))
                or game['episode_reward'] != game['score']
                or not (game['terminated'] or game['truncated'])
                or game['steps'] <= 0 or (max_steps and game['steps'] > max_steps)
                or (game['terminated'] and not game['game_over'])):
            raise ValueError('invalid evaluation boot, score or terminal record')
        if 'outcome_version' in game:
            if (game['outcome_version'] != OUTCOME_VERSION
                    or type(game.get('reserve_balls_visible')) is not int
                    or not 0 <= game['reserve_balls_visible'] <= 5
                    or type(game.get('game_won')) is not bool):
                raise ValueError('invalid outcome evidence')
            won = game['game_over'] and game['level'] == 8 and game['reserve_balls_visible'] > 0
            status = ('verified_win' if won else 'unverified_final_level'
                      if game['game_over'] and game['level'] == 8 else
                      'loss' if game['game_over'] else 'ongoing')
            if game['game_won'] != won or game.get('win_status') != status:
                raise ValueError('outcome evidence differs from reported result')
    result = summary(games)
    for key, value in result.items():
        if data[key] != value:
            raise ValueError('evaluation summary differs: '+key)
    return data


def verify_replay(checkpoint, replay, inspection, game):
    metadata, frames, actions = decode_replay(Path(replay).read_text())
    report = json.loads(Path(inspection).read_text())
    model_hash = digest(Path(checkpoint)/'model.safetensors')
    config = json.loads((Path(checkpoint)/'state.json').read_text())['config']
    if (metadata['checkpoint_sha256'] != model_hash
            or report['checkpoint_sha256'] != model_hash
            or report['replay_sha256'] != digest(replay)
            or report['seeded_action_mismatches'] != 0
            or report['all']['actions'] != len(actions)
            or len(frames) != len(actions)+1
            or not metadata['terminated'] or not metadata['game_over'] or metadata['truncated']):
        raise ValueError('replay or neural-action verification failed')
    if any(metadata[k] != config[k] for k in ('tstates', 'observation_stride', 'environment_version')):
        raise ValueError('replay screen timing or environment differs')
    for key in ('seed', 'score', 'level', 'steps', 'start_tstates'):
        if metadata[key] != game[key]:
            raise ValueError('replay differs from validation game: '+key)
    final_info = screen_info(frames[-1])
    if any(final_info[key] != metadata[key] for key in ('score', 'level', 'game_over')):
        raise ValueError('replay final screen differs from metadata')
    outcome = screen_outcome(frames[-1], final_info)
    for key, value in outcome.items():
        if ((key in metadata and metadata[key] != value)
                or (key in game and game[key] != value)):
            raise ValueError('replay screen outcome differs: '+key)
    metadata.update(outcome)
    return metadata


class EventTail:
    def __init__(self, path):
        self.path, self.offset, self.pending = Path(path), 0, b''

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as stream:
            if stream.seek(0, 2) < self.offset:
                raise ValueError('training log was truncated')
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.pending+chunk).split(b'\n')
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line]


def new_primary_record(result, previous, *, game_win=False):
    rank = level_rank(result, 8 if game_win else 10, game_win=game_win)
    return rank is not None and (previous is None or rank > previous)


def learner_arguments(run, resume, curriculum_score_interval=None, *, game_win=False):
    # PPO's absolute/additional budget flags are mutually exclusive, even at0.
    # Resume deliberately does not inherit either budget field, so --steps0
    # leaves both parsed limits zero without passing the second flag.
    result = ['--run', run, '--resume', resume, '--steps', 0]
    if curriculum_score_interval is not None:
        result += ['--curriculum-score-interval', curriculum_score_interval]
    if game_win:
        result += ['--target-game-win', '--target-level', 8, '--no-stop-on-target']
    return result


def latest_event(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size-65536))
        lines = stream.read().split(b'\n')[:-1]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


class Supervisor:
    def __init__(self, root, resume, baseline, baseline_secondary, poll_seconds=15,
                 curriculum_score_interval=None, artifacts=None, effort_artifacts=None,
                 baseline_selection=None, game_win=False, baseline_primary=None):
        self.root, self.resume, self.baseline = map(Path, (root, resume, baseline))
        self.config = json.loads((self.resume/'state.json').read_text())['config']
        self.game_win = game_win
        if game_win:
            self.config.update(target_game_win=True, target_level=8, stop_on_target=False)
        self.curriculum_score_interval = curriculum_score_interval
        if curriculum_score_interval is not None:
            if type(curriculum_score_interval) is not int or curriculum_score_interval < 0:
                raise ValueError('curriculum score interval must be a nonnegative integer')
            self.config['curriculum_score_interval'] = curriculum_score_interval
        self.artifacts = Path(artifacts) if artifacts else None
        self.effort_artifacts = Path(effort_artifacts) if effort_artifacts else None
        self.best_effort = None
        if self.effort_artifacts and (self.effort_artifacts/'current.json').exists():
            self.best_effort = json.loads((self.effort_artifacts/'current.json').read_text())['record']
            verify_replay(self.best_effort['checkpoint'], self.best_effort['replay'],
                          self.best_effort['inspection'], self.best_effort['replay_game'])
        self.poll_seconds = poll_seconds
        self.stop_requested = False
        self.learner = self.helper = None
        self.handles = []
        self.phase, self.steps, self.latest_primary = 'initializing', None, None
        self.misses, self.highest_full, self.highest_practice = 0, 1, None
        self.full_training_wins, self.practice_wins = 0, 0
        self.audited = set()
        self.expected_code = {str(p): digest(p) for p in map(Path, (
            'rl/ppo.py', 'rl/reference.py', 'rl/sil.py', 'rl/env.py',
            'rl/evaluate.py', 'rl/record.py', 'rl/inspect_policy.py', 'rl/supervise.py',
            'rl/artifacts.py', 'rl/curriculum.py', 'rl/snapshot.py', 'rl/vector.py', 'rl/outcome.py'))}
        if (self.config['environment_version'] != ENVIRONMENT_VERSION
                or self.config['tstates'] != 50000 or self.config['observation_stride'] != 2
                or self.config['eval_seed'] != 10000 or self.config['eval_games'] != 20
                or self.config['eval_max_steps'] != 200000
                or self.config['target_level'] != (8 if game_win else 10)
                or self.config['target_clears'] != 1 or self.config['eval_every'] != 500000
                or self.config['envs'] != 32 or self.config['curriculum_boot_envs'] != 16
                or self.config['curriculum_min_level'] != 8):
            raise ValueError('supervised protocol requires the declared Level10 validation settings')
        baseline_hash = digest(self.baseline/'model.safetensors')
        primary = checked_suite(baseline_primary or self.baseline/'evaluation.json', range(10000, 10020),
                                self.config, baseline_hash)
        secondary = checked_suite(baseline_secondary, range(10100, 10150), self.config, baseline_hash)
        self.selected = {'checkpoint': str(self.baseline), 'checkpoint_sha256': baseline_hash,
                         'primary': primary, 'secondary': secondary,
                         'combined': summary(primary['games']+secondary['games'])}
        if self.selected['combined']['incomplete_games']:
            raise ValueError('baseline must contain70 complete validation games')
        if game_win and self.selected['combined'].get('outcome_assessed_games') != 70:
            raise ValueError('all-eight baseline requires70 outcome-assessed validation games')
        if baseline_selection:
            recorded = json.loads(Path(baseline_selection).read_text())
            for key in ('checkpoint_sha256', 'primary', 'secondary', 'combined'):
                if recorded[key] != self.selected[key]:
                    raise ValueError('baseline selection differs: '+key)
            verify_replay(self.baseline, recorded['replay'], recorded['inspection'], recorded['replay_game'])
            self.selected = recorded
        resume_primary = checked_suite(self.resume/'evaluation.json', range(10000, 10020),
                                       self.config, digest(self.resume/'model.safetensors'))
        if resume_primary['incomplete_games']:
            raise ValueError('resume source must have a complete primary suite')
        self.screen_rank = max(self.rank(primary), self.rank(resume_primary))
        self.tail = EventTail(self.root/'learner/metrics.jsonl')

    def rank(self, result):
        return level_rank(result, 8 if self.game_win else 10, game_win=self.game_win)

    def meets_goal(self, result):
        return (not result['incomplete_games'] and result['complete_games'] > 0
                and (result.get('verified_wins', 0) if self.game_win else
                     result['level_reach_counts'].get('10', 0)) > 0)

    def heartbeat(self, **extra):
        log = self.root/'learner/metrics.jsonl'
        last = latest_event(log)
        write_status(self.root/'status.json', {
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'phase': self.phase,
            'supervisor_pid': os.getpid(), 'learner_pid': self.learner.pid if self.learner else None,
            'learner_running': self.learner is not None and self.learner.poll() is None,
            'steps': last.get('steps') if last else self.steps,
            'learner_last_event': last.get('event') if last else None,
            'learner_log_age_seconds': round(time.time()-log.stat().st_mtime, 1) if log.exists() else None,
            'latest_primary': self.latest_primary,
            'highest_complete_training': self.highest_full,
            'highest_restored_practice': self.highest_practice,
            'verified_full_training_wins': self.full_training_wins,
            'verified_restored_practice_wins': self.practice_wins,
            'selected_checkpoint': self.selected['checkpoint'],
            'selected_validation': self.selected['combined'],
            'goal': 'beat_all_eight_original_levels' if self.game_win else 'historical_level10',
            'consecutive_complete_primaries_without5': self.misses, **extra})

    def safeguards(self):
        if self.stop_requested or (self.root/'STOP').exists():
            raise InterruptedError('operator requested stop')
        if shutil.disk_usage(self.root).free < 5*1024**3:
            raise RuntimeError('less than5GiB disk free; checkpoints preserved, no automatic deletion')
        if any(digest(path) != expected for path, expected in self.expected_code.items()):
            raise RuntimeError('runtime code changed; review required before continuing')
        if self.learner is not None and self.learner.poll() not in (None, 0):
            raise RuntimeError('learner exited unsuccessfully')
        log = self.root/'learner/metrics.jsonl'
        if (self.learner is not None and self.learner.poll() is None and log.exists()
                and time.time()-log.stat().st_mtime > 900):
            raise RuntimeError('learner has produced no log event for15 minutes; review required')

    def command(self, module, arguments, output_name):
        self.safeguards()
        output = (self.root/output_name).open('x')
        self.handles.append(output)
        return subprocess.Popen([sys.executable, '-m', module, *map(str, arguments)],
                                stdout=output, stderr=subprocess.STDOUT)

    def job(self, module, arguments, output_name):
        self.helper = self.command(module, arguments, output_name)
        while self.helper.poll() is None:
            self.safeguards()
            self.heartbeat(helper_pid=self.helper.pid)
            time.sleep(self.poll_seconds)
        code = self.helper.returncode
        self.helper = None
        self.handles.pop().close()
        if code:
            raise RuntimeError(f'{module} exited{code}; see{output_name}')

    def stop_children(self):
        for child in (self.helper, self.learner):
            if child is not None and child.poll() is None:
                child.terminate()
        for child in (self.helper, self.learner):
            if child is not None:
                while child.poll() is None:
                    self.heartbeat(stop_pending=True)
                    time.sleep(self.poll_seconds)

    def audit(self, checkpoint, primary):
        directory = self.root/'audits'/checkpoint.name
        directory.mkdir(parents=True, exist_ok=False)
        model = checkpoint/'model.safetensors'
        model_hash = digest(model)
        self.phase = 'secondary_validation'
        self.job('rl.evaluate', [model, '--games', 50, '--envs', 20, '--seed', 10100,
                 '--max-steps', 200000, '--output', directory/'secondary.json'],
                 str(directory.relative_to(self.root)/'secondary.log'))
        secondary = checked_suite(directory/'secondary.json', range(10100, 10150),
                                  self.config, model_hash)
        combined = summary(primary['games']+secondary['games'])
        result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': model_hash,
                  'primary': primary, 'secondary': secondary, 'combined': combined,
                  'promoted': False}
        if new_primary_record(combined, self.rank(self.selected['combined']), game_win=self.game_win):
            game = max(primary['games']+secondary['games'],
                       key=effort_rank)
            replay, inspection = directory/'replay.html', directory/'inspection.json'
            self.phase = 'replay_verification'
            self.job('rl.record', [model, '--seed', game['seed'], '--max-steps', 0,
                     '--output', replay], str(directory.relative_to(self.root)/'record.log'))
            try:
                self.job('rl.inspect_policy', [model, replay, '--output', inspection],
                         str(directory.relative_to(self.root)/'inspect.log'))
            except RuntimeError:
                if 'seeded actions differ' not in (directory/'inspect.log').read_text():
                    raise
                self.job('rl.inspect_policy', [model, replay, '--output', inspection,
                         '--batch-size', 1], str(directory.relative_to(self.root)/'inspect-batch1.log'))
            verify_replay(checkpoint, replay, inspection, game)
            if digest(model) != model_hash:
                raise ValueError('candidate weights changed during verification')
            result.update(promoted=True, replay=str(replay), inspection=str(inspection),
                          replay_game=game)
            if self.artifacts:
                publish_record(result, self.artifacts)
            self.selected = result
            write_status(self.root/'selected.json', result)
        write_status(directory/'decision.json', result)
        self.consider_effort(checkpoint, primary, secondary)
        self.phase = 'training'
        return result

    def consider_effort(self, checkpoint, primary, secondary=None):
        if not self.effort_artifacts:
            return
        games = primary['games']+(secondary['games'] if secondary else [])
        games = [g for g in games if g['terminated'] and not g['truncated']]
        if not games:
            return
        game = max(games, key=effort_rank)
        if self.best_effort and effort_rank(game) <= effort_rank(self.best_effort['replay_game']):
            return
        # A single-game record never changes validation-based model selection.
        directory = self.root/'efforts'/f"{checkpoint.name}-seed-{game['seed']}"
        directory.mkdir(parents=True, exist_ok=False)
        model = checkpoint/'model.safetensors'
        replay, inspection = directory/'replay.html', directory/'inspection.json'
        self.phase = 'best_effort_replay'
        self.job('rl.record', [model, '--seed', game['seed'], '--max-steps', 0, '--output', replay],
                 str(directory.relative_to(self.root)/'record.log'))
        try:
            self.job('rl.inspect_policy', [model, replay, '--output', inspection],
                     str(directory.relative_to(self.root)/'inspect.log'))
        except RuntimeError:
            if 'seeded actions differ' not in (directory/'inspect.log').read_text():
                raise
            self.job('rl.inspect_policy', [model, replay, '--output', inspection, '--batch-size', 1],
                     str(directory.relative_to(self.root)/'inspect-batch1.log'))
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(model),
                      replay=str(replay), inspection=str(inspection), replay_game=game, primary=primary)
        if secondary:
            result['secondary'] = secondary
        publish_record(result, self.effort_artifacts, 'single_game_best_effort')
        self.best_effort = result
        write_status(directory/'record.json', result)
        self.phase = 'training'

    def final_test(self):
        # No fresh seed or result is accessed before complete validation/replay gates.
        selected = self.selected
        goal_name = 'all-eight winning' if self.game_win else 'Level10'
        if not selected.get('promoted') or not self.meets_goal(selected['combined']):
            raise ValueError(f'fresh tests require a verified, promoted {goal_name} policy')
        metadata = verify_replay(selected['checkpoint'], selected['replay'], selected['inspection'],
                                 selected['replay_game'])
        if not (verified_win(metadata) if self.game_win else metadata['level'] >= 10):
            raise ValueError(f'fresh tests require a {goal_name} replay')
        self.phase = 'freezing_winner' if self.game_win else 'freezing_level10'
        self.stop_children()
        frozen = self.root/('frozen-winner' if self.game_win else 'frozen-level10')
        frozen.mkdir(exist_ok=False)
        for name in ('model.safetensors', 'optimizer.npz', 'state.json', 'evaluation.json'):
            shutil.copy2(Path(selected['checkpoint'])/name, frozen/name)
        if digest(frozen/'model.safetensors') != selected['checkpoint_sha256']:
            raise ValueError('frozen model identity differs')
        with (self.root/'final-test-started.json').open('x') as out:
            json.dump({'checkpoint_sha256': selected['checkpoint_sha256'],
                       'seeds': [40000, 40099], 'max_steps': 0}, out)
        self.phase = 'fresh_final_test'
        self.job('rl.evaluate', [frozen/'model.safetensors', '--games', 100, '--envs', 20,
                 '--seed', 40000, '--max-steps', 0, '--output', self.root/'fresh-test.json'],
                 'fresh-test.log')
        result = checked_suite(self.root/'fresh-test.json', range(40000, 40100),
                               self.config, selected['checkpoint_sha256'], max_steps=0)
        if result['incomplete_games'] or result['complete_games'] != 100:
            raise ValueError('fresh final evaluation did not finish100 complete games')
        if self.game_win and result.get('outcome_assessed_games') != 100:
            raise ValueError('fresh final evaluation lacks100 screen-outcome assessments')
        self.phase = 'target_verified_and_tested'
        self.heartbeat(final_test={k: v for k, v in result.items() if k != 'games'})

    def process_events(self):
        for row in self.tail.read():
            self.steps = row.get('steps', self.steps)
            event = row['event']
            if event == 'error':
                raise RuntimeError('learner error: '+row['error'])
            if event in ('episode', 'curriculum_episode') and row['terminated'] and not row['truncated']:
                if event == 'episode':
                    self.highest_full = max(self.highest_full, row['level'])
                    self.full_training_wins += int(verified_win(row))
                else:
                    self.highest_practice = max(self.highest_practice or 1, row['level'])
                    self.practice_wins += int(verified_win({**row, 'full_game': True}))
            if event != 'validation':
                continue
            checkpoint = self.root/'learner'/f"step-{row['steps']:09d}"
            primary = checked_suite(checkpoint/'evaluation.json', range(10000, 10020),
                                    self.config, digest(checkpoint/'model.safetensors'))
            self.latest_primary = {k: v for k, v in primary.items() if k != 'games'}
            self.consider_effort(checkpoint, primary)
            if not primary['incomplete_games']:
                self.misses = self.misses+1 if not primary['level_reach_counts'].get('5', 0) else 0
            else:
                self.misses = 0
            if self.misses >= 5:
                raise RuntimeError('five consecutive complete primary suites miss Level5; review needed')
            if new_primary_record(primary, self.screen_rank, game_win=self.game_win):
                self.screen_rank = self.rank(primary)
                self.audited.add(row['steps'])
                result = self.audit(checkpoint, primary)
                if result['promoted'] and self.meets_goal(result['combined']):
                    self.final_test()
                    return True
        return False

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        write_status(self.root/'selected.json', self.selected)
        if self.artifacts and self.selected.get('replay'):
            publish_record(self.selected, self.artifacts)
        write_status(self.root/'protocol.json', {
            'resume': str(self.resume), 'baseline': str(self.baseline),
            'goal': 'beat_all_eight_original_levels' if self.game_win else 'historical_level10',
            'outcome_version': OUTCOME_VERSION if self.game_win else None,
            'curriculum_score_interval_override': self.curriculum_score_interval,
            'selected_artifacts': str(self.artifacts) if self.artifacts else None,
            'effort_artifacts': str(self.effort_artifacts) if self.effort_artifacts else None,
            'code_hashes': self.expected_code, 'learner_action_limit': 0,
            'primary': '20 reused seeds10000–10019, every500k actions,200k guard',
            'selection': 'Audit each new primary depth/count record on50 reused secondary games; '
                         'promote only higher complete70-game rank and verified uncapped replay.',
            'stops': ['operator STOP or signal', 'runtime/integrity error', 'disk free below5GiB',
                      'no learner log for15 minutes (fault watchdog, not a training time budget)',
                      'five consecutive complete primaries withoutLevel5', 'verified goal replay and100fresh games'],
            'no_codex_or_api_calls': True, 'no_automatic_hyperparameter_changes': True})
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        def stop(signum, frame):
            self.stop_requested = True
        for sig in old_handlers:
            signal.signal(sig, stop)
        try:
            if self.selected.get('promoted') and self.meets_goal(self.selected['combined']):
                self.final_test()
                return
            self.learner = self.command('rl.ppo', learner_arguments(
                                        self.root/'learner', self.resume, self.curriculum_score_interval,
                                        game_win=self.game_win),
                                        'learner-output.log')
            self.phase = 'training'
            while True:
                self.safeguards()
                if self.process_events():
                    return
                self.heartbeat()
                if self.learner.poll() is not None:
                    # A helper audit may finish after the learner's target-stop;
                    # drain validation events appended while that audit ran.
                    if self.process_events():
                        return
                    raise RuntimeError('learner ended without a verified final result; review required')
                time.sleep(self.poll_seconds)
        except BaseException as error:
            self.phase = 'stopping_for_review'
            self.heartbeat(reason=str(error))
            self.stop_children()
            self.phase = 'operator_stopped' if isinstance(error, InterruptedError) else 'needs_attention'
            self.heartbeat(reason=str(error))
            raise
        finally:
            for handle in self.handles:
                handle.close()
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-secondary', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--curriculum-score-interval', type=int)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--effort-artifacts', type=Path)
    parser.add_argument('--baseline-selection', type=Path)
    parser.add_argument('--baseline-primary', type=Path)
    parser.add_argument('--game-win', action='store_true',
                        help='target a verified win across all eight original levels')
    args = parser.parse_args()
    if not args.game_win:
        parser.error('the original game ends after eight levels; use --game-win for the authorized goal')
    runs = Path('runs').resolve()
    target = args.run.resolve()
    if (target == runs or not target.is_relative_to(runs) or args.run.exists()
            or not np.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60):
        parser.error('use a new directory within runs and a1–60 second poll interval')
    runs.mkdir(exist_ok=True)
    with (runs/'level10-supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another Level10 supervisor holds the lock')
        Supervisor(args.run, args.resume, args.baseline, args.baseline_secondary,
                   args.poll_seconds, args.curriculum_score_interval, args.artifacts,
                   args.effort_artifacts, args.baseline_selection, args.game_win,
                   args.baseline_primary).run()


if __name__ == '__main__':
    main()
