"""Aggregate random-rollout diagnostics from verified own pre-loss states.

No branch actions, routes, scores, snapshots or training examples are exported.
These isolated rollouts never enter training, resets, evaluation or ranking.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv
from .defense_exploration_actions import action_probabilities
from .defense_learning import sha256
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore
from .persistent_exploration import PersistentExploration


OFFSETS = (256, 128, 64)
EXTRA_DECISIONS = 256


def summarize(durations, lost, distance, stages):
    durations = np.asarray(durations)
    lost = np.asarray(lost)
    stages = np.asarray(stages)
    if (durations.ndim != 1 or len(durations) == 0
            or not np.issubdtype(durations.dtype, np.integer)
            or lost.shape != durations.shape or lost.dtype != np.bool_
            or stages.shape != durations.shape or np.any(durations <= 0)
            or np.any(durations > distance + EXTRA_DECISIONS)
            or np.any((stages < 1) | (stages > 3))):
        raise ValueError('invalid aggregate rollout records')
    if np.any(durations[~lost] != distance + EXTRA_DECISIONS):
        raise ValueError('a censored rollout must reach the fixed horizon')
    # Censored observations give only a lower bound on time to visible loss.
    return dict(branches=len(durations), visible_losses=int(lost.sum()),
                horizon_censored=int((~lost).sum()),
                observed_duration_quantiles=np.quantile(durations, [0, .25, .5, .75, 1]).tolist(),
                alive_beyond_original_visible_loss=int((durations > distance).sum()),
                alive_beyond_original_plus_32=int((durations > distance + 32).sum()),
                alive_beyond_original_plus_128=int((durations > distance + 128).sum()),
                stage_2_or_higher=int((stages >= 2).sum()),
                stage_3=int((stages >= 3).sum()))


def probe(bundle, branches=256, seed=104729):
    if (isinstance(branches, bool) or not isinstance(branches, int)
            or not 1 <= branches <= 4096 or not isinstance(seed, int) or seed < 0):
        raise ValueError('requires 1..4096 branches and a nonnegative integer seed')
    bundle = Path(bundle).resolve(strict=True)
    reviewed, frames, actions = analyze(bundle)
    config = json.loads((bundle / 'state.json').read_text())['config']
    with np.load(bundle / 'trace.npz', allow_pickle=False) as trace:
        rewards = trace['rewards']
        metadata = json.loads(str(trace['metadata']))
    if config.get('allow_enter', False) or config.get('eval_max_steps', 0):
        raise ValueError('requires the standard uncapped complete-game profile')
    native = Path(__file__).resolve().parents[1] / 'libtrs.so'
    if sha256(native) != config['native_sha256']:
        raise ValueError('native build differs from the replay')
    anchors = [dict(life=life['life'], decisions_before_visible_loss=d,
                    decision=life['visible_loss_frame'] - d)
               for life in reviewed['lives'] for d in OFFSETS
               if life['visible_loss_frame'] - d >= life['previous_visible_loss_frame']]
    wanted = {a['decision'] for a in anchors}
    snapshots = {}
    probabilities = action_probabilities('stage1-balanced', 20)
    env = DefenseEnv(tstates=config['tstates'], observation_stride=config.get('observation_stride', 1))
    results = []
    try:
        initial = env.reset(metadata['result']['seed'])
        np.testing.assert_array_equal(initial[-1], frames[0])
        for index, action in enumerate(actions):
            if index in wanted:
                snapshots[index] = capture(env)
            following, reward, terminal, truncated, info = env.step(int(action))
            np.testing.assert_array_equal(following[-1], frames[index + 1])
            if reward != rewards[index] or truncated or terminal != (index == len(actions) - 1):
                raise ValueError('native reproduction differs from the original trace')
        if set(snapshots) != wanted or not info['game_over']:
            raise ValueError('missing own replay anchors or native ending')
        for anchor in anchors:
            distance = anchor['decisions_before_visible_loss']
            saved = snapshots[anchor['decision']]
            durations, losses, stages = [], [], []
            for trial in range(branches):
                restore(env, saved)
                np.testing.assert_array_equal(env.video, frames[anchor['decision']])
                # Common random sequences across anchors; no screen-based choice.
                explorer = PersistentExploration(1, 20, 64, 1.5,
                    np.random.default_rng(np.random.SeedSequence([seed, trial])), probabilities)
                for step in range(distance + EXTRA_DECISIONS):
                    action = int(explorer.select(np.zeros(1, np.int32), 1.)[0])
                    _, _, terminal, truncated, info = env.step(action)
                    if truncated:
                        raise ValueError('unexpected environment truncation')
                    if terminal or info['life_lost']:
                        break
                durations.append(step + 1)
                losses.append(bool(terminal or info['life_lost']))
                stages.append(info['highest_stage'])
            results.append(dict(**anchor, **summarize(durations, losses, distance, stages)))
    finally:
        env.close()
    for name, digest in reviewed['source_hashes'].items():
        if sha256(bundle / name) != digest:
            raise RuntimeError('source changed during diagnostic')
    return dict(bundle=str(bundle), source_hashes=reviewed['source_hashes'],
                probe_source_sha256=sha256(Path(__file__)), native_sha256=config['native_sha256'],
                original_native_trajectory_reproduced=len(actions),
                original_neural_verification_reused=True, diagnostic_only=True,
                random_seed=seed, branches_per_anchor=branches, extra_decisions=EXTRA_DECISIONS,
                random_policy=dict(epsilon=1., max_repeat=64, exponent=1.5,
                                   fixed_action_probabilities=probabilities.tolist()),
                anchors=results, parameter_updates=0, training_data_written=False,
                ranking_eligible=False, snapshots_decoded=False,
                chosen_action_or_route_output=False,
                limitations=[
                    'Selected own replay states, not all training resets or representative complete games.',
                    'Longer visible survival is not physical collision timing or measured course passage.',
                    'Actions can change emulated work per decision; decision duration is not course position.',
                    'Failure to survive under finite random sampling does not prove unavoidability.',
                    'Censored durations are lower bounds, not time-to-loss measurements.',
                    'Common random action sequences across anchors; no actions are selected based on outcomes.',
                    'No actions, scores, trajectories, snapshots or branch examples are exported to training.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branches', type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    result = probe(args.bundle, args.branches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
