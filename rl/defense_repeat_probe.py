"""Read-only duration-use audit of an already verified own-policy replay.

Reconstruct every base command using frozen weights and the original visible
history/boundaries. Export aggregate duration counts, never training examples.
This is not a new native evaluation or a search for better actions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_alias_probe import replay_observation
from .defense_learning import REPEAT_ALGORITHM, load_policy, sha256
from .defense_loss_probe import analyze
from .defense_repeat import LearnedRepeatPolicy


def duration_summary(indices, starts, durations):
    indices, starts = np.asarray(indices), np.asarray(starts)
    if (indices.ndim != 1 or len(indices) == 0 or starts.shape != indices.shape
            or starts.dtype != np.bool_ or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(durations))):
        raise ValueError('requires duration indices and option-start flags for every base command')
    counts = np.bincount(indices, minlength=len(durations))
    choices = np.bincount(indices[starts], minlength=len(durations))
    return dict(base_commands=len(indices), neural_option_decisions=int(starts.sum()),
                decisions_by_duration=choices.tolist(), executed_commands_by_duration=counts.tolist(),
                long_option_command_fraction=float(counts[np.asarray(durations) > 1].sum()/len(indices)))


def probe(bundle):
    bundle = Path(bundle).resolve(strict=True)
    reviewed, frames, actions = analyze(bundle)
    config = json.loads((bundle/'state.json').read_text())['config']
    if config.get('algorithm') != REPEAT_ALGORITHM or config.get('allow_enter', False):
        raise ValueError('requires the standard learned-duration policy')
    repeat_source = Path(__file__).with_name('defense_repeat.py')
    if config['repeat_source_sha256'] != sha256(repeat_source):
        raise ValueError('duration executor source differs from the recorded implementation')
    policy, loaded = load_policy(bundle/'model.safetensors')
    if loaded != config or not isinstance(policy, LearnedRepeatPolicy):
        raise ValueError('duration policy/configuration mismatch')
    policy.reset_seed(reviewed['result']['seed'] + 1_000_000)
    durations = tuple(config['learned_repeats'])
    losses = {life['visible_loss_frame'] for life in reviewed['lives']}
    held_indices, starts = [], []
    choice = remaining = None
    decision_started = False
    original_infer = policy.infer

    def counting_infer(obs):
        nonlocal choice, remaining, decision_started
        values = original_infer(obs)
        if len(values) != 1:
            raise ValueError('serial diagnostic expected')
        choice = int(np.argmax(values[0])) // 20
        remaining = durations[choice]
        decision_started = True
        return values

    policy.infer = counting_infer
    cancelled = interrupted = 0
    for index, expected in enumerate(actions):
        decision_started = False
        obs = replay_observation(frames, index, config.get('observation_stride', 1))[None]
        selected = int(policy(obs)[0])
        if selected != int(expected):
            raise ValueError(f'learned base command differs at {index}')
        held_indices.append(choice)
        starts.append(decision_started)
        remaining -= 1
        if policy.memories[policy.serial_keys[0]][1] != remaining:
            raise ValueError('diagnostic countdown differs from policy execution')
        ended = index + 1 in losses
        if ended:
            cancelled += remaining
            interrupted += int(remaining > 0)
        policy.observe_boundaries(np.asarray([ended], dtype=bool))
    held_indices, starts = np.asarray(held_indices), np.asarray(starts)
    whole = duration_summary(held_indices, starts, durations)
    planned = sum(d*c for d,c in zip(durations, whole['decisions_by_duration'], strict=True))
    if planned != len(actions) + cancelled:
        raise ValueError('planned/executed/interrupted duration accounting mismatch')
    windows = []
    for life in reviewed['lives']:
        stop = life['visible_loss_frame']
        start = max(life['previous_visible_loss_frame'], stop - 64)
        windows.append(dict(life=life['life'], first_base_action=start, stop_exclusive=stop,
                            **duration_summary(held_indices[start:stop], starts[start:stop], durations)))
    for name, digest in reviewed['source_hashes'].items():
        if sha256(bundle/name) != digest:
            raise RuntimeError('source changed during diagnostic')
    return dict(bundle=str(bundle), source_hashes=reviewed['source_hashes'],
                probe_source_sha256=sha256(Path(__file__)), durations=durations,
                result=reviewed['result'], whole_replay=whole,
                interrupted_options=interrupted, cancelled_base_commands=cancelled,
                pre_visible_loss_windows=windows, replay_commands_reproduced=len(actions),
                original_native_verification=reviewed['original_native_verification'],
                native_reexecution=False, parameter_updates=0, training_data_written=False,
                ranking_eligible=False, diagnostic_only=True,
                limitations=[
                    'One selected verified replay, not a representative or fresh evaluation sample.',
                    'Durations describe neural commitments, not movement, survival, obstacle passage or a route.',
                    'Base-command counts include original animations; visible life loss can lag physical collision.',
                    'Window decisions count only starts inside the window; a hold can begin before it.',
                    'Native verification is reused; this diagnostic reconstructs policy commands from recorded screens.',
                    'Frozen weights only; no replay or diagnostic examples enter learning.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    result = probe(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
