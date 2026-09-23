"""Archive early rendered-screen approaches from verified own-policy games.

This source is proposal data only. The policy is never given a life-loss
clock, source offset, native snapshot, route, target action or extra reward.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OFFSETS = (370, 350, 300, 256)


def visible_approaches(frames, events, offsets=OFFSETS):
    frames = np.asarray(frames)
    if (frames.ndim != 3 or frames.shape[1:] != (16, 64)
            or frames.dtype != np.uint8 or len(offsets) != 4
            or tuple(sorted(set(offsets), reverse=True)) != tuple(offsets)
            or min(offsets) < 4):
        raise ValueError('invalid own rendered-screen trace')
    losses = [event for event in events if event.get('life_lost')]
    if len(losses) != 4:
        raise ValueError('requires four visible own-policy ship losses')
    samples, previous = [], 0
    for life, event in enumerate(losses, 1):
        stop = event['frame']
        if not isinstance(stop, int) or stop <= previous or stop >= len(frames):
            raise ValueError('invalid visible loss frame')
        if stop-max(offsets)-3 <= previous:
            raise ValueError('own life is too short for the fixed early screen window')
        screens = np.stack([frames[stop-offset-3:stop-offset+1]
                            for offset in offsets])
        if screens.shape != (4, 4, 16, 64):
            raise ValueError('incomplete own screen history')
        samples.append(dict(life=life, visible_loss_frame=stop,
            previous_visible_loss_frame=previous, visible_score_at_loss=event['score'],
            screens=screens))
        previous = stop
    if previous != len(frames)-1:
        raise ValueError('last own visible loss must end the complete game')
    return samples


def main():
    from .defense_learning import (game_rank, load_policy, record_game, sha256,
                                   verify_policy_trace, write_json)
    from .defense_world_data import disk_guard
    from .defense import ENVIRONMENT_VERSION, GAME_SHA256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--first-training-seed', type=int, required=True)
    parser.add_argument('--max-attempts', type=int, default=48)
    parser.add_argument('--offsets', type=int, nargs=4, default=OFFSETS,
                        help='four descending offsets before each visible loss')
    args = parser.parse_args()
    offsets = tuple(args.offsets)
    if (args.output.exists() or args.games < 1 or args.max_attempts < args.games
            or args.first_training_seed < 70000
            or tuple(sorted(set(offsets), reverse=True)) != offsets
            or min(offsets) < 4):
        parser.error('new archive, positive count and training-only seeds required')
    policy, config = load_policy(args.checkpoint/'model.safetensors')
    if (config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('eval_max_steps') != 0 or config.get('allow_enter')):
        raise ValueError('requires complete original own screen-only games')
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    files, games, skipped = [], [], []
    checkpoint_sha = sha256(args.checkpoint/'model.safetensors')
    for seed in range(args.first_training_seed, args.first_training_seed+args.max_attempts):
        if len(games) >= args.games:
            break
        frames, actions, rewards, result, events = record_game(policy, seed,
            tstates=config['tstates'], max_steps=0,
            observation_stride=config['observation_stride'])
        if (game_rank(result) is None or result['highest_stage'] != 1
                or result['missions_completed']):
            skipped.append(dict(seed=seed, reason='not a complete stage-one own game',
                                result=result))
            continue
        try:
            samples = visible_approaches(frames, events, offsets=offsets)
        except ValueError as error:
            skipped.append(dict(seed=seed, reason=str(error), result=result))
            continue
        verification = verify_policy_trace(args.checkpoint/'model.safetensors',
            frames, actions, rewards, result)
        if not verification['verified'] or verification['checkpoint_sha256'] != checkpoint_sha:
            raise RuntimeError('own complete-game replay did not verify')
        trace_hash = hashlib.sha256(frames.tobytes()+actions.tobytes()+rewards.tobytes()).hexdigest()
        games.append(dict(seed=seed, result=result, verification=verification,
                          trace_arrays_sha256=trace_hash,
                          visible_loss_frames=[item['visible_loss_frame'] for item in samples]))
        for item in samples:
            name = f"seed-{seed}-life-{item['life']}.npz"
            np.savez_compressed(args.output/name, screens=item['screens'])
            meta = {key: value for key, value in item.items() if key != 'screens'}
            meta.update(seed=seed, training_only=True, offsets=list(offsets),
                        npz_sha256=sha256(args.output/name),
                        source_checkpoint_sha256=checkpoint_sha,
                        full_game_trace_arrays_sha256=trace_hash)
            write_json((args.output/name).with_suffix('.json'), meta)
            files.append(name)
        disk_guard(args.output)
        print(json.dumps(dict(event='verified_own_game', seed=seed,
            selected_games=len(games), visible_loss_frames=games[-1]['visible_loss_frames'])),
            flush=True)
    if len(games) != args.games:
        raise RuntimeError('not enough eligible complete own games for early visual source')
    index = dict(source_checkpoint=str(args.checkpoint.resolve()),
        source_model_sha256=checkpoint_sha,
        source_state_sha256=sha256(args.checkpoint/'state.json'),
        source_sha256=sha256(Path(__file__)),
        game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
        offsets=list(offsets), selection='verified own complete training-game visible life losses',
        proposal_data_only=True, native_snapshot_read=False,
        hidden_state_read=False, action_target_read=False,
        games=games, skipped=skipped, files=files)
    write_json(args.output/'index.json', index)
    print(json.dumps(dict(event='finished', selected_games=len(games),
        skipped_games=len(skipped), early_life_samples=len(files))), flush=True)


if __name__ == '__main__':
    main()
