"""Read-only screen/action comparison of losses in verified own-policy replays.

White-screen flashes are visual alignment markers, not collision labels.
Nothing here is imported by training or used to choose actions or rewards.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_learning import game_rank, sha256


def loss_windows(frames, actions, rewards, metadata):
    names = action_names()
    if (frames.dtype != np.uint8 or frames.shape != (len(actions)+1, 16, 64)
            or actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer)
            or np.any(actions < 0) or np.any(actions >= len(names))
            or rewards.shape != actions.shape or not np.isfinite(rewards).all()
            or np.any(rewards < 0)):
        raise ValueError('invalid screen/action/reward trace')
    result = metadata['result']
    if game_rank(result) is None or result['highest_stage'] != 1:
        raise ValueError('requires a complete stage-one trace')
    cumulative = np.concatenate(([0.], np.cumsum(rewards, dtype=np.float64)))
    if cumulative[-1] != result['score']:
        raise ValueError('visible score increments disagree with final score')
    windows, previous, previous_score = [], 0, 0
    for event in metadata['events']:
        if not event.get('life_lost'):
            continue
        stop = event['frame']
        if (not isinstance(stop, int) or not previous < stop <= len(actions)
                or cumulative[stop] != event['score']):
            raise ValueError('invalid visible loss event')
        start = max(previous, stop-128)
        # Solid white semigraphics cell, excluding HUD. Death flashes fill
        # most of the viewport; require >50%, not a policy/game-state signal.
        white = (frames[start:stop+1, 1:] == 0xBF).mean(axis=(1, 2))
        candidates = np.flatnonzero(white > .5)
        flash = start+int(candidates[0]) if len(candidates) else None
        # Never silently call the visible loss frame the collision frame.
        decision_stop = flash if flash is not None else stop
        decision_start = max(previous, decision_stop-64)
        counts = np.bincount(actions[decision_start:decision_stop], minlength=len(names))
        positive = np.flatnonzero(rewards[previous:stop])+previous
        windows.append(dict(
            life=len(windows)+1, previous_visible_loss_frame=previous,
            visible_loss_frame=stop, visible_score_at_loss=event['score'],
            score_gained_this_life=event['score']-previous_score,
            first_major_white_flash_in_last_128_frames=flash,
            flash_to_visible_loss_actions=stop-flash if flash is not None else None,
            action_window=[decision_start, decision_stop],
            action_counts=dict(zip(names, counts.tolist(), strict=True)),
            movement_action_fraction=float(counts[1:9].sum()/counts.sum()),
            recent_visible_score_increments=[dict(frame=int(i+1), points=float(rewards[i]))
                                            for i in positive[-8:]],
            panel_frames=[max(previous, decision_stop-offset) for offset in (64, 32, 8, 1)]))
        previous, previous_score = stop, event['score']
    if len(windows) != 4 or previous != len(actions):
        raise ValueError('requires four observed ship losses ending the game')
    return windows


def analyze(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle/'manifest.json').read_text())
    required = ('model.safetensors', 'state.json', 'trace.npz', 'verification.json')
    hashes = manifest['hashes']
    for name in required:
        if name not in hashes or sha256(bundle/name) != hashes[name]:
            raise ValueError('replay checksum mismatch: '+name)
    verification = json.loads((bundle/'verification.json').read_text())
    if not verification.get('verified') or verification['checkpoint_sha256'] != hashes['model.safetensors']:
        raise ValueError('requires an originally verified own-policy replay')
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        frames, actions, rewards = trace['frames'], trace['actions'], trace['rewards']
        metadata = json.loads(str(trace['metadata']))
    if (metadata.get('action_names') != list(action_names())
            or metadata.get('checkpoint_sha256') != hashes['model.safetensors']
            or verification['verified_actions'] != len(actions)):
        raise ValueError('trace provenance/action profile mismatch')
    windows = loss_windows(frames, actions, rewards, metadata)
    for name in required:
        if sha256(bundle/name) != hashes[name]:
            raise RuntimeError('source changed during analysis: '+name)
    return dict(bundle=str(bundle), source_hashes={name: hashes[name] for name in required},
                result=metadata['result'], tstates=metadata['tstates'],
                diagnostic_only=True, promotion_eligible=False, native_reexecution=False,
                training_data_written=False, parameter_updates=0,
                original_native_verification=verification, lives=windows), frames, actions


def render_sheet(path, label, report, frames, actions):
    """Render unaltered recorded screen bytes with the emulator's font."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('var/AnotherMansTreasureMIII64C.ttf', 26)
    glyphs = []
    for code in range(256):
        glyph = Image.new('RGB', (8, 26), 'black')
        ImageDraw.Draw(glyph).text((0, -1), chr(0xE000+code), font=font, fill='white')
        glyphs.append(glyph)
    sheet = Image.new('RGB', (2048, 4*464+40), '#18202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), label+' | Rows: lives. Columns: 64 / 32 / 8 / 1 actions before alignment marker, NOT collision.', fill='white')
    for row, life in enumerate(report['lives']):
        for col, index in enumerate(life['panel_frames']):
            x, y = col*512, 40+row*464
            draw.text((x+6, y+3), f"Life {life['life']} | frame {index} | next: {action_names()[int(actions[index])]}", fill='white')
            marker = 'flash' if life['first_major_white_flash_in_last_128_frames'] is not None else 'loss (no flash sampled)'
            draw.text((x+6, y+20), f"Life total {life['score_gained_this_life']} | marker: {marker}", fill='white')
            for r, cells in enumerate(frames[index]):
                for c, code in enumerate(cells):
                    sheet.paste(glyphs[int(code)], (x+8*c, y+46+26*r))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True, help='new diagnostic directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing diagnostic output')
    analyses = [analyze(bundle) for bundle in args.bundles]
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, (report, frames, actions) in enumerate(analyses):
        filename = f'policy-{index+1}-losses.png'
        report['screen_sheet'] = filename
        render_sheet(args.output/filename, str(args.bundles[index]), report, frames, actions)
        reports.append(report)
    output = dict(probe_source_sha256=sha256(Path(__file__)), reports=reports,
                  limitations=[
                      'Selected best replays, not representative or matched-seed evaluations.',
                      'Equal score does not prove equal course progress or identical collisions.',
                      'White flashes are visible alignment heuristics, not physical collision timestamps.',
                      'Frames include original game animations; action counts do not measure motion.',
                      'No hidden RAM, route oracle, training examples, or policy changes.'])
    (args.output/'report.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps([dict(bundle=r['bundle'], life_scores=[l['score_gained_this_life'] for l in r['lives']])
                      for r in reports], indent=2))


if __name__ == '__main__':
    main()
