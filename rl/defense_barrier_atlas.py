"""Read-only atlas of visible screens before own stage-one life losses.

This is a diagnostic over recorded screen/action traces, never a trainer,
reward, policy input, or physical collision detector. The visible loss marker
can occur after the collision, so its time is only an alignment aid.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_learning import sha256, write_json


OFFSETS = (64, 32, 8, 1)


def collect(archive):
    archive = Path(archive)
    index = json.loads((archive/'index.json').read_text())
    entries, prior = [], {}
    for filename in index['files']:
        path = archive/filename
        meta = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != meta['npz_sha256']:
            raise ValueError('source checksum mismatch: '+filename)
        source = meta['source']
        seed, life = source['seed'], source['life']
        if life != len([entry for entry in entries if entry['seed'] == seed])+1:
            raise ValueError('nonconsecutive visible life-loss records')
        with np.load(path, allow_pickle=False) as arrays:
            screens = arrays['screens'].copy()
            actions = arrays['actions'].copy()
        if (screens.shape != (128, 16, 64) or screens.dtype != np.uint8
                or actions.shape != (128,) or np.any(actions >= len(action_names()))):
            raise ValueError('unexpected own-screen source shape')
        previous = prior.get(seed, 0)
        score = source['visible_loss_score']-previous
        if score < 0:
            raise ValueError('visible score decreased')
        prior[seed] = source['visible_loss_score']
        recent = actions[-64:]
        counts = np.bincount(recent, minlength=len(action_names()))
        entries.append(dict(seed=seed, life=life, source=filename,
            source_npz_sha256=meta['npz_sha256'], displayed_life_score=score,
            visible_loss_frame=source['visible_loss_frame'],
            selection='128 own neural actions before visible loss',
            last_64_pure_right_actions=int(counts[[4, 6, 8]].sum()),
            last_64_pure_left_actions=int(counts[[3, 5, 7]].sum()),
            last_64_pure_movement_actions=int(counts[1:9].sum()),
            last_64_firing_or_side_fire_actions=int(counts[9:].sum()),
            panels=[dict(actions_before_visible_loss=offset,
                screen_index=127-offset, next_action=action_names()[int(actions[128-offset])])
                for offset in OFFSETS],
            screens=screens))
    if len(entries) != 48 or len(prior) != 12 or any(
            sum(entry['seed'] == seed for entry in entries) != 4 for seed in prior):
        raise ValueError('expected twelve complete four-life own-policy training games')
    return index, entries


def render(path, entries, life):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('var/AnotherMansTreasureMIII64C.ttf', 26)
    glyphs = []
    for code in range(256):
        glyph = Image.new('RGB', (8, 26), 'black')
        ImageDraw.Draw(glyph).text((0, -1), chr(0xE000+code), font=font, fill='white')
        glyphs.append(glyph)
    selected = [entry for entry in entries if entry['life'] == life]
    sheet = Image.new('RGB', (2048, len(selected)*464+40), '#18202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), 'Original neural training games; columns 64 / 32 / 8 / 1 actions before VISIBLE LOSS, not collision', fill='white')
    for row, entry in enumerate(selected):
        for col, panel in enumerate(entry['panels']):
            x, y = col*512, row*464+40
            draw.text((x+6, y+3),
                f"Seed {entry['seed']} | life {life} | -{panel['actions_before_visible_loss']} | next {panel['next_action']}",
                fill='white')
            draw.text((x+6, y+20),
                f"Displayed life points {entry['displayed_life_score']} | visible-loss alignment only", fill='white')
            for r, cells in enumerate(entry['screens'][panel['screen_index']]):
                for c, code in enumerate(cells):
                    sheet.paste(glyphs[int(code)], (x+8*c, y+46+26*r))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite diagnostic output')
    index, entries = collect(args.archive)
    args.output.mkdir(parents=True, exist_ok=False)
    for life in range(1, 5):
        render(args.output/f'life-{life}-screens.png', entries, life)
    scores = [entry['displayed_life_score'] for entry in entries]
    count = sum(2500 <= score <= 2650 for score in scores)
    right = sum(entry['last_64_pure_right_actions'] for entry in entries)
    left = sum(entry['last_64_pure_left_actions'] for entry in entries)
    report = dict(source_archive=str(args.archive.resolve()),
        source_index_sha256=sha256(args.archive/'index.json'),
        diagnostic_source_sha256=sha256(Path(__file__)),
        training_data_written=False, parameter_updates=0,
        policy_or_reward_inputs_changed=False, exact_collision_time_known=False,
        sampled_games=len(index['games']), visible_losses=len(entries),
        displayed_life_score_histogram=dict(sorted(Counter(scores).items())),
        losses_in_2500_to_2650_band=count,
        last_64_pure_right_actions=right, last_64_pure_left_actions=left,
        entries=[{key: value for key, value in entry.items() if key != 'screens'} for entry in entries],
        panels=[f'life-{life}-screens.png' for life in range(1, 5)],
        limitations=['Selected self-play training games, not a random sample of all policies.',
                     'Displayed points are not an obstacle coordinate or passage marker.',
                     'Visible loss can lag physical collision; panels do not prove wall versus projectile death.'])
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(games=len(index['games']), losses=len(entries),
        score_band=count, last_64_pure_right=right, last_64_pure_left=left)))


if __name__ == '__main__':
    main()
