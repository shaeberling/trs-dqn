"""Render selected recorded screen frames for post-hoc visual diagnosis only."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .env import screen_info
from .inspect_policy import decode_replay


def render_frames(replay, indices, output):
    metadata, frames, _ = decode_replay(Path(replay).read_text())
    if not indices or any(i < 0 or i >= len(frames) for i in indices):
        raise ValueError('frame indices must be within the recorded game')
    columns = min(3, len(indices))
    sheet = Image.new('RGB', (columns*512, ((len(indices)+columns-1)//columns)*450), 'black')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('var/AnotherMansTreasureMIII64C.ttf', 26)
    for panel, index in enumerate(indices):
        x, y = panel % columns*512, panel//columns*450
        info = screen_info(frames[index])
        draw.text((x+5, y+5), f"Frame {index} | Level {info['level']} | Score {info['score']}", fill='white')
        for row, cells in enumerate(frames[index]):
            for col, code in enumerate(cells):
                draw.text((x+col*8, y+30+row*26-1), chr(0xe000+int(code)), font=font, fill='white')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render_frames(args.replay, args.frames, args.output)


if __name__ == '__main__':
    main()
