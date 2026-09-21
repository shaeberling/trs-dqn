"""Reproducible static ending audit of the supplied CMD; never runs a policy."""

import argparse
import hashlib
import json
from pathlib import Path

from trs.cmd import CMD
from .defense import GAME_SHA256


class Image:
    def __init__(self):
        self.data = bytearray(65536)

    def poke(self, address, value):
        self.data[address] = value


def audit(path=Path("var/defense.cmd")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GAME_SHA256:
        raise ValueError("Not the audited game")
    image = Image()
    entry = CMD(image).load(str(path))
    checks = [
        (0x6F97, "3a ed 7c 3d 28 08 3d 28 0a cd 02 a0 18 08 cd d0 73 18 03 cd e4 81",
         "Read P1 stage. Stage 1 calls 73D0; stage 2 calls 81E4; stage 3 calls A002."),
        (0x6FB6, "3a ed 7c 3c 32 ed 7c fe 04 20 c5 3e 01 32 ed 7c 18 be",
         "After success increment P1 stage. Compare 4; if equal reset stage to 1. Both branches return to 6F86."),
        (0xA016, "3a 0f 70 b7 cc f9 a2",
         "Stage 3 calls completion animation A2F9 only when outcome flag is zero (success)."),
        (0xA358, "21 6f a3 11 16 3e 01 15 00 ed b0",
         "Copy 21 bytes of completion text from A36F to visible screen 3E16, row 8 column 22."),
        (0x7070, "21 95 70 11 d4 3d 01 18 00 ed b0",
         "Loss routine copies the 24-character GAME OVER message to row 7 column 20."),
    ]
    verified = []
    for address, expected, meaning in checks:
        expected = bytes.fromhex(expected)
        actual = bytes(image.data[address:address+len(expected)])
        if actual != expected:
            raise ValueError(f"Instruction contract mismatch at {address:04x}")
        verified.append(dict(address=f"0x{address:04X}", bytes=actual.hex(" "), interpretation=meaning))
    success = bytes(image.data[0xA36F:0xA36F+21]).decode("ascii")
    loss = bytes(image.data[0x7095:0x7095+24]).decode("ascii")
    if "YOU did it" not in success or "GAME OVER PLAYER 1" not in loss:
        raise ValueError("Ending text contract mismatch")
    return dict(game="Obstacle Run / Missile Defense", game_sha256=digest,
                cmd_entry=f"0x{entry:04X}", method="read-only CMD decoding and instruction-byte contracts",
                original_stage_cycle=[1, 2, 3, 1], success_text=success, loss_text=loss,
                successful_gameplay_observed=False,
                conclusion="Three-stage mission cycles; no finite last-level exit in the audited dispatcher. A loss GAME OVER is not a mission victory.",
                remaining="Observe the successful completion screen and subsequent stage wrap in an unmodified learned-policy playthrough.",
                checks=verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(), indent=2)+"\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
