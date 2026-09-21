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
        (0x7432, "21 a9 76 22 9b 81 3e 06 32 9a 81",
         "Stage 1 initializes its obstacle-stream pointer to 76A9 and row-update countdown to six."),
        (0x747E, "cd 07 ac cd de 7f 2a 9b 81 7e 32 0f 70 b7 c8",
         "After the stage-1 update, copy the next stream byte to the outcome flag; return success when it is zero, not at a score threshold."),
        (0x80F2, "3a 9a 81 3d 32 9a 81 20 42 3e 06 32 9a 81",
         "Decrement row-update countdown; decode a new stream row only at zero, then reload six."),
        (0x811E, "2a 9b 81 11 00 45 7e b7 28 0c 3d 28 1a 3d 28 2a 3d 28 34 3d 28 14 23 22 9b 81 f1 c1 d1 e1 c9 21 46 00 cd 11 67 18 f3 23 18 dc 3e bf 1e 00 12 1e 3f 12 13 12 1e 7f 12 23 18 cc 23 5e 23 46 23 7e 23 12 13 10 fc 18 bf e5 d5 21 06 5e 11 09 00 06 1d 7e b7 28 03 19 10 f9 2b e5 c1 d1 e1 23 7e 02 03 3e 01 02 03 23 7e 02 23 03 7a 02 03 7e 02 af 03 02 03 02 03 02 03 02 23 c3 24 81",
         "Stream decoder: zero ends a row; commands 1/4 occupy one byte, and commands 2/3 consume three parameter bytes. Store pointer just past the row terminator."),
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
    # Decode the immutable program asset only. These positions/counts are not
    # exposed to the environment, a policy, reward, or curriculum selection.
    cursor, rows, command_counts = 0x76A9, 0, {str(op): 0 for op in range(5)}
    while image.data[cursor]:
        rows += 1
        while True:
            op = image.data[cursor]
            if op not in range(5):
                raise ValueError(f"Unknown stage-1 stream command at {cursor:04X}")
            command_counts[str(op)] += 1
            cursor += 4 if op in (2, 3) else 1
            if cursor > 0x7BD8:
                raise ValueError("Stage-1 stream exceeds its original data boundary")
            if op == 0:
                break
    if (rows, cursor) != (126, 0x7BD8):
        raise ValueError("Stage-1 course differs from the audited stream")
    success = bytes(image.data[0xA36F:0xA36F+21]).decode("ascii")
    loss = bytes(image.data[0x7095:0x7095+24]).decode("ascii")
    if "YOU did it" not in success or "GAME OVER PLAYER 1" not in loss:
        raise ValueError("Ending text contract mismatch")
    return dict(game="Obstacle Run / Missile Defense", game_sha256=digest,
                cmd_entry=f"0x{entry:04X}", method="read-only CMD decoding and instruction-byte contracts",
                original_stage_cycle=[1, 2, 3, 1], success_text=success, loss_text=loss,
                stage_one_course=dict(start="0x76A9", end_marker="0x7BD8", rows=rows,
                                      command_counts=command_counts,
                                      progression="Automatic obstacle-stream exhaustion; not a score threshold",
                                      policy_progress_measured=False),
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
