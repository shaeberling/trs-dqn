"""TRS-80 emulator launcher.

Play mode opens a window and lets a human play with the keyboard:

    python main.py -m Play [--game breakdown|cosmic|defense]

See GOALS.md for the project assignment and a tour of the emulator's
Python API (headless stepping, key injection, screen and memory access).
"""

import argparse

from trs import TRS, Key

# Each game config names the program image to load and a boot sequence for
# programmatic use via trs.boot(): integer entries run the CPU for that many
# T-states, Key entries hold a key down while booting continues.
CONFIGS = {
    "breakdown": {
        "name": "breakdown",
        "cmd": "var/breakdown.cmd",
        "boot": [1000000, 1000000, 1000000, Key.SPACE, 1000000, 1000000,
                 1000000, 1000000, 1000000, 1000000, 800000],
    },
    "cosmic": {
        "name": "cosmic",
        "cmd": "var/cosmic.cmd",
        "boot": [1000000, Key.CLEAR, Key._1, 1000000, 1000000, 1678000],
    },
    "defense": {
        "name": "defense",
        "cmd": "var/defense.cmd",
        "boot": [1000000],
    },
}


def main():
    parser = argparse.ArgumentParser(description="TRS-80 emulator")
    parser.add_argument("-m", "--mode", default="Play", choices=["Play"],
                        help="Play: run the game in a window at original "
                             "speed for a human player")
    parser.add_argument("--game", default="breakdown",
                        choices=sorted(CONFIGS))
    args = parser.parse_args()

    trs = TRS(CONFIGS[args.game], 1, 20.0, False)
    trs.run_cpu()
    trs.mainloop()


if __name__ == "__main__":
    main()
