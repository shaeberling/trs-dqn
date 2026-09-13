# TRS-80 Emulator + Games

A TRS-80 Model I emulator (Z80 CPU in C, thin Python bindings) with a few
classic games, drivable both interactively and programmatically.

**Your assignment lives in [GOALS.md](GOALS.md).** Read it first.

The reinforcement-learning implementation, training/evaluation commands, and
watch mode are documented in [TRAINING.md](TRAINING.md).

The trained screen-only agent has reached **level 5** in a complete validation
game: [watch the 278-point replay](results/level5/replay.html). On **100 fresh
complete test games**, mean score was **83.32**, median **72.5**, best **167**,
and highest level **3**; none reached level 5. This is a first-reach milestone,
not reliable level-5 play. See the [frozen model](models/breakdown-level5/README.md),
[test records](results/level5/trained.json), and [experiment notes](LEVEL5.md).

The original level-2 model and
[88-point public replay](https://breakdown-learned-replay.saschah.chatgpt.site)
remain unchanged. The new level-5 replay is a separate standalone HTML file.

## Setup

> **NOTE**: Use Python 3.12. Check out the repository recursively — the Z80
> core is a git submodule.

```bash
git clone --recursive <repo url>
cd <repo>

# If you forgot --recursive:
git submodule init && git submodule update

python3.12 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

make
```

`make` compiles the existing emulator core into `libtrs.so` in the repo root.
The Python bindings resolve this library by its repository path on macOS and
Linux. Run commands from the repo root so the game and font assets are found.

## Playing a game yourself

```bash
python main.py -m Play --game breakdown
```

A window opens running the game at original TRS-80 speed. Use the arrow keys
and space bar.

## Hardware being emulated

| Spec | Value |
|---|---|
| CPU | Zilog Z80 @ 1.77408 MHz |
| RAM | 64 KB, fully accessible from Python |
| Display | 64×16 character cells; graphics characters give 128×48 pixels |
| Video memory | 1 KB at address 0x3C00 |
| Keyboard | Memory-mapped matrix at 0x3800 |

The Python API for driving the emulator headless (stepping the CPU, pressing
keys, capturing the screen, reading memory) is documented in
[GOALS.md](GOALS.md).

## Repo layout

| Path | Purpose |
|---|---|
| `native/trs.c` + `libz80/` | Emulator core (C) |
| `trs/` | Python bindings: CPU, RAM, keyboard, screenshot, window UI |
| `var/*.cmd` | Game program images (Breakdown, Cosmic Fighter, Defense) |
| `main.py` | Launcher with the game boot configurations |
