# TRS-80 Emulator + Games

A TRS-80 Model I emulator (Z80 CPU in C, thin Python bindings) with a few
classic games, drivable both interactively and programmatically.

**Your assignment lives in [GOALS.md](GOALS.md).** Read it first.

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

export LD_LIBRARY_PATH=$(pwd)
make
```

`make` builds `libtrs.so` (the emulator core) into the repo root; the Python
bindings load it from the current working directory, so run everything from
the repo root.

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
