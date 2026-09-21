# TRS-80 Emulator + Games

A TRS-80 Model I emulator (Z80 CPU in C, thin Python bindings) with a few
classic games, drivable both interactively and programmatically.

**Your assignment lives in [GOALS.md](GOALS.md).** Read it first.

The reinforcement-learning implementation, training/evaluation commands, and
watch mode are documented in [TRAINING.md](TRAINING.md).

For **Obstacle Run / Missile Defense** (the existing `--game defense`), see
[DEFENSE.md](DEFENSE.md): controls, binary audit, a separate screen-only
environment, complete-game diagnostic results and portable replays. This new
game has not been trained yet; Breakdown's commands and models remain separate.

The committed screen-only model reached **level 5** in a complete validation
game: [watch the 278-point replay](results/level5/replay.html). On **100 fresh
complete test games**, mean score was **83.32**, median **72.5**, best **167**,
and highest level **3**; none reached level 5. This is a first-reach milestone,
not reliable level-5 play. See the [frozen model](models/breakdown-level5/README.md),
[test records](results/level5/trained.json), and [experiment notes](LEVEL5.md).

The original level-2 model and local replay remain unchanged. The
[public replay](https://breakdown-learned-replay.saschah.chatgpt.site) now shows
the verified 585-point all-eight-level win (site version 2). The level-5
replay remains a separate standalone HTML file.

The user-confirmed target is now **beat all eight original levels**, without
changing the game. The original ends after the eighth, so Level 10 is
unreachable; this is confirmed by the
[author](https://pski.net/breakdown-a-new-trs-80-game/) and a
[read-only audit of this exact game binary](results/level10/game-level-cap-audit.json).
A frozen policy has now **beaten all eight levels**: the 585-point winning
replay has all 49,159 neural actions verified, with a spare ball remaining.
Checkpoint comparison and the frozen winner's 100 fresh games are complete.
Fresh mean **278.75**, median **277.5**, best **583**, highest **Level 8**;
**zero verified wins** and **three unverified final-level endings**. This
demonstrates winning capability, not reliable winning. All processes have
exited, and the new training trial is preserved. Last-ball
Level-8 endings are conservatively reported as unverified, not assumed losses.
Historical comparisons and
validation-only progress are recorded in [LEVEL10.md](LEVEL10.md).
The selected winning reference reaches **Level 8 eleven times across 70 reused
complete games**, including **one verified victory**. Mean **298.94**, median
**280**, best **585**. This is validation progress, not a fresh success-rate
estimate or evidence that it wins reliably.

Stable, standalone replay links (updated only after neural-action verification):

- [Winning effort: 585 points / all eight levels](results/level10/best-effort/replay.html),
  all **49,159 actions** verified.
- [Selected model's winning replay](results/level10/best/replay.html),
  currently the same verified game; its 70-game results are above.

The frozen weights are preserved in [models/breakdown-all-eight](models/breakdown-all-eight/README.md).
See [the completed all-eight report](ALL_EIGHT.md) and
[all 100 fresh game records](results/level10/all-eight-fresh-test.json).

[View the illustrated 10-slide training presentation](https://breakdown-learned-replay.saschah.chatgpt.site/presentation.html):
screen-only learning, no action oracle, DQN → PPO, self-generated training aids,
the DeepMind Atari DQN comparison, and the verified result with its limitations.
An [offline HTML copy](results/training-presentation.html) and
[editable source/build instructions](presentation/README.md) are included.

Both directories retain versioned model/optimizer/configuration/validation and
replay bundles; older versions are never removed. The previous long run stopped
gracefully at 236M actions after a depth plateau. The within-level progress
experiment then stopped cleanly at **120,102,912** actions, with its latest
checkpoint and full log preserved. Its eight complete evaluation rounds did
not replace the selected model. See
[the supervision instructions](TRAINING.md#continuous-local-supervision).
The committed level-5 package and its test results above remain unchanged.
The earlier [598-point Level-8 effort](results/level10/supervised-best-effort-598-replay.html)
also remains preserved; a verified victory takes precedence over its higher score.

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
