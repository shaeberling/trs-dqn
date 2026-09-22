# TRS-80 Emulator + Games

A TRS-80 Model I emulator (Z80 CPU in C, thin Python bindings) with a few
classic games, drivable both interactively and programmatically.

**Your assignment lives in [GOALS.md](GOALS.md).** Read it first.

The reinforcement-learning implementation, training/evaluation commands, and
watch mode are documented in [TRAINING.md](TRAINING.md).

For **Obstacle Run / Missile Defense** (the existing `--game defense`), see
[DEFENSE.md](DEFENSE.md): controls, binary audit, a separate screen-only
environment, complete-game diagnostic results and portable replays. A fresh
PPO learner now runs via `python -m rl.defense_train` (20 actions by default,
21 with optional policy-selected Enter); Breakdown's
commands, model weights and results remain separate.
An independent value-learning alternative is available through
`python -m rl.defense_dqn`: Double DQN with prioritized own-experience replay,
complete-game greedy evaluation and verified replays. See its settings and
experimental results in [DEFENSE.md](DEFENSE.md).
Its optional `--bootstrap-heads 5` experiment uses per-game value-head
exploration with frozen random priors and greedy ensemble evaluation; it
does not add demonstrations, hidden-state inputs or reward bonuses.
Ordinary Defense DQN also supports `--exploration-max-repeat 64`: training-only
random action persistence, with ordinary learned greedy evaluation unchanged.
The default `1` preserves the original independent-action exploration. See
[the controlled comparison](DEFENSE.md#persistent-random-exploration) for its
motivation, rate calibration and limitations.
Optional `--curriculum-boot-epsilon` gives reserved boot-only training workers
a separate fixed exploration rate; other workers retain the normal schedule.
This tests exploration allocation while preserving screen-only inputs,
score-only rewards and ordinary greedy from-boot evaluation.
Optional `--greedy-trace-cut` tests current-policy cuts in scalar DQN's
multi-step training targets. It requires compact replay, retains only own
visible trajectories, and leaves acting/evaluation unchanged. See the
[controlled calibration](DEFENSE.md#current-policy-trace-cutting-calibration).
Optional `--spr-weight .1` adds an SPR-inspired auxiliary prediction loss on
the learner's own visible trajectories. Acting remains ordinary greedy DQN;
future screens are training targets only, not policy inputs or extra rewards.
See [the visual-prediction experiment](DEFENSE.md#auxiliary-visual-prediction).
Optional `--inverse-weight .01` instead trains an auxiliary classifier on the
learner's own adjacent screen pairs and recorded actions. It adds no intrinsic
reward and changes no acting inputs. See [the inverse-action experiment](DEFENSE.md#inverse-action-representation-experiment).
Both Defense trainers support `--observation-stride` to space their four
visible input frames independently of action duration (default 1 preserves
existing models). Checkpoint evaluation and verified replays retain that setting.
Ordinary Defense DQN also has optional resets to opaque states reached during
its own training; evaluation always starts from boot. See the reset controls
and experimental limitations in [DEFENSE.md](DEFENSE.md).
Its default score-based archive selection is unchanged; `--curriculum-cells
screen` optionally uses the existing coarse visible-screen fingerprints, with
explicit archive-capacity controls and no extra reward or policy input.
Optional DQN `--curriculum-trigger life-loss --curriculum-lookback 64` instead
archives actual own states 64 decisions before a visible ship loss. This is a
training-reset experiment, not a collision oracle or an evaluation-time aid;
the default remains progress-triggered archives. See
[the own-loss reset experiment](DEFENSE.md#own-loss-triggered-reset-experiment).
Optional `--curriculum-restored-life-only` ends already-restored training
segments at their first visible ship loss. It requires life-terminal learning;
boot games and complete-game evaluation remain unchanged. See
[the focused-practice comparison](DEFENSE.md#restored-life-only-practice).
Optional DQN `--exploration-actions stage1-balanced` changes persistent random
exploration to a fixed distribution over command groups, reducing duplicate
stage-1 firing choices. All 20 learned actions and greedy evaluation remain
unchanged; see [the sampler experiment](DEFENSE.md#balanced-command-group-exploration).
Experimental `--quantiles 32` learns distributions of score returns; optional
`--quantile-exploration-power 1.5` changes training action selection only.
Evaluation still uses greedy mean values. An own scalar DQN can initialize
this model with `--init-from-dqn`; this starts fresh Adam, not optimizer resume.
Separate `rl.defense_evaluate --quantile-power 1.5` probes can play and verify
the alternative learned criterion without changing training or the shared best.
`--compact-replay` optionally shares byte-identical visible frames in DQN's
training buffer, preserving exact observations and sampling while reducing
screen-storage duplication. It works with ordinary and bootstrap DQN.
Defense PPO also supports optional learned screen-history memory through
`--recurrent-hidden 128`: a residual GRU, sequence training and independently
reset evaluation memory. This is experimental, not a verified stage clear;
see [the recurrent experiment](DEFENSE.md#recurrent-screen-history-experiment).
Its optional `--freeze-recurrent-base` trains only the memory and residual heads
over an own learned, frozen policy; it requires compatible full-policy
initialization and does not introduce a scripted controller or demonstrations.

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
