# Cosmic Fighter: ready-to-launch learning setup

**Status: prepared, not training.** The user asked to stop before starting a
training process. No Cosmic optimizer update, smoke run or production run has
been launched. Defense remains paused and its best model/replay untouched.

## Original program and visible controls

`var/cosmic.cmd` is the original launcher image (SHA-256
`4999f2d853dddc0ee58198a03b597fd848d5011068037b1ed8f2926d218190e7`).
`main.py` already boots it with CLEAR and one-player selection. Its original
on-screen instructions say to move left/right, fire with F or Space, press D
to dock and refuel, and play until all ships are exhausted. They also mention
an extra ship every 10,000 points. The game can be opened for human play with
`venv/bin/python main.py --game cosmic`.

Independent headless boot checks confirmed that LEFT and RIGHT move the ship,
Space and F both fire, and NOOP/held Space games reach the visible
`Game Over Player 1` message. UP/DOWN did not move the initial ship in the
checked state. D is retained as its own learnable choice for the documented
docking/refueling behavior; it was not used to script a dock. The seven
learnable commands are NOOP, LEFT, RIGHT, Space, LEFT+Space, RIGHT+Space and
D. No menu, abort or reset key is available to the playing policy.

The top video-memory row visibly displays the score and reserve-ship `[` icons;
the second row displays the fuel meter. Both score and icons blink, so
`rl.cosmic.screen_info` treats blank frames as **unknown**, never zero, and
settles changed text before issuing score reward. A terminal message in the
original screen ends an episode; a time cap is explicitly a truncation, never
a complete game. The source image is hash-pinned. The policy's entire input
is four raw 16×64 video-memory frames. No nonvideo game RAM, scripted route,
demonstration, position label or shaped reward enters learning.

Ten complete non-learning random-control diagnostic games, seeds 100–109 at
100,000 T-states/action, scored **775 mean / 800 median / 1,600 best**. A
NOOP complete game scored 0; held Space scored 50. In each checked game,
the sum of displayed-score increments equaled the terminal visible score.
These are integration baselines, **not trained-policy results**. The native
program's instructions do not identify a finite final level or victory
screen; the initial objective is therefore improved complete-game score,
not a fabricated win threshold.

## Learner and result protocol

`rl.cosmic_dqn` is a separate seven-action dueling Double DQN using the
existing raw-screen encoder, prioritized compact own-experience replay and
five-step score returns. The reward is only positive change in the game's
displayed score (times a constant 0.01 optimizer-unit conversion). A visible
ship loss closes a training TD return; original-boot evaluation plays the
whole game. Training exploration is epsilon-random among the same seven
physical choices; greedy evaluation uses learned Q-values only. The model,
target network, Adam state and RNG are saved on a normal budget stop or
SIGTERM. A restart refills in-memory replay from new own play; it is not an
exact trajectory continuation.

Scheduled checks use ten fixed complete original-boot games. A new best is
chosen by complete-game mean score, then its highest-scoring game is recorded
as a portable HTML replay. Publication reloads frozen weights and reproduces
every neural action, reward and screen from original boot. `rl.cosmic_evaluate`
provides independent frozen-model checks and a separate non-promoting replay
bundle. Truncated evaluations cannot publish a best.

## Proposed bounded launch (requires user approval)

1. Run a **fresh** 65,536-action plumbing smoke with 8,192 replay slots,
   1,024 warmup actions and one fixed ten-game check at the end. Require
   finite updates, complete games, a verified replay and healthy disk. Never
   initialize production from this smoke.
2. If the smoke passes, run a separate **fresh** seed-73, 1,048,576-action
   pilot: eight emulator workers, 100,000 T-states/action, four visible frames,
   100,000 replay slots, 10,000 warmup, batch 64, one update per 16 aggregate
   actions, five-step return, gamma 0.997, epsilon 1.0→0.1 over 500,000
   aggregate actions, 5,000-action per-game cap. Fixed ten-game checks at
   262,144-action intervals use seeds 70000–70009. The cap never counts as a
   complete game. A separate 64-game set, seeds 80000–80063, is reserved for
   the frozen validation-selected model after the pilot, not checkpoint
   selection.
3. Protect each exact run with the 8 GiB disk watcher. Keep optimizer states,
   logs and intermediate replays in ignored local run directories, and back up
   any checkpoint needed for resumption outside Git before pruning it. Publish
   only a deliberately selected frozen model, evaluation and verified replay
   under the repository's artifact policy. Do not let score alone imply
   a victory screen or unseen-map generalization: the start seed changes the
   original timing, not the programmed enemies.

The setup and pure-environment tests can be checked now without training:

```sh
venv/bin/python -m unittest tests.test_cosmic tests.test_defense_disk_watch -v
```

When approved, the smoke command is:

```sh
venv/bin/python -m rl.cosmic_dqn \
  --run runs/cosmic-dqn-smoke-01 \
  --artifacts runs/cosmic-dqn-smoke-01-artifacts \
  --steps 65536 --capacity 8192 --warmup 1024 --eval-every 65536
```

The fresh pilot would use:

```sh
venv/bin/python -m rl.cosmic_dqn \
  --run runs/cosmic-dqn-pilot-01 \
  --artifacts results/cosmic/learned \
  --steps 1048576
```

After each trainer starts, launch the exact-process guard with:

```sh
venv/bin/python -m rl.defense_disk_watch \
  --run runs/cosmic-dqn-pilot-01 --min-free-gib 8 --interval 30
```

For a later independent frozen-model check:

```sh
venv/bin/python -m rl.cosmic_evaluate \
  runs/cosmic-dqn-pilot-01/step-000001048576/model.safetensors \
  --output results/cosmic/evaluation-80000.json \
  --replay-output results/cosmic/recheck-80000 \
  --games 64 --seed 80000
```

All commands above are instructions, not claims that training has started.
