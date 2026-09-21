# Obstacle Run / Missile Defense: integration and training feasibility

The user-confirmed new game is the executable in `Missile_Defense.zip`.
It identifies itself as **Obstacle Run**, by Arno Puder (1983/84), and is
already present as `var/defense.cmd`. No emulator rebuild, disk controller,
new ROM, binary patch or duplicate game asset is needed.

Status: **screen-only PPO training is running independently of Breakdown**, with
parallel emulator workers, resumable checkpoints, complete-game validation and
automatic verified best-effort replays. See the commands and monitoring paths
below. A successful mission has not yet been verified.
Breakdown's frozen models, published site and results are unchanged. Shared
network/sampler code now supports configurable action counts while preserving
the original six-action defaults.

## Play and inspect

```sh
venv/bin/python main.py -m Play --game defense
```

Allow the animated title to appear. Enter shows instructions; **F1** maps to
the TRS-80 CLEAR key and starts player selection. Press **1** for one player.
Stage introductions finish automatically (Enter can skip them during manual
play). Arrow keys move, Space fires. F2 is BREAK.

| Input | Effect |
|---|---|
| Eight arrow directions, including diagonals | Movement |
| Space | Forward fire |
| Left + Right together | Side fire in stage 1 |
| Left + Right + Space | Both firing modes in stage 1 |
| CLEAR + BREAK (F1 + F2) | Abort to title; excluded from policy actions |

One important assembly-level distinction: stage 1 compares the **whole** key
byte for movement, so adding Space to an arrow stops movement while firing.
Stages 2 and 3 mask off Space before decoding movement and allow moving while
firing. The environment therefore offers **20 fixed actions**: nine movement/
no-op choices, those nine with Space, and the two side-fire combinations.
Some actions have equivalent effects in a particular stage. There is no
stage-aware controller choosing or overriding policy actions.

An optional **21st action, Enter**, is available with `--allow-enter`. The
neural policy alone chooses whether to press it; there is no automatic intro
skipper. It can dismiss stage introductions, but it cannot trigger CLEAR+BREAK
or select a new game. Existing 20-action checkpoints keep their original
action mapping and timing. A different action profile requires a fresh run,
not loading an incompatible optimizer/head into an existing checkpoint.

## Evidence and game identity

The supplied ZIP contains `command.CMD` and `disk_0.dmk`, not assembly source.
The analysis below is a read-only **disassembly of the original Z80 binary**.
`command.CMD` is byte-for-byte identical to `var/defense.cmd` (24,704 bytes).
The disk's active `DEFENSE.CMD` file was independently extracted in memory:
it is identical too. All 720 data-sector CRCs checked successfully.

Executable SHA-256:
`9b887e46223eb2ce9d2ada2a7588435a90be51f9e5326a592193faeafa322644`

DMK SHA-256:
`31bd3aca846f61b34ed60242d7404d4bad0f03cae67512fb74e0942932f8e801`

The CMD enters at `0xA870`. Static code addresses below are engineering audit
references, **not policy inputs or reward sources**.
The [reproducible ending audit](results/defense/game-ending-audit.json) checks
the instruction bytes directly; rerun with `venv/bin/python -m rl.defense_audit`.

| What the code establishes | Relevant addresses |
|---|---|
| CLEAR starts; one/two-player selection reads number-key matrix | `0x6F16`, `0x6F23`, `0x6F3A` |
| Four ships per player; one-player mode disables player 2 | `0x6F6F` |
| Three stage dispatches; increment stage, then wrap 4 to 1 | `0x6F86`–`0x700E` |
| Stage 1 firing and exact-byte movement comparisons | `0x758B`, `0x75AC`, `0x75E4` |
| Stage 2 movement masks Space; forward fire tests bit 7 | `0x8518`, `0x85B0` |
| Stage 3 movement masks Space; forward fire tests bit 7 | `0xA709`, `0xA7C4` |
| CLEAR + BREAK abort | `0x7683` |
| Decrement the current player's ships; draw game over at zero | `0x7010`, `0x7070` |
| Seven-digit score addition; periodic survival points | `0x7D86`, `0xABEB` |
| HUD formats score and `*` ship markers, then copies to video | `0x7DFD`–`0x7EFC` |
| Successful stage 3 calls congratulations animation | `0xA016`–`0xA01A`, `0xA2F9` |
| Writes `YOU did it` on row 8 | `0xA358`–`0xA361` |

## Observations, rewards and outcomes

`rl.defense.DefenseEnv` has the same basic reset/step shape as Breakdown but
independent game logic and action IDs. The observation is four raw `uint8`
16×64 screen frames: only video memory `0x3C00`–`0x3FFF`. The current game's
private score, ship coordinates, collision flags, random generator, stage
variables and off-screen graphics buffer are **not read by the environment**.

Booting uses CLEAR and 1 only before gameplay, waiting for the actual player
prompt. A seeded title-timing offset provides deterministic reset variation;
it is not a claim of independent, randomized game layouts. No automatic firing,
steering, life recovery, expert demonstrations or action oracle is supplied.

Visible signals:

- **Score:** leftmost 16 columns of row 0; variable-width digits followed by
  ships. The centered number is the high score, not the player's score.
- **Ships:** `*` characters following the score; initially four. A lost ship
  restarts that stage automatically. Blanking/scrolling screens mean unknown,
  not zero lives. Life-loss flags may lag the collision until the HUD updates.
- **Stage:** distinctive intro sentences: “You are now entering the”, “Find
  your way through the”, and “Now at last you enter the”. The last observed
  stage is retained during gameplay; it is not an always-visible level counter.
- **Mission completion:** `YOU did it` on row 8 (substring at video `0x3E1B`).
  Generic `CONGRATULATIONS` alone is insufficient: high-score entry also uses it.
- **Loss/episode end:** `GAME OVER PLAYER 1`, row 7 at `0x3DD7`.

The score deliberately blinks. Changed HUD copies are allowed to settle with
the **same chosen keys**. At game over, the original death animation must
finish updating the zero-ship score before reward is finalized: stopping at
the first GAME OVER can miss the last 20 points. The wrapper waits only for
that visible update, then stops. All reward is visible score difference;
there are no bonuses for lives, stages or mission detection.

The game loops its three stages. A defensible first achievement is **complete
one three-stage mission**, witnessed by `YOU did it`, while a full evaluation
game continues until all ships are gone. Higher score alone does not prove
mission completion. A diagnostic action cap is reported as **truncated**, never
as a win or complete loss.

## Verified smoke tests and replays

| Controller (not learned) | Complete games | Mean | Median | Best | Highest stage |
|---|---:|---:|---:|---:|---:|
| Uniform random actions, seeds 0–9 | 10 | 286 | 280 | 320 | 1 |
| No input, seeds 100–109 | 10 | 280 | 280 | 280 | 1 |

All 20 diagnostic games detected four ship losses and ended at game over.
No mission was completed. These are integration baselines, not evidence of
learning. Survival itself earns points, so compare stage/mission progress as
well as score when evaluating a future model.

A separate read-only engineering diagnostic compared final visible totals
with the internal score for 20 random games, exposing the terminal-redraw
issue described above and confirming the fix. Those internal reads are not
in the environment, baseline recorder or any policy/reward path; no game
memory was patched and no training examples were produced from them.

- [Random-action replay](results/defense/random-baseline/replay.html): 1,603
  actions; every resulting screen and reward checked by re-execution.
- [No-input replay](results/defense/noop-baseline/replay.html): 1,535 actions,
  likewise checked.
- Full records: each directory's `report.json`; compressed screen/action/
  reward trace in `trace.npz`; no-input [screen samples](results/defense/noop-baseline/screens.png).

Reproduce into a **new directory** (existing evidence is never overwritten):

```sh
venv/bin/python -m rl.defense_smoke --output runs/defense-random-check --games 10 --seed 0
venv/bin/python -m rl.defense_smoke --output runs/defense-noop-check --games 10 --seed 100 --policy noop
venv/bin/python -m unittest tests.test_defense -v
node tests/test_defense_replay.js
```

## Is there enough information to train?

**Yes.** Inputs, keyboard actions, actual score reward, resets and loss boundaries
are established and exercised in the real emulator. The separate environment
is ready for a learner adapter:

```python
from rl.defense import DefenseEnv, ACTIONS

env = DefenseEnv(seed=7, max_steps=0)  # 0 means no artificial episode cap
try:
    observation = env.reset()
    # action = your_new_policy(observation), integer in [0, len(ACTIONS))
    observation, reward, terminated, truncated, info = env.step(0)
finally:
    env.close()
```

The new learner is a fresh 20-action MLX PPO policy on this Mac. It reuses the
screen encoder/optimizer implementation, **not Breakdown's weights, six-action
head, curriculum, level-ranking rules or game-specific evaluation**. The existing
`rl.ppo` / `rl.train` CLI remains Breakdown-only. Defense has its own command:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-ppo-01 \
  --artifacts results/defense/learned --envs 32 --rollout 128 \
  --batch-size 512 --epochs 4 --eval-every 100000 --eval-games 10 --eval-envs 10
```

The command defaults to unlimited training and complete, uncapped episodes.
Reward is the actual visible score delta multiplied by a fixed 0.01 for the
optimizer's units; there is no shaping or clipping. Ship loss is not a separate
learning terminal by default. SIGINT/SIGTERM saves `latest` and exits cleanly.
Resume with `--resume runs/defense-ppo-01/latest`; the optimizer and policy RNG
are restored, while emulator episodes restart from boot (not exact trajectory
continuation). Evaluation uses fixed validation seeds 10000–10009; do not use
fresh-test seeds to tune the model.

- Live progress: `runs/defense-ppo-03-enter/status.json` and `metrics.jsonl`
  (the optional learned-Enter experiment; see notes below).
- Historical checkpoints: `runs/defense-ppo-*/step-*/`, including optimizer,
  configuration, policy weights and each completed validation suite.
- Stable best effort, once a validation candidate is verified:
  [replay](results/defense/learned/best/replay.html) and
  [weights](results/defense/learned/best/model.safetensors).
- Every promotion appends an immutable bundle under `results/defense/learned/versions/`
  and atomically switches the `best` symlink. Earlier versions are preserved.
  Each contains weights, configuration, evaluation, SHA-256 manifest, action/
  screen/reward trace, and a verification report.

Selection prioritizes completed missions, then highest stage, then score, among
**complete from-boot games only**. The stable best is a best single effort, not
a claim of reliable mean performance. Before promotion, the frozen weights are
reloaded and must exactly reproduce every neural action, reward and screen.
No unverified or truncated replay replaces the best.

Evaluate a frozen checkpoint independently (choose a new output path):

```sh
venv/bin/python -m rl.defense_evaluate \
  results/defense/learned/best/model.safetensors \
  --output runs/defense-fresh-evaluation.json --games 10 --seed 20000 --envs 10
```

The goal remains to observe and verify the original game's mission-ending
screen and subsequent behavior. The three-stage wrap in the binary is not
permission to relabel a ship-loss GAME OVER as a victory or manufacture more
levels. Keep improving until the successful completion sequence is observed.

### Experiment log: initial training and recovery

- `defense-ppo-01`: 531,200 sampled training actions before the strict HUD
  settling check stopped a worker. The trainer saved policy/optimizer state
  and closed its workers; this was not a completed training goal. Its full
  [log](results/defense/training/ppo-01/metrics.jsonl) and
  [configuration](results/defense/training/ppo-01/config.json) are preserved.
- Best frozen checkpoint from that run: step 303,104, score **340**, 1,615
  verified neural actions. Ten complete validation games: mean **310**, median
  **300**, highest stage **1**, no completed mission.
- The guard now permits continuously animated rows that contain no readable
  score/lives, while still rejecting unstable numeric HUDs. It makes no change
  to the timing or outputs of previously successful runs. Both archived learned
  replays were re-executed after the fix: all actions, screens and rewards were
  identical. The new run also records the environment source hash.
- `defense-ppo-02`: resumes run 01's saved policy, optimizer and RNG at action
  counter 531,200, with the same hyperparameters and a fresh set of from-boot
  emulator episodes. No demonstrations or hidden-state gameplay inputs added.
- Run 02 was subsequently checkpointed and paused cleanly at **1,542,912**
  cumulative actions. No stage 2 was observed. Its best frozen checkpoint,
  step **1,133,312**, reached **380** points (1,658 verified neural actions);
  ten-game mean **358**, median **360**, highest stage **1**. Later validation
  fell back to mean 294. All intermediate checkpoints remain in `runs/`, and
  its [complete log](results/defense/training/ppo-02/metrics.jsonl) is preserved.
- A [temporal probe](results/defense/temporal-probes/defense-temporal-932608-400k.json)
  held each frozen-model action for 400,000 rather than 100,000 T-states. On
  the same ten validation seeds, mean changed from 318 to 320 and best stayed
  340; neither protocol reached stage 2. This does not establish improvement.
  The [400k no-input check](results/defense/temporal-probes/noop-400k.json) still
  scored 280 and lost all four ships in ten complete games.
- `defense-ppo-03-enter` is a **fresh 21-action model**, not a resumed
  incompatible 20-action optimizer. Start command is the training command
  above with `--run runs/defense-ppo-03-enter --allow-enter`. All other
  hyperparameters remain the same. Its action counter starts at zero; the
  earlier 1,542,912 actions are separate prior experiment compute.
- The Enter-control test reduced a no-input game from about 1,535 to **1,043**
  decisions without changing its score (280), number of ship losses (four),
  or loss ending. All three between-life stage-1 intros were still observed
  in sampled screen frames, even with Enter continuously held. A 64-action
  GPU smoke test completed model reload and exact 21-action replay verification.
  This is an efficiency hypothesis to test through learning, not a mission win.

Remaining validation: stage 2/3 controls and mission-success detection are
supported by disassembly and parser tests, but **not yet exercised by an
unmodified complete playthrough reaching those stages**. Validate them when a
learned policy reaches them; do not patch memory or supply scripted expert play
to manufacture a success. The screen parser is single-player only. If a new
binary changes its HUD or awards extra ships, its assumptions must be audited
again. Keep models and diagnostics tied to the recorded executable hash.
