# Obstacle Run / Missile Defense: integration and training feasibility

The user-confirmed new game is the executable in `Missile_Defense.zip`.
It identifies itself as **Obstacle Run**, by Arno Puder (1983/84), and is
already present as `var/defense.cmd`. No emulator rebuild, disk controller,
new ROM, binary patch or duplicate game asset is needed.

Status: **screen-only PPO training is running independently of Breakdown**, with
parallel emulator workers, resumable checkpoints, complete-game validation and
automatic verified best-effort replays. See the commands and monitoring paths
below. A successful mission has not yet been verified.
The current standard-policy best is **10,480 points**, with **2,580** neural
actions exactly reverified; its ten-game mean is **9,981**, median **10,380**, all stage 1.
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

The [stage-one course audit](results/defense/stage-one-course-audit.json) verifies
that the original first stage advances automatically when its scrolling
obstacle stream is exhausted, **not when a score threshold is reached**. Static
decoding finds 126 stream rows between `0x76A9` and the end marker at `0x7BD8`;
the original row-update countdown is six. These are engineering facts about
the immutable game asset, not measurements of any policy's progress. No stream
pointer, row counter, layout-derived route, or other hidden state is supplied
to the policy, reward, or curriculum. A learned full-game stage transition
must still be observed on screen before reporting a clear.

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

- Live progress: `runs/defense-ppo-17-fresh-seed/status.json`,
  `runs/defense-dqn-22-fresh/status.json`,
  `runs/defense-ppo-23-life-age/status.json` and
  `runs/defense-dqn-24-bootstrap/status.json`, each with an adjacent
  `metrics.jsonl`. Earlier trials have stopped cleanly; their outcomes and
  archived resumable checkpoints are recorded below. Confirm a status file's
  PID is still alive before treating it as evidence of a running learner.
- Historical checkpoints: `runs/defense-ppo-*/step-*/` and `runs/defense-dqn-*/step-*/`, including optimizer,
  configuration, policy weights and each completed validation suite.
- Stable best effort, once a validation candidate is verified:
  [replay](results/defense/learned/best/replay.html) and
  [weights](results/defense/learned/best/model.safetensors).
- Every promotion appends an immutable bundle under `results/defense/learned/versions/`
  and atomically switches the `best` symlink. Earlier versions are preserved.
  Each contains weights, configuration, evaluation, SHA-256 manifest, action/
  screen/reward trace, and a verification report.
- With concurrent learners, the single `rl.defense_collect` process owns that
  shared archive. Its status/log live in `runs/defense-collector/`. The learner
  processes write only to their separate experiment artifact roots.

Selection prioritizes completed missions, then highest stage, then score, among
**complete from-boot games only**. The stable best is a best single effort, not
a claim of reliable mean performance. Before promotion, the frozen weights are
reloaded and must exactly reproduce every neural action, reward and screen.
No unverified or truncated replay replaces the best.

The collector checks the isolated archives every 30 seconds, pins an immutable
source version, checks its checksums, and invokes the same full frozen-policy
verification before a stronger candidate can replace the global best. It does
not promote sampling diagnostics, change training, or push to GitHub. A
destination lock prevents duplicate collectors. Do not simultaneously point a
trainer directly at the shared archive while this collector owns it.

```sh
venv/bin/python -u -m rl.defense_collect \
  --source runs/defense-ppo-05-low-entropy/artifacts \
  --source runs/defense-ppo-06-life-boundary/artifacts \
  --source runs/defense-ppo-07-curriculum/artifacts \
  --source runs/defense-ppo-08-long-rollout/artifacts \
  --source runs/defense-ppo-09-curriculum-life/artifacts \
  --source runs/defense-ppo-10-long-credit/artifacts \
  --source runs/defense-ppo-11-exploration/artifacts \
  --source runs/defense-ppo-12-lookback/artifacts \
  --source runs/defense-ppo-13-screen-cells/artifacts \
  --source runs/defense-ppo-14-long-horizon/artifacts \
  --source runs/defense-ppo-15-bias-noise/artifacts \
  --source runs/defense-ppo-16-strong-bias-noise/artifacts \
  --source runs/defense-ppo-17-fresh-seed/artifacts \
  --source runs/defense-ppo-18-weight-noise/artifacts \
  --source runs/defense-ppo-19-moderate-weight-noise/artifacts \
  --source runs/defense-ppo-20-encoder-transfer/artifacts \
  --source runs/defense-ppo-21-long-lookback/artifacts \
  --source runs/defense-dqn-22-fresh/artifacts \
  --source runs/defense-ppo-23-life-age/artifacts \
  --source runs/defense-dqn-24-bootstrap/artifacts \
  --output results/defense/learned --run runs/defense-collector --interval 30
```

The collector passed an isolated end-to-end check: it reloaded run 06's frozen
380-point policy and reproduced all 1,679 actions/screens/rewards before
publishing into the smoke-test directory. A second collector targeting the
live destination was rejected. The production collector is now running; it
leaves the existing global best untouched when source ranks are tied or lower.

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
- Run 03 stopped cleanly at **1,073,152** actions. Its ten validation rounds
  never reached stage 2. The strongest round was step **200,704**: mean **326**,
  median **320**, best **340**; the last round fell to mean **286**. Its
  [full log](results/defense/training/ppo-03-enter/metrics.jsonl) and strongest
  [resumable checkpoint](results/defense/training/ppo-03-enter/step-000000200704/state.json)
  are archived, including weights and optimizer. The separate 380-point
  20-action best remains unchanged.
- A further [100-game random-action timing check](results/defense/temporal-probes/random-400k-100-games.json)
  at 400,000 T-states yielded mean **288.8**, median **280**, best **320**,
  all stage 1. No episodes hit its diagnostic cap. These diagnostic actions
  are not supplied to the learner.
- `defense-ppo-04-sil` resumes run 03's strongest compatible checkpoint with
  four optional self-imitation updates per rollout. All other training
  settings are unchanged. The auxiliary replay contains only this learner's
  own newly collected training screens, actions and discounted score rewards;
  no evaluation games, replay files, demonstrations or hidden-state targets.
  Positive return advantages select the useful updates. A bounded suffix
  ends at a real learning terminal; truncations are discarded. Replay memory
  is not checkpointed and refills after resuming (the sampling RNG is saved).
  The inherited action counter starts at 200,704, not at run 03's final count.
- Before activation, a [4,096-action smoke run](results/defense/training/sil-smoke-01/metrics.jsonl)
  exercised the auxiliary optimizer and complete-game evaluation. The optimizer
  made 64 PPO updates plus 16 self-imitation updates; its selected 300-point
  replay reproduced all **1,152** neural actions/screens/rewards after reloading
  frozen weights. This is an integration check, not improved performance.
- Run 04 then completed **1,032,192 additional actions**, stopping cleanly at
  inherited counter **1,232,896**, with 928 new complete training games.
  Across ten validation rounds, its best mean was **312**, below the starting
  checkpoint's **326**; the final round averaged **302**. Every round's best
  was at most 320, and no game reached stage 2. This configuration did not
  improve the policy. Its [complete log](results/defense/training/ppo-04-sil/metrics.jsonl)
  and [final resumable checkpoint](results/defense/training/ppo-04-sil/final-checkpoint/state.json)
  are retained; the newer lower-entropy continuation now has the machine.

Continue the self-imitation experiment from the archived starting checkpoint
into a new run directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-sil-reproduction \
  --resume results/defense/training/ppo-03-enter/step-000000200704 \
  --artifacts runs/defense-sil-reproduction/artifacts --sil-updates 4
```

`--sil-updates 0` (the default for a fresh run) leaves plain PPO unchanged.
Only one active production learner should write to `results/defense/learned`;
parallel experiments must use separate artifact directories. Breakdown keeps
its original action count and visible-score parser defaults.

### Sampling diagnostic: preserved 400-point replay

On the frozen 380-point checkpoint and the same ten validation seeds, reducing
sampling temperature from 1 to **0.5** increased mean score from **358 to 384**,
median from **360 to 380**, and best from **380 to 400**. Temperature **0.25**
gave mean/median/best **380**. Every game remained in stage 1. These are reused
validation results, not fresh testing or evidence of mission completion.

The [400-point replay](results/defense/sampling-probes/temperature-050/replay.html)
reproduced all **1,743** neural actions, screens and rewards after reloading the
original weights with temperature 0.5. Its separate bundle includes weights,
configuration, evaluation, trace and checksums. The HTML explicitly labels this
an evaluation-only sampling probe. It does not replace `learned/best`, whose
original temperature-1 policy and replay remain unchanged. Loading these same
weights without the temperature override reproduces the original policy, not
the diagnostic. This result motivates testing lower training entropy if the
current self-imitation run plateaus; it does not yet establish that doing so
will improve learning.

```sh
venv/bin/python -m rl.defense_evaluate \
  results/defense/sampling-probes/temperature-050/model.safetensors \
  --temperature .5 --seed 10000 --games 10 --envs 10 \
  --output runs/defense-temperature-reproduction.json \
  --replay-output runs/defense-temperature-reproduction
```

The optional recorder refuses incomplete suites, changed weights, existing
output directories and temporal overrides; it never promotes a diagnostic
into the live learner's best-artifact directory.

A later temperature-0.5 check on the frozen **500-point** model (counter
4,434,688) used the same ten validation seeds. Mean changed from **462 to 464**,
median stayed **460**, and best changed from **500 to 520**, all stage 1.
The [separate diagnostic replay](results/defense/sampling-probes/step-4434688-temperature-050/replay.html)
reproduced all **1,837** neural actions after reload. This small mean change
does not establish a useful sampling improvement, and the subsequently trained
standard-policy 560-point model is stronger. The diagnostic remains separate
from the shared best; its weights are the unchanged older 500-point model.

On the later frozen **10,280-point** model at counter **7,244,544**, two more
sampling diagnostics used the same ten validation seeds and unchanged action
timing. These are exploratory comparisons, not independent fresh-test results.

| Sampling temperature | Mean | Median | Best | Highest stage |
|---|---:|---:|---:|---:|
| 1 (original evaluation) | 8,834 | 9,115 | 10,280 | 1 |
| 0.5 | 8,951 | 9,125 | 10,260 | 1 |
| 1.5 | 9,107 | 9,815 | 10,280 | 1 |

Neither setting reached stage 2 or completed a mission. The
[temperature-0.5 bundle](results/defense/sampling-probes/step-7244544-temperature-050/replay.html)
and [temperature-1.5 bundle](results/defense/sampling-probes/step-7244544-temperature-150/replay.html)
preserve unchanged weights, complete evaluations, traces and reloaded-policy
verification. These diagnostic bundles are excluded from best-model promotion.
This small comparison does not establish that changing training entropy would
help; these probes did not change the active learners' settings.

An additional [action-timing diagnostic](results/defense/sampling-probes/step-8342272-tstates-50000.json)
used the frozen **10,480-point** model at counter **8,342,272**, ordinary
temperature-1 sampling, and the same ten validation seeds. Reducing each action
from **100,000 to 50,000 T-states** (twice as many policy decisions per emulated
second) gave mean **5,809**, median **5,785**, best **8,740**, all stage 1 and no
mission. The original timing gave mean **9,981**, median **10,380**, best
**10,480**. All diagnostic games completed without an action cap.
This is an explicit evaluation-only timing override, not a newly trained model
or a best-replay promotion. It provides no evidence that changing the frozen
policy's control rate alone helps; it does not rule out learning separately at
the shorter interval. Both active learners retain their original timing.

A read-only action-alias check used the frozen
[10,480-point policy and its own trace](results/defense/learned/versions/step-000008342272-125346536cb1-seed-10004/manifest.json),
querying 109 four-frame observations at action indices **280–388** before its
first ship loss. Mean categorical entropy was **0.21745 nats**; merging the
nine forward-fire aliases (action IDs 9–17, equivalent in stage 1) into one
probability gave **0.19499 nats**. Only **0.02246 nats**, about 10%, came from
variation within those aliases. Mean movement probability was **0.60089**;
forward-fire-only probability was **0.18815**. This narrow pre-collision window
does not support redundant fire choices as the dominant source of apparent
exploration. It is not an all-state or later-stage result. The diagnostic
changed no weights, controls or training inputs, and the 20-action set remains
unchanged.

### Lower-entropy continuation

`defense-ppo-05-low-entropy` resumes the original 20-action checkpoint at
**1,133,312** actions (the 380-point model, mean 358). Its entropy coefficient
is **0.002**, down from 0.02; other PPO/gameplay settings are unchanged and
self-imitation is disabled. This directly tests learning with less pressure
to keep the action distribution diffuse. It does not scale inference logits:
validation remains at temperature 1. The starting model and optimizer are
[archived together](results/defense/training/ppo-05-low-entropy/start-checkpoint/state.json).

Its first ten-game validation, after 102,400 additional actions, had mean
**330**, median **330**, best **360**, all stage 1. Thus there is **no improvement
claim** yet. The next two rounds averaged 332 and 338, still below the starting
checkpoint. It initially ran alongside the self-imitation experiment, with
separate artifact roots to avoid competing writers; run 04 is now paused.
The experiment's local
verified replay is `runs/defense-ppo-05-low-entropy/artifacts/best/replay.html`;
the older 380-point policy and separate 400-point sampling probe remain archived.

Reproduce from the archived optimizer into a new run directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-low-entropy-reproduction \
  --resume results/defense/training/ppo-05-low-entropy/start-checkpoint \
  --artifacts runs/defense-low-entropy-reproduction/artifacts \
  --entropy .002 --sil-updates 0
```

Later, after **602,112 additional actions**, run 05 reached a ten-game mean
of **370**, median **380**, best **380**, still all stage 1. This is improved
validation consistency relative to the starting mean 358, not a new best
individual score or fresh-test success rate. The
[step-1,735,424 checkpoint](results/defense/training/ppo-05-low-entropy/step-000001735424/state.json)
preserves its weights, optimizer and complete evaluation. The next validation
means were 362 and 326, so this is not a claim of monotonic improvement.

Continued unchanged, run 05 subsequently set verified bests of **420**, **440**,
and **460**. At counter **3,234,560**, ten complete validation games averaged
**416**, median **420**, best **460**, still all stage 1 with no mission.
The collector reloaded the frozen weights and reproduced all **1,675** actions,
screens and rewards before publishing the current best. Its
[resumable checkpoint](results/defense/training/ppo-05-low-entropy/step-000003234560/state.json)
includes the optimizer and evaluation. The next two validation means were 412
and 410. This improvement came from ordinary from-boot training, not the
optional curriculum below; all intermediate replay bundles remain preserved.

Run 05 then regressed for six consecutive validation rounds (means 318, 296,
322, 324, 318, 316). It was stopped cleanly at **4,094,720** inherited actions:
**2,961,408 additional actions**, **1,824 new complete training games**, and
29 validation rounds. Its [complete log](results/defense/training/ppo-05-low-entropy/metrics.jsonl)
and [final resumable checkpoint](results/defense/training/ppo-05-low-entropy/final-checkpoint/state.json)
are preserved. Run 06 continued; the freed slot started the own-experience
curriculum from run 05's strongest archived checkpoint, not its regressed end.

### Ship-loss learning-boundary comparison

`defense-ppo-06-life-boundary` starts from exactly the same checkpoint and
settings as run 05, with only `--life-terminal` enabled. A configuration diff
confirms this is the only difference besides run/artifact directories. The
boundary prevents estimated returns from crossing a visible ship loss;
**reward remains the actual score delta**. The emulator is not reset after a
ship loss: the original game continues through its own introduction, and all
evaluation games still run from boot until all four ships are gone.

Its first ten-game validation, after **102,400 additional actions**, had mean
**368**, median **370**, best **380**, all stage 1, compared with run 05's mean
330 at the same training increment. This is an encouraging early comparison,
not stage completion or a statistically established success-rate improvement.
The second round fell to mean **314**, median **320**, best **340**; the early
gain is therefore not evidence of stable superiority.
Run 06 used its isolated archive throughout training;
run 06's local replay is
`runs/defense-ppo-06-life-boundary/artifacts/best/replay.html`.

Reproduce using the command above with a new run/artifact directory and
`--life-terminal`. Its exact
[configuration](results/defense/training/ppo-06-life-boundary/resume-config.json)
and [first validation](results/defense/training/ppo-06-life-boundary/first-validation.json)
are preserved. The original 380-point best and separate 400-point sampling
replay remain available; no unsuccessful candidate replaces them.

Run 06 subsequently reached **400 points with the ordinary temperature-1
policy**, at counter **2,034,432** (901,120 additional training actions).
Ten complete validation games: mean **360**, median **360**, best **400**,
all stage 1, no mission completion. The collector reloaded the frozen model
and verified all **1,682** actions, screens and rewards, then promoted it to
the [shared best replay](results/defense/learned/best/replay.html) and
[weights](results/defense/learned/best/model.safetensors).
This is a new trained-policy best, distinct from the older temperature-0.5
diagnostic. Neither demonstrates a stage clear. All previous versions remain.

Run 06 later matched the strongest ten-game validation mean, **416**, at
counter **3,435,264** (median 420, best 440), still all stage 1. It continued
unchanged while separate experiments ran alongside it.

At counter **4,434,688**, unchanged run 06 set a new standard-policy best of
**500 points**: ten complete validation games averaged **462**, median **460**,
still all stage 1, with no completed mission. The collector reloaded the frozen
weights and exactly verified all **1,901** actions, rewards and screens before
promoting the [shared replay](results/defense/learned/best/replay.html) and
[weights](results/defense/learned/best/model.safetensors). The
[resumable milestone](results/defense/training/ppo-06-life-boundary/step-000004434688/state.json)
also preserves its optimizer and full evaluation. This is **3,301,376 additional
actions** since its starting checkpoint, not training from scratch at that
counter. The earlier 460-point model and every prior replay remain archived.

Continuing unchanged, run 06 reached **560 points** at counter **4,733,696**
(**3,600,384 additional actions**). Ten complete validation games: mean **506**,
median **510**, best **560**, all stage 1, no mission. The collector verified
every one of the **1,975** neural actions/screens/rewards before promotion.
Its [resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000004733696/state.json)
preserves weights, optimizer and the complete evaluation. The 500-point bundle
and all earlier milestones remain available.

The next round, counter **4,836,096**, produced a **580-point** best effort,
with all **1,904** actions exactly verified by the collector. Its ten-game
mean was **480**, median **460**, all stage 1, with no completed mission.
This improves the best individual replay, not mean performance: the preceding
560-point checkpoint averaged 506 and remains archived separately. The
[580-point resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000004836096/state.json)
includes weights, optimizer and its complete evaluation.

After validation means of 522, 522, 542 and 560, unchanged run 06 reached
**600 points** at counter **5,434,112**. Ten complete validation games averaged
**564**, median **570**, best **600**, all stage 1 with no completed mission.
The collector reloaded the weights and verified all **2,003** actions, screens
and rewards before promotion. Its
[resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000005434112/state.json)
preserves the optimizer and complete evaluation alongside the shared replay.
This improves both the best effort and reused-seed validation mean, not a
fresh-test success rate or evidence of stage completion.

Run 06 eventually stopped cleanly at counter **7,199,488**, after **6,066,176
additional actions**, **3,423 new complete training games**, and **60 validation
rounds**. No game reached stage 2; its best-so-far effort stayed at 600 for
17 further rounds after first reaching that score. The strongest mean was
**586**; the latest matching
[checkpoint](results/defense/training/ppo-06-life-boundary/step-000007035648/state.json)
is preserved with optimizer and evaluation. Its
[complete log](results/defense/training/ppo-06-life-boundary/metrics.jsonl) and
[final resumable state](results/defense/training/ppo-06-life-boundary/final-checkpoint/state.json)
are archived. Run 08 subsequently reached 620 and later 10,280; its final
results are below. The freed slot tested curriculum resets with the successful
ship-loss learning boundaries.

### Own-experience curriculum: run 07

`rl.defense_snapshot` adapts the existing native snapshot API to Defense's
visible score, ships, stage, screen history and outcome bookkeeping. It saves
and restores an opaque emulator state reached through actual gameplay; it
does not decode internal bytes, edit game state, select actions, or expose
anything beyond the usual screen observation to a policy. It does not rewind
the boot RNG and rejects incompatible timing, action profiles and malformed
snapshots. No emulator rebuild was needed.

Native regression tests captured the original learned 380-point trajectory at
action 80, then reproduced every remaining **1,578** screen/action/reward
transition through game over in both the original process and a newly spawned
process. The native terminal-text stop was also restored correctly. These
recorded actions are test fixtures only, not training examples. The 21-action
profile and invalid-state rejection are tested separately.

`rl.defense_curriculum` optionally resets training workers to these unmodified,
own-reached states. Archive bins use visible stage and score earned since the
current life/stage began, not cumulative points from previous lost ships.
Each stage has a bounded number of bins and reservoir entries. Workers can
share snapshots reached within the same run; protected boot workers always
start normally. There is no archive-file loader or external demonstration input.

Restored segments are marked `full_game=False`, cannot enter best-game ranking,
and report only newly earned score. Training counters distinguish boot games
from restored segments, including in self-imitation provenance. Reward remains
the actual score delta, and evaluation always uses complete ordinary from-boot
games. Archives are not checkpointed: after resume they refill from new play.
The default curriculum probability is zero, leaving normal training unchanged.

An isolated [8,192-action smoke run](results/defense/training/curriculum-smoke-01/metrics.jsonl)
completed three boot games and four restored segments with protected-worker
sharing and self-imitation enabled. Its two-game validation mean was 290;
the selected replay reproduced all 1,548 actions. A further
[4,096-action resume check](results/defense/training/curriculum-smoke-resume-01/metrics.jsonl)
restored optimizer/RNG state, rebuilt archives from new play, and verified a
1,557-action replay. These are integration checks, not performance gains.

`defense-ppo-07-curriculum` resumed run 05's preserved 460-point checkpoint
at counter **3,234,560**. Only the curriculum is enabled: reset probability 0.5,
same-run snapshot sharing, and eight protected boot workers out of 32. All
PPO, reward, action, timing and evaluation settings are inherited unchanged;
self-imitation and ship-loss learning boundaries remain disabled. Its
[configuration](results/defense/training/ppo-07-curriculum/resume-config.json)
is archived. A new action counter segment is additional compute, not a
continuation from run 05's final counter. Reproduce in a new directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-curriculum-trial \
  --resume results/defense/training/ppo-05-low-entropy/step-000003234560 \
  --artifacts runs/defense-curriculum-trial/artifacts \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8
```

The single collector has been restarted with run 07 as an additional isolated
source. Restored training segments cannot replace the best replay: promotion
still requires frozen-policy verification of a complete from-boot game.

After **102,400 additional actions**, run 07's
[first ten-game validation](results/defense/training/ppo-07-curriculum/first-validation.json)
had mean **410**, median **420**, best **440**, all stage 1 and no mission.
This is slightly below the starting checkpoint's mean 416, not improvement
yet. Its isolated replay reproduced all **1,733** neural actions after weight
reload; the then-current 460-point best remained unchanged. Boot-game and restored
segment counters remain separate in the live log.

The second validation mean rose to **420** and the third to **438** (median
**440**, best **460**) after **303,104 additional actions**, at counter
**3,537,664**. The [third checkpoint](results/defense/training/ppo-07-curriculum/step-000003537664/state.json)
preserves model, optimizer and the complete ten-game evaluation. This improves
consistency on reused validation seeds, not fresh-test performance: every game
remained in stage 1, with no mission completed. This did not replace the shared
best single-effort replay because its best score only tied the original
460-point model. Run 06 subsequently set the 600-point best described above.
Both historical runs were subsequently paused as described here.

After **2,027,520 additional actions**, run 07 stopped cleanly at inherited
counter **5,262,080**. It completed **974 new boot games** and **641 restored
practice segments**, with 20 complete validation rounds. Best validation mean
was 438, but the last three rounds each averaged 320; no game reached stage 2.
The early gain did not persist. Its [complete log](results/defense/training/ppo-07-curriculum/metrics.jsonl)
and [final resumable checkpoint](results/defense/training/ppo-07-curriculum/final-checkpoint/state.json)
are preserved, as is the earlier higher-mean checkpoint. The freed slot now
runs the following comparison; no archive states were transferred to it.

### Longer-rollout comparison: run 08

`defense-ppo-08-long-rollout` resumes the preserved 600-point checkpoint at
counter **5,434,112**, changing only **rollout length from 128 to 256**.
The [configuration](results/defense/training/ppo-08-long-rollout/resume-config.json)
inherits the same 32 workers, 512-example optimizer minibatches, four epochs,
learning rate, entropy, discount, ship-loss learning boundaries and score reward.
Curriculum and self-imitation are disabled. Game timing remains 100,000 T-states
per policy action, so this changes training batches, not the game or controls.

Motivation: in the [archived 460-point replay](results/defense/learned/versions/step-000003234560-5d80be3dfa03-seed-10000/replay.html),
three of thirteen non-overlapping full 128-action chunks had no reward; two
consisted entirely of explicitly visible introduction text. No full 256-action
chunk was entirely reward-free. These replay observations are not measurements of
every actual multi-worker training batch, nor proof of a cause. The hypothesis
is that longer batches mix introductions and active play more consistently.
They do not skip screens, inject actions, or add reward. Run 06 initially stayed
as an unchanged comparison; complete from-boot validation decides performance.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-rollout-reproduction \
  --resume results/defense/training/ppo-06-life-boundary/step-000005434112 \
  --artifacts runs/defense-long-rollout-reproduction/artifacts --rollout 256
```

The single collector includes run 08's isolated artifact directory. It still
requires exact frozen-policy replay verification before a stronger complete
effort can replace the shared best.

After **106,496 additional actions**, run 08's
[first validation](results/defense/training/ppo-08-long-rollout/first-validation.json)
averaged **552**, median **580**, best **600**, all stage 1 with no mission.
Its selected complete replay reproduced all **1,982** neural actions after
reloading the model. This is below the starting checkpoint's mean 564, not an
improvement claim. It tied the then-current shared best score and therefore did
not replace the existing 600-point bundle. Run 08 continued unchanged.

After **1,204,224 additional actions**, the longer-rollout trial reached
**620 points** at counter **6,638,336**. Ten complete validation games: mean
**574**, median **580**, best **620**, all stage 1 and no mission. The collector
reloaded the frozen model and exactly reproduced all **2,047** actions, screens
and rewards before promoting it. The
[resumable milestone](results/defense/training/ppo-08-long-rollout/step-000006638336/state.json)
preserves weights, optimizer and the complete evaluation. A preceding round
averaged 584, so the best single replay is not also the highest-mean checkpoint.
The 600-point model and all older versions remain available. This is progress
on reused validation seeds, not yet evidence of a stage clear or mission win.

Run 08 subsequently reached **10,280 points**, with all **2,495** neural
actions verified in its own [best replay](results/defense/training/ppo-08-long-rollout/best-effort/replay.html).
Its strongest ten-game mean was **9,358**, median **9,510**, best **10,280**, at
counter **8,940,288**; that [resumable checkpoint](results/defense/training/ppo-08-long-rollout/step-000008940288/state.json)
is preserved separately. It did not exceed 10,280 over five consecutive
validation rounds, and no training or evaluation game reached stage 2.
The run stopped cleanly at **9,325,312**, after **3,891,200 additional actions**,
**1,890 new complete training games** and **38 validation rounds**. Its
[complete log](results/defense/training/ppo-08-long-rollout/metrics.jsonl) and
[final model and optimizer](results/defense/training/ppo-08-long-rollout/final-checkpoint/state.json)
are archived. Its training slot now runs the longer-credit comparison below;
the independent run 09 was not interrupted.

### Curriculum with ship-loss boundaries: run 09

`defense-ppo-09-curriculum-life` resumes the preserved 620-point model at
counter **6,638,336**. Only own-experience curriculum resets are enabled relative
to that checkpoint: probability 0.5, same-run snapshot sharing, and eight
protected boot workers. All PPO settings, including **256-action rollouts and
ship-loss learning boundaries**, remain unchanged. The
[configuration](results/defense/training/ppo-09-curriculum-life/resume-config.json)
records this comparison. Run 08 initially remained the unchanged baseline;
it was later archived as described above. Run 09 was subsequently archived too,
after the results documented below.

This is not a restart of the earlier failed curriculum's final policy. Run 07
started from a weaker 460-point model, used 128-action rollouts and did not have
ship-loss learning boundaries. Run 09 starts with an empty archive and fills it
only from its own newly reached states; it loads no demonstrations or prior
snapshot archive. The same screen-only input, actual score-delta reward,
restored-segment bookkeeping and complete from-boot evaluation rules apply.
The collector includes its isolated output, but restored practice segments
remain ineligible to replace the best complete-game replay.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-curriculum-life-reproduction \
  --resume results/defense/training/ppo-08-long-rollout/step-000006638336 \
  --artifacts runs/defense-curriculum-life-reproduction/artifacts \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8
```

After **106,496 additional actions**, run 09's
[first validation](results/defense/training/ppo-09-curriculum-life/first-validation.json)
averaged **554**, median **560**, best **580**, all stage 1 and no mission.
Its replay reproduced all **1,927** neural actions after reload. This is below
the starting checkpoint's mean 574, not an improvement. Runtime checks after
34 boot games and 20 restored segments found no protected-worker restores or
segment reward-accounting violations. Both live runs continue unchanged; the
verified 620-point shared best remains preserved.

At counter **6,941,440**, after **303,104 additional actions**, run 09 reached
**2,950 points** in a complete ordinary from-boot validation game. Ten games:
mean **1,055**, median **650**, best **2,950**, all stage 1, no mission. The
collector reloaded the frozen weights and reproduced all **2,008** actions,
screens and rewards before promoting the shared best. Independent trace checks
confirmed an initial visible score of zero with four ships, total reward of
2,950, and only stage-1 introduction text; this is not a restored practice
segment or a claimed stage clear. The
[resumable milestone](results/defense/training/ppo-09-curriculum-life/step-000006941440/state.json)
preserves weights, optimizer and the complete evaluation. All older models
and verified replays remain available; both then-active runs continued unchanged.
The visible reward trace contains thirty 20-point increments and awards of
100, 750 and 1,500 points. The large score gain therefore does not by itself
demonstrate a longer survival time or a later stage.

The next checkpoint, counter **7,039,744** (**401,408 additional actions**),
reached **10,200 points**. Ten complete games averaged **7,019**, median **7,090**,
best **10,200**, all stage 1 with no completed mission. The collector verified
all **2,495** actions, screens and rewards from the reloaded frozen policy;
trace checks again confirmed zero initial score, four ships, and no stage-2
intro or mission message. Its
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007039744/state.json)
and immutable replay bundle are preserved separately from the 2,950-point
milestone. This remains a score improvement, not a verified stage clear.

At counter **7,146,240**, after **507,904 additional actions**, run 09 reached
**10,260 points**, with all **2,562** actions exactly verified from boot.
Ten complete validation games: mean **8,169**, median **8,030**, best **10,260**,
still all stage 1 and no mission. The
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007146240/state.json)
and complete replay bundle are preserved; both then-active experiments continued.

At counter **7,244,544**, after **606,208 additional actions**, run 09 reached
**10,280 points**, with all **2,519** actions exactly verified from boot.
Ten complete validation games: mean **8,834**, median **9,115**, best **10,280**,
still all stage 1 and no mission. The immutable replay bundle preserves its
weights, evaluation, action trace and verification alongside the earlier milestones.

Run 09 also improved consistency at counter **7,342,848**: ten complete games
averaged **9,527**, median **10,240**, best **10,280**. This ties the best
individual score, so it does not replace the best-effort replay. Its separate
[model and optimizer checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007342848/state.json)
is preserved for resuming training. All ten games remained stage 1, without
a successful mission; these reused validation seeds are not a fresh test set.

At counter **8,342,272**, after **1,703,936 additional actions**, run 09 reached
**10,480 points**, with all **2,580** actions exactly verified from boot.
Ten complete validation games: mean **9,981**, median **10,380**, best **10,480**,
still all stage 1 and no mission. Its separate
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000008342272/state.json)
and immutable replay bundle are preserved. No successful stage clear has
been observed in this run or the archived longer-rollout comparison.

Run 09 eventually stopped cleanly at counter **9,677,568**, after **3,039,232
additional actions**, **1,010 new complete boot games**, **740 restored practice
segments**, and **30 complete validation rounds**. Its best stayed at 10,480
for thirteen further rounds after first reaching that score, with no stage-2
or mission-success event in training or evaluation. The strongest mean was
**10,439**, median **10,460**, best **10,480**, at counter **9,243,392**; that
[model and optimizer](results/defense/training/ppo-09-curriculum-life/step-000009243392/state.json)
are preserved separately from the best-effort model. Its
[complete log](results/defense/training/ppo-09-curriculum-life/metrics.jsonl) and
[final resumable state](results/defense/training/ppo-09-curriculum-life/final-checkpoint/state.json)
are archived. The successful score milestones remain available. The freed
training slot then tested stronger exploration, while run 10 continued unchanged.

### Longer-credit comparison: run 10

`defense-ppo-10-long-credit` resumes the preserved **10,480-point** checkpoint
at counter **8,342,272**. The only changed training parameter is **GAE lambda
0.95 to 0.99**; the [configuration](results/defense/training/ppo-10-long-credit/resume-config.json)
was compared directly with its parent checkpoint. The existing return code
propagates an error backward within a learning segment by `gamma * lambda`
per action. With unchanged gamma 0.997, the new trace decays more slowly.
The hypothesis is that this helps assign delayed score rewards to earlier
choices; it is not a demonstrated improvement yet.

The screen input, 20 actions, timing, score-only reward, 256-action rollouts,
learning rate, entropy, curriculum settings, protected boot workers and
ship-loss boundaries are unchanged. Model, optimizer and policy RNG resume
from the saved checkpoint; the own-experience snapshot archive starts empty
and refills from newly played states. There are no game/episode action caps
or wall-clock limits. Three focused checks passed for analytic trace decay,
episode-boundary isolation and optimizer-resume equivalence before launch.
Run 09 initially continued unchanged as the reference experiment, and was later
archived as described above. Run 10 was later archived too. Experiments write isolated
artifacts, with the single verification collector watching their sources.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-credit-reproduction \
  --resume results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --artifacts runs/defense-long-credit-reproduction/artifacts --gae-lambda .99
```

After **106,496 additional actions**, run 10's
[first complete validation](results/defense/training/ppo-10-long-credit/first-validation.json)
averaged **9,275**, median **10,390**, best **10,460**, all stage 1 and no mission.
Its local best replay exactly reproduced all **2,568** neural actions after
reloading the weights; the [verification record](results/defense/training/ppo-10-long-credit/first-verification.json)
is preserved. This first result is below its starting checkpoint's mean 9,981
and best 10,480, so it does not replace the shared best or establish improvement.
The experiment continued unchanged beyond this initial validation.

Run 10 stopped cleanly at **9,988,864**, after **1,646,592 additional actions**,
**498 new complete boot games**, **375 completed restored segments** and
**16 validation rounds**. It tied 10,480 but never exceeded it or reached stage 2.
Its strongest ten-game mean was **10,473**, median **10,480**, best **10,480**,
at counter **9,046,784**. That
[resumable checkpoint](results/defense/training/ppo-10-long-credit/step-000009046784/state.json),
the [full log](results/defense/training/ppo-10-long-credit/metrics.jsonl),
[final model/optimizer](results/defense/training/ppo-10-long-credit/final-checkpoint/state.json)
and [verified best replay](results/defense/training/ppo-10-long-credit/best-effort/replay.html)
are preserved. The replay reproduces all **2,562** actions. This improved
score consistency, not stage completion. Its freed slot now tests the
earlier-state curriculum described below; run 11 continues unchanged.

### Stronger-exploration comparison: run 11

`defense-ppo-11-exploration` resumes the **same 10,480-point checkpoint** at
counter **8,342,272** as run 10. Its only changed parameter relative to that
parent is **entropy coefficient 0.002 to 0.01**, confirmed by comparing the
[saved configuration](results/defense/training/ppo-11-exploration/resume-config.json).
GAE lambda remains 0.95 here. Run 10 tested longer credit assignment and
run 11 tests stronger exploration, each changing one parameter from their
common preserved starting point. Neither intervention has yet established
that it can clear the first stage.

Motivation: the original curriculum run repeatedly learned high-scoring
stage-1 behavior without surviving to stage 2. Encouraging more varied learned
actions during training is a hypothesis for escaping that behavior, not a
guaranteed improvement. This is different from the earlier evaluation-only
sampling probes: ordinary temperature-1 sampling is retained at evaluation.
All screen-input, score-reward, timing, action, ship-loss, curriculum and
protected-boot settings remain unchanged. The snapshot archive starts empty,
using only this run's newly reached states. There are no wall-clock or action
limits. The collector includes this isolated source; the existing best model
and replay remain available throughout training.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-exploration-reproduction \
  --resume results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --artifacts runs/defense-exploration-reproduction/artifacts --entropy .01
```

After **106,496 additional actions**, run 11's
[first validation](results/defense/training/ppo-11-exploration/first-validation.json)
averaged **9,585**, median **10,140**, best **10,260**, all stage 1 and no mission.
Its local replay reproduced all **2,500** neural actions after reloading the
frozen model; the [verification record](results/defense/training/ppo-11-exploration/first-verification.json)
is preserved. This is below its starting model's mean 9,981 and best 10,480;
it does not replace the shared best. The experiment continues unchanged.

Run 11 was subsequently stopped gracefully at counter **10,865,408**, after
**2,523,136 additional actions**, 795 new completed boot games, 524 completed
restored segments and 25 ten-game validations. It tied 10,480 but never exceeded
it or reached stage 2. Its strongest validation mean was **10,450** (median
10,460) at counter **9,349,888**; the final validation mean was 9,353.
The [full log](results/defense/training/ppo-11-exploration/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-11-exploration/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-11-exploration/step-000009349888/state.json)
and [verified best replay](results/defense/training/ppo-11-exploration/best-effort/replay.html)
are retained. Increased entropy alone did not resolve this trial's bottleneck.

### Optional earlier-state curriculum

`--curriculum-lookback N` is disabled by default (`N=0`). With the option
enabled, a score-progress event can archive a state actually visited **N
actions earlier**, instead of the state at the reward event. The hypothesis
is that a practice reset with more lead-in may help when a rewarding state is
already close to a collision. It does not identify obstacles, select a route,
or provide actions, demonstrations, fabricated states, or extra reward.

The bounded opaque history contains at most `N+1` own-play snapshots per worker.
It clears at resets, detected ship losses, stage changes and terminal outcomes;
insufficient history is skipped. Newly reached stages are still archived
immediately. Archive buckets and reset score baselines use the **saved state's
own visible score**, not the later triggering score. Logs distinguish capture
and trigger actions/scores. Neither this history nor those labels reach the
policy. Snapshots and history are not loaded from demonstration files or
checkpointed; fresh runs refill them through their own gameplay. Ordinary
full-game evaluation still starts from boot without curriculum resets.

All **184 tests** passed, including exact earlier-state provenance, peer restore
and score accounting, life/stage boundary bookkeeping, protected-worker routing,
bounded history, and a full-game comparison proving that collection alone does
not change screens, rewards or outcomes. Stage-transition bookkeeping is unit
tested; this is not a claim of an actual learned stage-2 reach.

A separate [integration smoke run](results/defense/training/lookback-smoke-01/resume-config.json)
used four workers, a 32-action lookback and **16,384 additional training
actions**, then ten uncapped complete evaluation games. It collected 306 archive
entries, including 23 from an active restored segment; no restored segment had
finished when the bounded training smoke test ended. Full restored-episode
reward accounting is separately covered by the native regression tests.
Evaluation mean **9,854**, median **10,230**, best **10,260**, all stage 1 and
no mission. Its [frozen replay](results/defense/training/lookback-smoke-01/replay/replay.html)
reproduced all **2,433** neural actions. The log, model, optimizer, evaluation
and replay bundle are preserved. This is an integration check, not evidence
of improved performance; the smoke artifact source is excluded from the
production collector.

### Earlier-state curriculum: run 12

`defense-ppo-12-lookback` resumes run 10's strongest-mean checkpoint at
counter **9,046,784** (mean **10,473**, best **10,480**). Its only changed
training parameter is **curriculum lookback 0 to 32 actions**; the
[saved configuration](results/defense/training/ppo-12-lookback/resume-config.json)
was compared with the parent checkpoint, treating the previously absent
lookback option as its zero default. The curriculum source hash records the
new, tested implementation. All other training and game settings are inherited,
including GAE lambda 0.99, entropy 0.002, 32 workers, 256-action rollouts,
eight protected boot workers, score-only reward and ordinary evaluation.

The archive and bounded history start empty and fill only through new own play.
No prior snapshot archive or diagnostic trace is loaded. The run has no action
or wall-clock limit, and its isolated artifact source is included in the single
verification collector. Run 11 remains active as the exploration comparison.
This is an experiment aimed at the unresolved stage-1 bottleneck, not a claim
that the lookback hypothesis has improved performance.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-lookback-reproduction \
  --resume results/defense/training/ppo-10-long-credit/step-000009046784 \
  --artifacts runs/defense-lookback-reproduction/artifacts --curriculum-lookback 32
```

After **106,496 additional actions**, run 12's
[first validation](results/defense/training/ppo-12-lookback/first-validation.json)
averaged **10,431**, median **10,470**, best **10,480**, all stage 1 and no mission.
Its local replay reproduced all **2,544** neural actions; the
[verification record](results/defense/training/ppo-12-lookback/first-verification.json)
is preserved. This ties the shared best and is slightly below the starting
mean of 10,473, so it does not establish improvement or replace the existing
best-effort replay. The experiment continues unchanged.

Run 12 subsequently stopped gracefully at counter **11,094,784**, after
**2,048,000 additional actions**, 640 new completed boot games, 401 completed
restored segments and 20 ten-game validations. It never exceeded 10,480 or
reached stage 2. Its strongest mean was **10,474**, median 10,480, at counter
**10,447,616**; the final validation mean was 10,444. This tiny mean increase
over the parent (10,473) is not evidence of a meaningful gain on reused seeds.
The [full log](results/defense/training/ppo-12-lookback/metrics.jsonl),
[final checkpoint](results/defense/training/ppo-12-lookback/final-checkpoint/state.json),
[strongest-mean optimizer checkpoint](results/defense/training/ppo-12-lookback/step-000010447616/state.json)
and [best local replay](results/defense/training/ppo-12-lookback/best-effort/replay.html)
are preserved. Earlier resets alone did not resolve this trial's bottleneck.

### Optional screen-cell curriculum

`--curriculum-cells screen` changes the training archive's grouping from score
bins to coarse screen fingerprints. This is inspired by the archive-and-return
idea in [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9), not an
implementation of its full algorithm or demonstration-based robustification.
Only states reached by the current learner are retained; no external gameplay,
scripted actions, hidden-state labels or extra rewards are introduced.

The archive key decodes the latest frame's graphics characters, omits the HUD
row, averages the remaining 45×128 binary raster into 9×16 blocks, quantizes
each block to eight levels, then hashes the result. Text is absent from this
graphics-only key. The **policy input remains all four original screen frames**,
including text; this encoding is used only to group training reset states.
The default `--curriculum-cells score` retains the previous behavior.

Screen changes are sampled every `--curriculum-screen-interval` actions
(default 32). Each stage retains at most `--curriculum-bins` distinct cells,
using the smallest deterministic hash priorities as a score-independent,
bounded sample. Per-cell snapshots use reservoir sampling. This fixed
representation and bounded selection are implementation choices, not claims
to reproduce the paper's cell-selection scheme. Ordinary reset selection is
uniform across retained stages, cells and snapshots. A lookback, if enabled,
uses the actual earlier snapshot's screen key and score baseline.

All **190 regression tests** passed, including native snapshot restoration,
reward-free screen-cell collection, unchanged full-game screens/rewards,
peer routing, protected boot workers, bounded order-independent cell retention
and exact graphics encoding. No learned stage-2 or mission completion is
established by these tests.

A [bounded integration run](results/defense/training/screen-cells-smoke-01/resume-config.json)
trained for **16,384 new actions** from run 10's strongest-mean checkpoint.
It used four workers (one protected boot worker), 128 cells per stage, one
snapshot per cell, 32-action sampling and lookback, and reset probability 1
for the other workers. It logged 376 archive events covering 243 distinct
screen keys, including 105 events from restored play; four boot games and one
restored segment completed with exact visible-score reward accounting.
The [subsequent ten complete games](results/defense/training/screen-cells-smoke-01/evaluation.json)
averaged **8,009**, median **7,755**, best **10,280**, all stage 1. All **2,478**
actions of its [replay](results/defense/training/screen-cells-smoke-01/replay/replay.html)
were reproduced after reloading the model. This checks integration, not
improvement: performance is below the parent. Its model, optimizer and log are
preserved, and this smoke/probe source is excluded from the best collector.

### Diverse-screen curriculum: run 13

`defense-ppo-13-screen-cells` starts from the same run-10 checkpoint at counter
**9,046,784** as run 12. Relative to that continuing lookback comparison, it
changes archive grouping to screen cells, capacity from 16 score bins to 128
screen cells per stage, and snapshots per cell from four to one. Screen-key
changes are sampled every 32 actions. These are several archive changes, not
a single-parameter comparison. Both trials retain a 32-action lookback, GAE
lambda 0.99, entropy 0.002, 32 workers, 256-action rollouts, 50% reset probability
for eligible workers, and eight protected boot-only workers. The
[configuration](results/defense/training/ppo-13-screen-cells/resume-config.json)
records the exact settings and new implementation hashes.

The archive starts empty and uses only new own-play states. The smoke model
and archive are not reused. There are no wall-clock, episode-action or total
training-action limits. Ordinary ten-game evaluations start from boot, and
the single verification collector now includes this trial's isolated artifact
source. This run replaces the stopped exploration trial, not the preserved
best model or the continuing run 12.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-screen-cells-reproduction \
  --resume results/defense/training/ppo-10-long-credit/step-000009046784 \
  --artifacts runs/defense-screen-cells-reproduction/artifacts \
  --curriculum-lookback 32 --curriculum-cells screen --curriculum-bins 128 \
  --curriculum-per-bin 1 --curriculum-screen-interval 32
```

After **106,496 additional actions**, run 13's
[first ten complete validation games](results/defense/training/ppo-13-screen-cells/first-validation.json)
averaged **9,362**, median **8,960**, best **10,480**, all stage 1 and no mission.
Its local replay reproduced all **2,505** neural actions; the
[verification record](results/defense/training/ppo-13-screen-cells/first-verification.json)
is retained. This ties the shared best score but is below the parent mean;
it is not evidence of improvement. The full run continues unchanged.

Run 13 subsequently stopped gracefully at counter **10,570,496**, after
**1,523,712 additional actions**, 458 new completed boot games, 302 completed
restored segments and 15 ten-game validations. It tied 10,480 but never
exceeded it or reached stage 2. Its strongest mean was **10,422** (median
10,445) at **9,849,600**, below its parent's mean of 10,473; the final
validation mean was 10,271. The
[full log](results/defense/training/ppo-13-screen-cells/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-13-screen-cells/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-13-screen-cells/step-000009849600/state.json)
and [verified best replay](results/defense/training/ppo-13-screen-cells/best-effort/replay.html)
are preserved. This tested screen-cell configuration did not resolve the
observed bottleneck; the optional implementation remains available.

### Longer-return PPO: run 14

`defense-ppo-14-long-horizon` replaces the stopped run 12, while run 13 continues.
It resumes the preserved run-12 checkpoint at **10,447,616**, changing three
training settings together: rollout **256 → 1,024**, discount factor
**0.997 → 0.999**, and GAE lambda **0.99 → 0.999**. All other learning/game
settings are inherited, including score-bin curriculum, 32-action lookback,
eight boot-only workers, ship-loss learning boundaries, screen-only input and
visible-score-only reward. The newly introduced screen-cell mode stays off.
The [saved configuration](results/defense/training/ppo-14-long-horizon/resume-config.json)
records the settings and backward-compatible curriculum implementation hash.
There are no episode, total-action or wall-clock limits.

The motivation is longer credit assignment, not a proven diagnosis. In the
preserved 10,480-point trace, the four ship segments last 410, 723, 730 and
717 actions (later segments include inter-ship animations), each scoring
2,620. The old untruncated GAE residual kernel `(gamma*lambda)^k` has a
53.1-action half-life; the new one has a 346.4-action half-life. Actual direct
credit is also cut off by rollout ends and life boundaries; learned value
bootstrapping can carry information farther. Longer traces increase variance.
These calculations do not identify a route or show that longer credit will
escape the observed local behavior.

Testing different return horizons is also motivated by
[Agent57's discussion of long-term credit](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/).
This trial remains PPO: it does not implement Agent57, use its intrinsic rewards,
or introduce a meta-controller. All **191 regression tests** passed, including
analytic 1,024-step GAE weights, terminal masking and bootstrap propagation
at the new parameters. The same single collector includes this run and keeps
the previous verified best available until a genuine improvement is verified.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-horizon-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-long-horizon-reproduction/artifacts \
  --gamma .999 --gae-lambda .999 --rollout 1024
```

After four long rollouts (**131,072 additional actions**), run 14's
[first ten complete evaluation games](results/defense/training/ppo-14-long-horizon/first-validation.json)
averaged **10,208**, median **10,480**, best **10,480**, all stage 1 and no
mission. Its replay reproduced all **2,580** neural actions; the
[verification record](results/defense/training/ppo-14-long-horizon/first-verification.json)
is preserved. This is below its parent's mean and does not establish a gain.
The original shared best remains unchanged, and both full trials continue.

Run 14 subsequently stopped gracefully at **12,512,000**, after **2,064,384
additional actions**, 600 new completed boot games, 407 completed restored
segments and 20 ten-game validations. It never exceeded 10,480 or reached
stage 2. Its strongest mean was **10,458**, median 10,460, at **10,775,296**;
the final validation mean was 9,170. The
[full log](results/defense/training/ppo-14-long-horizon/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-14-long-horizon/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-14-long-horizon/step-000010775296/state.json)
and [verified best replay](results/defense/training/ppo-14-long-horizon/best-effort/replay.html)
are retained. The tested longer-return configuration did not resolve this
trial's bottleneck; this does not rule out other horizons or longer training.

### Optional persistent policy-bias exploration

`--policy-bias-noise STD` defaults to zero. When enabled, each training worker
draws independent zero-mean Gaussian offsets for the learned actor's output
bias parameters. The offsets persist across actions and rollout boundaries,
and are redrawn on a visible ship loss or episode boundary. They do not choose
an action, identify an obstacle, or encode a route. The learned screen-dependent
logits plus the sampled parameter offsets define the categorical policy.

The method is a restricted adaptation of
[parameter-space exploration](https://arxiv.org/abs/1706.01905): only output
biases are perturbed, with a fixed user-selected scale. It is not the paper's
whole-network, adaptive-scale algorithm. PPO retains each sample's offset and
uses that same offset in the updated log probability, ratio, entropy and KL
calculation. Noise is not optimized as a parameter. The value baseline remains
screen-only and unperturbed, which may increase estimation error/variance.
The separate noise RNG is saved; resume restarts emulator episodes and draws
fresh offsets from that RNG. No additional observation or reward is supplied.
Combining this option with the separate SIL path is rejected as untested.

Evaluation and published replays use the **unperturbed learned model**. Thus a
good noisy training episode cannot replace the standard-policy best on its
own. All **198 tests** passed, covering persistent and selective redraws, RNG
resume, zero-noise equivalence, matching sampling/learning likelihoods, finite
updates and unchanged ordinary sampling. Existing full runs 13 and 14 started
before this implementation and continue without policy-bias noise.

A separate [four-worker integration run](results/defense/training/bias-noise-smoke-01/resume-config.json)
used standard deviation 1 for **16,384 new actions**, starting from run 12's
strongest-mean model. Four boot games and one restored segment completed; the
noise RNG and optimizer are retained in the
[checkpoint](results/defense/training/bias-noise-smoke-01/checkpoint/state.json).
Its [ten unperturbed complete evaluation games](results/defense/training/bias-noise-smoke-01/checkpoint/evaluation.json)
averaged **10,201**, median **10,460**, best **10,480**, all stage 1. All **2,482**
actions of its [replay](results/defense/training/bias-noise-smoke-01/replay/replay.html)
were reproduced. This validates integration, not improved performance; the
smoke artifact source is excluded from the production collector.

### Persistent exploration: run 15

`defense-ppo-15-bias-noise` resumes the same run-12 checkpoint at counter
**10,447,616** used to start run 14, but retains its original 256-action
rollouts, discount 0.997 and GAE lambda 0.99. Its sole changed learning
parameter relative to that parent is **policy-bias noise 0 → 1**. The
[configuration](results/defense/training/ppo-15-bias-noise/resume-config.json)
records this and the tested implementation hashes. The previously absent
screen-cell setting retains its score-bin default; reward, observations,
controls, action timing and optimizer state are unchanged.

The smoke checkpoint is not reused. This is a third independent full learner,
with 32 workers, eight protected boot workers, 50% curriculum resets for
eligible workers, newly collected own-state archives and no action/time caps.
Runs 13 and 14 continue unchanged. Its complete-game evaluation uses the
unperturbed policy, and its isolated artifact source is included in the single
verification collector. Memory pressure and aggregate throughput are monitored
with the extra learner active; noisy training scores are not standard-policy
validation results and cannot directly promote the best replay.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-bias-noise-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-bias-noise-reproduction/artifacts --policy-bias-noise 1
```

After **106,496 additional actions**, run 15's
[first ten complete unperturbed games](results/defense/training/ppo-15-bias-noise/first-validation.json)
averaged **10,440**, median **10,460**, best **10,480**, all stage 1 and no mission.
Its replay reproduced all **2,501** neural actions; the
[verification record](results/defense/training/ppo-15-bias-noise/first-verification.json)
is preserved. This is below its starting mean of 10,474, not a gain. Initial
three-learner monitoring found roughly 1,670 aggregate training actions/second,
stable swap usage on the follow-up check and 37% reported free memory; these
are short observations, not a hardware benchmark. All three trials continue,
with the original verified best unchanged.

### Stronger persistent exploration: run 16

`defense-ppo-16-strong-bias-noise` replaces stopped run 13. It starts from
exactly the same counter-**10,447,616** checkpoint as the continuing run 15,
changing only **policy-bias noise standard deviation 1 → 2** relative to that
trial. The [saved configuration](results/defense/training/ppo-16-strong-bias-noise/resume-config.json)
was compared directly with run 15: apart from paths, that scalar is the sole
difference. It uses the same tested implementation, 32 workers, eight protected
boot workers, score-bin/lookback curriculum, ordinary score reward and uncapped
complete-game evaluation. There are no total-action or wall-clock limits.

The hypothesis is that larger, temporally consistent perturbations will sample
more different behaviors than scale 1. This is not evidence of better play and
may instead disrupt already learned behavior. Only a complete unperturbed
evaluation and verified replay can promote the shared best. Runs 14 and 15
continue unchanged, and the single collector includes run 16 while retaining
the earlier artifact sources.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-strong-bias-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-strong-bias-reproduction/artifacts --policy-bias-noise 2
```

After **106,496 additional actions**, run 16's
[first ten unperturbed complete games](results/defense/training/ppo-16-strong-bias-noise/first-validation.json)
averaged **10,314**, median **10,360**, best **10,480**, all stage 1 and no mission.
Its [verification record](results/defense/training/ppo-16-strong-bias-noise/first-verification.json)
confirms all **2,531** replay actions. This is below the common parent's mean
10,474 and run 15's first mean 10,440; no improvement is established. The
trial continues, and the existing shared best has not been replaced.

Run 16 subsequently stopped gracefully at **12,512,000** actions after
**2,064,384 new actions**, **653** additional complete boot games, **400**
restored training segments and **20** complete ten-game evaluations. No
training or evaluation record reached stage 2 or completed a mission. Best
remained **10,480**; peak mean was **10,455**, median **10,460**, at counter
**10,955,520**. The last evaluation at **12,454,656** averaged **10,434**,
median **10,440**, best **10,480**. The stronger fixed bias perturbation did
not escape the observed plateau in this trial. Its
[full log](results/defense/training/ppo-16-strong-bias-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-16-strong-bias-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-16-strong-bias-noise/step-000010955520/evaluation.json)
and [2,531-action verified replay](results/defense/training/ppo-16-strong-bias-noise/best-effort/replay.html)
are preserved. The fresh-seed and smaller output-weight-noise learners remain
active; the collector retains the stopped trial's immutable source.

### Independent initialization: run 17

`defense-ppo-17-fresh-seed` replaces stopped run 14. The checkpoint ancestry of
the continuing noise trials runs through 12 → 10 → 9 → 8 → 6 → 2 → 1, ending
at the original seed-41 initialization. Run 17 instead starts at **counter 0**
with **seed 73**, random network weights, a fresh optimizer and empty
own-experience archives. It loads no previous model or gameplay data. This
tests a new learning trajectory, not a continuation or a claim of immediate
improvement over the already trained models.

It uses the established 32-worker, 256-action-rollout settings: entropy 0.002,
discount 0.997, GAE lambda 0.99, ship-loss learning boundaries, score-bin
curriculum with 32-action lookback, 50% resets for eligible workers and eight
protected boot-only workers. Policy-bias noise and SIL are disabled. Input,
reward and the original game remain unchanged; complete evaluation uses the
same ten validation seeds. The [configuration](results/defense/training/ppo-17-fresh-seed/config.json)
records all settings. There are no action or wall-clock limits, and this
fresh model needs time to learn the early game again. The two established
noise trials continue alongside it; the collector includes all three and
preserves the existing best until a better verified result exists.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-fresh-seed-reproduction \
  --artifacts runs/defense-fresh-seed-reproduction/artifacts --seed 73 \
  --rollout 256 --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8 \
  --curriculum-lookback 32
```

At **106,496 training actions**, run 17's
[first ten complete games](results/defense/training/ppo-17-fresh-seed/first-validation.json)
averaged **308**, median **300**, best **320**, all stage 1 and no mission.
Its [first replay](results/defense/training/ppo-17-fresh-seed/first-replay/replay.html)
reproduced all **1,604** neural actions; the
[optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/step-000000106496/state.json)
and full replay/model bundle are preserved as this independent run's baseline.
This is early learning from random initialization, far below the established
best, not a replacement for that model or evidence of stage progress.

At **1,900,544** actions, the fresh-seed learner's ten complete games averaged
**434**, median **440**, best **460**, still stage 1. That
[optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/step-000001900544/evaluation.json)
is preserved as an intermediate improvement over its 308-point initial
validation mean. Its first 460-point effort occurred at **1,400,832**; that
[checkpoint](results/defense/training/ppo-17-fresh-seed/step-000001400832/evaluation.json)
and its [verified replay](results/defense/training/ppo-17-fresh-seed/replay-460/replay.html)
are also preserved. These are two different models: the replay belongs to
the earlier first-best checkpoint, not the later higher-mean one. Neither
approaches the established 10,480-point global best. This independent learner
continues from its own experience, with no prior model or demonstration data.

### Optional state-dependent output-weight exploration

Run 15 (bias-noise SD 1) stopped gracefully at **12,708,608** actions after
**2,260,992 new actions**, **696** additional complete boot games, **497**
restored training segments and **22** complete ten-game evaluations. Neither
training nor evaluation reached stage 2. Best remained **10,480**; peak mean
was **10,470**, median **10,480**, at counter **11,455,232**. Its last evaluation
at **12,651,264** averaged **9,829**, median **10,470**, best **10,480**.
The [full log](results/defense/training/ppo-15-bias-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-15-bias-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-15-bias-noise/step-000011455232/evaluation.json)
and [2,501-action verified best replay](results/defense/training/ppo-15-bias-noise/best-effort/replay.html)
are preserved. This is a plateau in the tested configuration, not evidence
that every parameter-noise method fails. The stronger-bias and independent
fresh-seed learners continue unchanged.

`--policy-weight-noise STD` is a training-only alternative to bias noise; both
default to zero and cannot be combined. Each worker draws a Gaussian matrix
with the same shape as the learned actor's output weights, and retains it
until a visible life/episode boundary. For screen features `h`, the policy
logits are `(W + delta_W) h + b`. Unlike a constant bias offset, the effect
therefore depends on the screen. The value head is unperturbed. No action is
selected by a script, and no game data other than the screen is introduced.

This extends the restricted
[parameter-noise adaptation](https://arxiv.org/abs/1706.01905) to the actor's
output weight matrix, not the whole network. Its scale is fixed, not adaptive;
it is not a full reproduction of that paper or NoisyNet. Evaluation, model
exports and the shared-best collector still use the unperturbed learned
network. The stored weight format and default prediction path are unchanged.

PPO stores each actual life draw once in a rollout bank, with time-major
worker indices for sampling and shuffling; it does not duplicate the full
matrix for every observation. The original draw is reused when computing
the new policy likelihood. Gradients pass through the screen encoder exactly
as if the actor weights had been perturbed directly; noise itself is constant
for differentiation. The independent noise RNG is checkpointed, with fresh
life draws on resume. SIL combinations are rejected as untested.

All **201 regression tests** passed, including exact perturbed-weight
likelihood/gradient comparisons for every model parameter, finite updates,
rollout-bank ordering and snapshot isolation, zero-noise equivalence, and
invalid-mode checks. The stricter gradient-tree comparison was also rerun
separately. Existing production learners started before this change and
continue using their original settings.

Two isolated four-worker checks resumed run 12's 10,447,616-action checkpoint
and each trained for 16,384 additional actions. They used rollout 256, batch
256, one protected boot-only worker and otherwise inherited the parent
settings. Their unperturbed evaluations each contain ten complete games:

| Output-weight noise SD | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0 (matched control)](results/defense/training/weight-noise-control-01/checkpoint/evaluation.json) | 9,929 | 10,460 | 10,480 | 2,554 |
| [0.02](results/defense/training/weight-noise-smoke-01/checkpoint/evaluation.json) | 324 | 320 | 380 | 1,703 |
| [0.005](results/defense/training/weight-noise-smoke-02/checkpoint/evaluation.json) | 7,668 | 7,920 | 10,460 | 2,523 |

Both remained in stage 1 with no successful mission. These are regressions
from the parent's 10,474 validation mean, not successful exploration results.
The smaller noise did less harm in this small test, but it did not improve
the model. Full logs, configurations, optimizer checkpoints and verified
replays are preserved, including the failed larger-noise trial. Neither is
a collector source or a replacement for the global best. The matched no-noise
control also regressed, but retained much more performance than either noisy
trial. It used the same parent, seed, action count and training settings with
both noise options disabled. Its full checkpoint and verified replay are also
preserved. These single-seed checks do not establish general causality or
predict the outcome with the normal 32-worker training setup.

A subsequent [32-worker calibration](results/defense/training/weight-noise-calibration-01/resume-config.json)
used noise SD **0.005**, the same parent, and the parent's normal rollout 256,
batch 512, eight boot-only workers and all other learning settings. After
**32,768 new actions** (four rollouts), its ten complete unperturbed games
averaged **10,453**, median **10,460**, best **10,480**, still stage 1.
Its [replay](results/defense/training/weight-noise-calibration-01/replay/replay.html)
reproduced **2,553** actions. This retained the parent's performance much
better than the four-worker check; it is not an improvement or evidence that
noise helps exploration. The complete calibration log, optimizer and replay
bundle are preserved separately and excluded from automatic promotion.

`defense-ppo-18-weight-noise` continues from that calibration's optimizer at
counter **10,480,384**, with no training or episode action limit and complete
ten-game evaluations every 100,000 training actions (rounded to rollout
boundaries). Its [configuration](results/defense/training/ppo-18-weight-noise/resume-config.json)
retains noise SD 0.005 and the normal 32-worker settings. Resume restores the
policy, optimizer and RNGs but restarts emulator episodes and its own-state
archive; it draws fresh life perturbations. Evaluation remains unperturbed.
The sole collector includes this run alongside all previous sources, and
only verified strictly better efforts can replace the shared best.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-weight-noise-reproduction \
  --resume results/defense/training/weight-noise-calibration-01/checkpoint \
  --artifacts runs/defense-weight-noise-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Run 18's [first complete evaluation](results/defense/training/ppo-18-weight-noise/first-validation.json),
at **10,586,880** actions (**106,496** additional actions after calibration),
averaged **10,193**, median **10,460**, best **10,480**. All ten games ended
in stage 1 without a mission. Its [verification record](results/defense/training/ppo-18-weight-noise/first-verification.json)
confirms **2,502** reproduced actions; the source checkpoint and replay hashes
match. This is an early continuation result, not a new best. The full source
checkpoint and replay remain in the local run directory; the calibration's
complete bundle is committed above. The existing shared best remains unchanged.

Run 18 subsequently stopped gracefully at **12,233,472** actions after
**1,753,088** additional actions beyond calibration, **535** complete boot
games, **342** restored training segments and **17** ten-game evaluations.
No training or evaluation record reached stage 2 or completed a mission.
Best remained **10,480**; peak mean was **10,460**, median **10,460**, at
**12,086,016**. Its last evaluation at **12,184,320** averaged **8,520**,
median **9,145**, best **10,460**. The observed depth plateau, rather than a
wall-clock limit or a claim that this method can never work, prompted a new
experiment in its slot. The [full log](results/defense/training/ppo-18-weight-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-18-weight-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-18-weight-noise/step-000012086016/evaluation.json)
and [2,502-action verified best replay](results/defense/training/ppo-18-weight-noise/best-effort/replay.html)
are preserved. The larger-noise trial and fully fresh learner continue.

The larger SD **0.02** was also checked for four rollouts with the same normal
32-worker setup and the same run-12 parent, rather than inferring its behavior
solely from the failed four-worker check. Its
[ten complete unperturbed games](results/defense/training/weight-noise-calibration-02/checkpoint/evaluation.json)
averaged **7,783**, median **8,180**, best **10,480**, all stage 1. The
[replay](results/defense/training/weight-noise-calibration-02/replay/replay.html)
verified **2,520** actions. Full logs, configuration, optimizer and replay are
preserved, and this calibration is excluded from the collector. It was still
substantially more disruptive than SD 0.005 in this comparison, so no long
SD-0.02 run was launched. These are short, single-seed validation comparisons,
not independent success-rate estimates or proof about eventual learning.

The midpoint SD **0.01** passed the same four-rollout check: its
[ten complete games](results/defense/training/weight-noise-calibration-03/checkpoint/evaluation.json)
averaged **10,462**, median **10,480**, best **10,480**, all stage 1. Its
[replay](results/defense/training/weight-noise-calibration-03/replay/replay.html)
verified **2,524** actions. The full checkpoint, configuration, log and replay
are preserved. This retained performance in the short calibration, not a
stage advance or a demonstrated exploration benefit.

`defense-ppo-19-moderate-weight-noise` now continues from that calibration's
optimizer at **10,480,384**, with the same unlimited 32-worker settings and
100,000-action evaluation interval as run 18, but noise SD **0.01** instead of
0.005. Each trial has its own calibrated parent and RNG history. The
[configuration](results/defense/training/ppo-19-moderate-weight-noise/resume-config.json)
records this provenance. The collector includes both trials and the fresh
seed learner, with the same strict replay-verification gates. No larger-noise
regression was substituted for the shared best.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-moderate-weight-reproduction \
  --resume results/defense/training/weight-noise-calibration-03/checkpoint \
  --artifacts runs/defense-moderate-weight-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Run 19's [first ten complete games](results/defense/training/ppo-19-moderate-weight-noise/first-validation.json),
after **106,496** additional actions, averaged **9,728**, median **10,370**,
best **10,480**, all stage 1 without a mission. Its
[verification record](results/defense/training/ppo-19-moderate-weight-noise/first-verification.json)
confirms **2,580** reproduced actions. The local first-evaluation checkpoint
and immutable replay bundle have matching weights and validated manifests;
the full calibration bundle is committed above. This early result is below
its starting mean, not an improvement; training continues without replacing
the global best on a tied score.

### Read-only policy/value-gradient diagnostic

A [frozen-model diagnostic](results/defense/diagnostics/gradient-probe-8342272.json)
used the global-best model's own 2,580-action replay to compare its policy plus
entropy gradient with its weighted value-loss gradient. It reconstructed only
screen stacks, visible-score rewards and visible life boundaries; no optimizer
update occurred, and every model parameter was checked unchanged. In ten full
256-action trajectory blocks, the shared-encoder value/policy norm ratio was
**0.46–5.34**, median **1.45**. This does not show overwhelming value-gradient
dominance throughout that selected trajectory, and is not a diagnosis of the
plateau. It is not representative live training data: per-block normalization,
one selected game and the older checkpoint's GAE lambda 0.95 differ from the
continuing learners. The final 20-action partial block is recorded separately.
Reward scaling, optimizer and architecture remain unchanged on this evidence.

### Optional fresh decision heads with the learner's own screen encoder

`--initialize-encoder CHECKPOINT_DIRECTORY` starts a **new** Defense learner
using only the convolutional layers and shared 256-unit screen-feature layer
from an earlier own-trained Defense checkpoint. Both actor and value heads
remain freshly randomized from the new seed; the usual initial actor scale
of 0.1 still applies. The optimizer, action/episode counters, policy RNG,
emulator episodes and own-state archives are fresh. No recorded actions,
trajectories or opaque native snapshots are loaded. All encoder parameters
remain trainable, and observations/rewards/action selection are unchanged.

This is a proposed way to test a different decision policy without discarding
all learned visual features. It is inspired by partial-reset work such as
[Nikishin et al. (2022)](https://proceedings.mlr.press/v162/nikishin22a.html),
but **not a reproduction**: that paper retains replay data, while this PPO
experiment retains only learned encoder weights and starts new on-policy
experience. A fresh action counter reports additional training, not the
total cost including pretraining. Source checkpoint/state hashes and the
source's training-action count are explicitly recorded in configuration.
The checkpoint must match Defense's game, environment, action profile,
parameter names, shapes and dtypes; non-finite encoder weights and a source
that changes during loading are rejected before copying anything.

Initialization and optimizer resume are mutually exclusive. A later `--resume`
continues the new learner normally and preserves its ancestry without
reapplying initialization or requiring the old source path to remain present.
Without the new option, fresh-training and resume behavior are unchanged.
Initialization is applied only when requested for a new learner; it does not
alter already-running learners.

Before considering neuron recycling, a separate
[read-only activity probe](results/defense/diagnostics/activation-probe.json)
measured every policy-input screen from four frozen models' own replays.
Only **4–6 of 256** hidden units were never active in each sample; none of the
convolutional channels was entirely inactive. This does not establish a
widespread dead-neuron failure or rule out other representation problems.
The statistic is inspired by
[Sokar et al. (2023)](https://proceedings.mlr.press/v202/sokar23a.html); **ReDo
has not been implemented or used for training**. Selected replay coverage is
not a representative sample of every state the model could encounter.

All **206 regression tests** passed, including exact encoder-copy/head-
preservation checks, unchanged optimizer/source files, malformed or changing
source rejection, and a real-emulator fresh-start/resume test. An isolated
[four-worker integration run](results/defense/training/encoder-transfer-smoke-01/config.json)
used seed 73 and run 12's encoder at source counter 10,447,616, with fresh
heads/optimizer and its own counter starting at zero. After **16,384 new
actions**, its [ten complete games](results/defense/training/encoder-transfer-smoke-01/checkpoint/evaluation.json)
averaged **282**, median **280**, best **300**, all stage 1 with no mission.
Its [replay](results/defense/training/encoder-transfer-smoke-01/replay/replay.html)
reproduced **1,536** neural actions. Full logs, configuration, checkpoint and
replay are preserved. This verifies integration, not improved learning or
preservation of the old policy's performance. The run is excluded from the
collector. It preceded the full-size trial described below.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-encoder-transfer-check \
  --initialize-encoder results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-encoder-transfer-check/artifacts --seed 73 \
  --envs 4 --rollout 256 --batch-size 256 --entropy .002 --gae-lambda .99 \
  --life-terminal --curriculum-probability .5 --curriculum-share \
  --curriculum-boot-envs 1 --curriculum-lookback 32 --mlx-cache-mb 256 \
  --eval-envs 4 --eval-every 16384 --steps 16384
```

`defense-ppo-20-encoder-transfer` replaces stopped run 18. It initializes
directly from run 12's encoder at source counter **10,447,616**, not from the
four-worker smoke model. Its own counter starts at **zero**. The
[configuration](results/defense/training/ppo-20-encoder-transfer/config.json)
uses the same seed **73**, fresh-head initialization, 32 workers, rollout 256,
batch 512, entropy 0.002, discount 0.997, GAE lambda 0.99, life boundaries and
score-bin/lookback curriculum settings as run 17. Neither uses policy noise
or SIL. Apart from paths and initialization provenance, the only new config
field difference is an explicit zero for weight noise, which was absent
(and disabled) in run 17's earlier code. All copied features remain trainable.

This is a comparison of fresh versus own-pretrained visual features, not
training from scratch at the same total interaction cost: the source encoder
already embodies earlier learning. No output head, optimizer history,
trajectory or native state is transferred. Training and complete-game
evaluation are uncapped. The collector includes the new trial and retains
all previous artifact sources; the global best has not been replaced.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-encoder-transfer-reproduction \
  --initialize-encoder results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-encoder-transfer-reproduction/artifacts --seed 73 \
  --rollout 256 --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8 \
  --curriculum-lookback 32
```

Run 20's [first ten complete games](results/defense/training/ppo-20-encoder-transfer/step-000000106496/evaluation.json)
at **106,496 additional actions** averaged **302**, median **300**, best **320**,
all stage 1. Its [replay](results/defense/training/ppo-20-encoder-transfer/first-replay/replay.html)
verified **1,603** neural actions, and the full optimizer/model bundle is
preserved. Run 17's corresponding first mean was 308 with the same 320 best;
this early single-seed comparison establishes **no transfer advantage**.
The decision heads are learning anew, and the source encoder's 10,447,616
earlier actions must not be omitted when discussing training cost. Both
fresh-policy trials continue without replacing the stronger global model.

Run 17 subsequently reached **600** at its own counter **3,604,480**:
[ten-game mean 570, median 580](results/defense/training/ppo-17-fresh-seed/step-000003604480/evaluation.json),
all stage 1. The full optimizer/model and a
[2,004-action verified replay](results/defense/training/ppo-17-fresh-seed/replay-600/replay.html)
are preserved alongside the earlier 460-point checkpoint. Run 20 reached
**460** at its own counter **704,512**, with
[mean 380, median 360](results/defense/training/ppo-20-encoder-transfer/step-000000704512/evaluation.json)
and a [1,673-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-460/replay.html).
These are different training ages, not a matched performance comparison.
Both are intermediate improvements within their own trials, below the shared
10,480-point policy; neither has reached stage 2 or a successful mission.

At its own counter **1,400,832**, run 20 improved to
[mean 444, median 440, best 480](results/defense/training/ppo-20-encoder-transfer/step-000001400832/evaluation.json).
Its full optimizer/model and [1,734-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-480/replay.html)
are preserved. Run 17 at the same fresh-action counter had mean **404**,
median **400**, best **460** on these validation seeds. This one-seed,
reused-validation comparison is a small local advantage for encoder transfer,
not evidence of better final performance or lower total training cost: run 20
also used an encoder pretrained for 10,447,616 actions. Both remain in stage 1.

Later saved milestones are run 17's **1,370**-point effort at **4,603,904**
actions ([mean 615, median 520](results/defense/training/ppo-17-fresh-seed/step-000004603904/evaluation.json),
[2,002 verified replay actions](results/defense/training/ppo-17-fresh-seed/replay-1370/replay.html))
and run 20's **580** at **1,802,240** new actions
([mean 540, median 550](results/defense/training/ppo-20-encoder-transfer/step-000001802240/evaluation.json),
[1,958 verified replay actions](results/defense/training/ppo-20-encoder-transfer/replay-580/replay.html)).
Full model/optimizer bundles accompany both. The isolated 1,370 effort did not
persist in subsequent run-17 batches, whose best scores returned to around 600;
it is a capability observation, not reliable performance. Neither trial has
advanced beyond stage 1, and both continue without changing the shared best.

Run 20 reached **600** at **2,203,648** new actions:
[ten-game mean 552, median 580](results/defense/training/ppo-20-encoder-transfer/step-000002203648/evaluation.json).
The full optimizer/model and [1,985-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-600/replay.html)
are preserved. All ten games remained in stage 1 without a mission.

Run 17's next rare breakthrough was **2,890** at **6,504,448** actions:
[ten complete games](results/defense/training/ppo-17-fresh-seed/step-000006504448/evaluation.json)
averaged **791**, median **560**, all stage 1. Its
[verified replay](results/defense/training/ppo-17-fresh-seed/replay-2890/replay.html)
reproduced **2,102** actions. The next batch at 6,602,752 returned to mean
**568**, median **570**, best **580**: this is not yet stable higher-scoring play.
Run 20 reached **640** at **3,301,376** new actions, with
[mean 582, median 580](results/defense/training/ppo-20-encoder-transfer/step-000003301376/evaluation.json)
and [2,081 verified actions](results/defense/training/ppo-20-encoder-transfer/replay-640/replay.html).
Both full model/optimizer checkpoints are preserved. Neither reached stage 2
or replaced the stronger shared best; the counters differ and run 20's encoder
also includes the previously documented pretraining cost.

Run 20 subsequently stopped normally at **3,817,472** new actions after
**1,656** complete boot games, **1,058** restored segments and **38** ten-game
validations. Its last mean/median/best were **580 / 580 / 600** at 3,801,088;
peak mean was **592**, median **600**, best **600** at 2,506,752. The
[final optimizer](results/defense/training/ppo-20-encoder-transfer/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-20-encoder-transfer/step-000002506752/evaluation.json)
and [full log](results/defense/training/ppo-20-encoder-transfer/metrics.jsonl)
are preserved alongside the earlier 640-point replay. No real training or
evaluation game reached stage 2 or a mission. The repeated score/depth plateau,
not a wall-clock deadline, motivated assigning its slot to bootstrap DQN.
The fresh-seed PPO lineage continues independently.

A [screen-encoding audit](results/defense/diagnostics/screen-encoding-audit.json)
also checked all **2,581** frames of the global-best replay. Its **112** distinct
character codes all fall within the encoder's graphics or ASCII ranges.
All **64** semigraphics glyphs matched the actual emulator font at their six
block interiors when compared with the model's rendering table. No dropped
character or graphics-bit-layout mismatch was found. This covers the recorded
stage-1 game, not unseen stages or the quality of learned visual features;
the input pipeline was left unchanged.

### Broader frozen-policy validation

The unchanged global-best model at counter **8,342,272** was evaluated on
[100 additional complete games](results/defense/validation/step-8342272-seeds-30000-30099.json),
seeds **30000–30099**, with eight workers, temperature 1, original 100,000-
T-state action duration and no episode action cap. Mean score was **9,482.3**,
median **10,280**, best **10,480**. **Zero** games reached stage 2, stage 3 or a
successful mission; all 100 ended normally at zero ships. The model's hash
was unchanged before/after evaluation. This broader batch did not reveal a
rare stage clear missed by the routine ten-game checks. It is additional
**validation**, not a held-out final test or proof that a clear is impossible.
The report remains evaluation-only and cannot automatically promote a replay.
No new best was found, and the existing verified best bundle is unchanged.

```bash
venv/bin/python -u -m rl.defense_evaluate \
  results/defense/learned/versions/step-000008342272-125346536cb1-seed-10004/model.safetensors \
  --output runs/defense-expanded-validation-reproduction.json \
  --games 100 --seed 30000 --envs 8 --max-steps 0
```

The longer-lookback run 21 later reached a ten-game mean **10,478**, median
**10,480**, best **10,480**, at counter **11,750,144**. Its
[full frozen checkpoint](results/defense/training/ppo-21-long-lookback/step-000011750144/evaluation.json)
was preserved and evaluated on the **same 100 broader validation seeds** above.
That [comparison](results/defense/validation/step-11750144-seeds-30000-30099.json)
gave mean **10,384.2**, median **10,480**, best **10,480**, with **63/100** games
matching that best and minimum score **7,940**. Relative to the older frozen
model on matching seeds, scores were higher in 96 games, equal in three and
lower in one; the mean difference was **901.9 points**. This is improved score
consistency on reused validation seeds, not a fresh final-test estimate or
proof of better stage-clearing ability.

All **100** games still ended at zero ships in **stage 1**, with no stage 2/3
or successful mission. Weights were hash-checked unchanged. A separate
[2,581-action verified replay](results/defense/validation/step-11750144-seeds-30000-30099-replay/replay.html)
preserves this exact model's best game; its bundle and report remain explicitly
evaluation-only and ineligible for automatic promotion. The shared best-effort
link remains unchanged on a tied score. The stronger average has not solved
the progression plateau.

### Longer own-state lead-in experiment

A [read-only replay timing check](results/defense/diagnostics/lookback-lead-in.json)
compared eligible lookback candidates within each life of the existing best
replay. The last eligible 32-action candidates were **51–63 actions** before
visible ship loss; 128-action candidates would be **147–159 actions** earlier.
These are candidate indices, not live archive selections or exact collision
times. The diagnostic created/restored no native states, and neither its
indices nor replay actions are supplied to training. The hypothesis is simply
that earlier own-reached resets leave more opportunity to change behavior.

The existing lookback option required no production-code change. A new
real-emulator test checks exact opaque snapshot/screen equality at lag 128,
bounded history, saved-state score metadata, and clearing at every ship loss
and reset. All **207 regression tests** passed. An isolated
[four-worker check](results/defense/training/lookback128-smoke-01/resume-config.json)
resumed run 12 at **10,447,616** and added **16,384** actions. Apart from paths,
its configuration differs from the earlier no-noise four-worker control only
in lookback **32 → 128**. Its 307 lagged archive events had exact 128-action
offsets; four boot games and one restored segment completed during training.

Its [ten complete validation games](results/defense/training/lookback128-smoke-01/checkpoint/evaluation.json)
averaged **9,304**, median **9,590**, best **10,480**, all stage 1, compared with
the control's mean **9,929**, median **10,460**, best **10,480**. This is not an
improvement. The [verified replay](results/defense/training/lookback128-smoke-01/replay/replay.html)
reproduced **2,578** neural actions. Full logs, optimizer and replay are
preserved; this short check is excluded from the global collector.

Run 19's moderate weight-noise trial stopped cleanly at **12,602,112** after
**2,121,728** additional actions, **636** boot games, **379** restored segments
and **21** ten-game evaluations. No observed stage advance or mission occurred.
Peak mean was **10,456**, median **10,480**, at **12,184,320**; the final
evaluation averaged **9,549**, median **10,035**, best **10,480**. Its
[full log](results/defense/training/ppo-19-moderate-weight-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-19-moderate-weight-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-19-moderate-weight-noise/step-000012184320/evaluation.json)
and [verified best replay](results/defense/training/ppo-19-moderate-weight-noise/best-effort/replay.html)
are preserved. Its observed depth plateau, not a wall-clock deadline, motivated
reassigning its slot.

`defense-ppo-21-long-lookback` now resumes directly from the same strong run-12
parent, **not** from the four-worker check. It changes lookback 32 to 128 with
32 workers, rollout 256, batch 512, discount 0.997, GAE lambda 0.99, entropy
0.002, life boundaries, shared score-bin curriculum and eight boot-only
workers. Noise and SIL remain off; training and evaluation are uncapped.
The [configuration](results/defense/training/ppo-21-long-lookback/resume-config.json)
records the current source hash and explicit defaults added since run 12.
This full-size trial tests the hypothesis despite the lower short-check mean;
no gain is claimed. Fresh-seed run 17 and own-encoder/fresh-head run 20 continue.
The sole collector retains all earlier sources and includes run 21, with the
same complete-game and frozen-policy verification gates.

Run 21's [first ten complete games](results/defense/training/ppo-21-long-lookback/step-000010554112/evaluation.json)
at **10,554,112** (**106,496** new actions) averaged **9,672**, median **10,385**,
best **10,480**, all stage 1 without a mission. Its
[first replay](results/defense/training/ppo-21-long-lookback/first-replay/replay.html)
verified **2,520** actions; the full model/optimizer checkpoint is preserved.
This is below its parent's mean and does not replace the shared best.

Run 21 eventually stopped cleanly at **12,380,928**, after **1,933,312** new
actions, **582** boot games, **394** completed restored segments and **19**
ten-game validations. Peak mean was **10,478** at **11,750,144** (the frozen
100-game comparison is above); the last evaluation averaged **10,210**,
median **10,460**, best **10,480**. No training or evaluation record reached
stage 2 or a mission. Its [full log](results/defense/training/ppo-21-long-lookback/metrics.jsonl)
and [final optimizer](results/defense/training/ppo-21-long-lookback/final-checkpoint/state.json)
are preserved in addition to the earlier checkpoints and verified replays.
The depth plateau, despite improved score consistency, motivated using its
slot for a different archive criterion; no wall-clock deadline forced the stop.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-long-lookback-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-long-lookback-reproduction/artifacts \
  --curriculum-lookback 128 --steps 0 --eval-every 100000
```

### Shorter-action learning check

An isolated [four-worker adaptation check](results/defense/training/short-action-smoke-01/resume-config.json)
resumed the same run-12 parent at **10,447,616**, changing action duration from
100,000 to **50,000 T-states**. It added **16,384 actions** with the same
settings as the earlier four-worker no-noise control; configuration differences
are only paths and action duration. This changes the policy's control cadence
without changing the original executable. It also halves the emulated-time
span of the four-frame input, rollout, lookback and unchanged per-action
discount horizon, so it is not a matched game-time or credit-horizon experiment.

Its [ten complete games](results/defense/training/short-action-smoke-01/checkpoint/evaluation.json)
averaged **5,261**, median **5,410**, best **7,370**, all stage 1 with no mission.
The [verified replay](results/defense/training/short-action-smoke-01/replay/replay.html)
reproduced **4,843** neural actions at the shorter cadence. Full configuration,
log, optimizer and replay are preserved. The check exited normally and is
excluded from the collector; the three full learners retain their timing.

To separate immediate timing mismatch from adaptation, the **exact same frozen
parent** was also evaluated at 50,000 T-states on the same ten seeds, without
parameter updates. This [parent comparison](results/defense/training/short-action-smoke-01/frozen-parent-at-50000.json)
averaged **7,939**, median **7,760**, best **10,310**, all stage 1. Its weights
were hash-checked unchanged, and the report is explicitly evaluation-only and
ineligible for promotion. Thus the brief adaptation batch reduced the mean
relative to this parent at the same cadence; it did not merely inherit the
entire observed drop from changing timing. The 100,000-T-state trained control
averaged **9,929**, median **10,460**, best **10,480**.

No boot game or restored segment finished during the short adaptation batch:
the four training episodes were still in progress at its checkpoint. Together
with its small size and changed physical-time horizons, this limits any claim
about eventual shorter-action learning. No full-size replacement was launched
on these results; the earlier-reset and fresh-policy trials continue.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-short-action-check \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-short-action-check/artifacts --tstates 50000 \
  --envs 4 --rollout 256 --batch-size 256 --curriculum-boot-envs 1 \
  --mlx-cache-mb 256 --eval-envs 4 --eval-every 16384 --steps 10464000
```

### Independent Double-DQN training path

`python -m rl.defense_dqn` provides a separate value-learning alternative to
the PPO trials. It reuses the repository's MLX dueling network, Double-Q target
calculation, prioritized replay and n-step return code, generalized to Defense's
20 actions while retaining Breakdown's six-action defaults. Method references:
[Double DQN](https://arxiv.org/abs/1509.06461),
[prioritized experience replay](https://arxiv.org/abs/1511.05952) and
[dueling networks](https://arxiv.org/abs/1511.06581). This is not Rainbow, a
bootstrapped ensemble, an exact paper reproduction or evidence of improvement.
It tests a different learning/update process after the repeated PPO depth
plateaus; no particular cause of those plateaus has been established.

Fresh DQN training starts with random weights and uses **only its own new
screen transitions**. It does not load PPO trajectories, evaluation traces,
demonstrations, emulator snapshots or pretrained features. Uniform random
epsilon-greedy exploration is training-only; complete-game evaluation and
exported replays use the **greedy learned Q-values**. The policy loader checks
the algorithm explicitly, rejects unsupported algorithms and refuses sampling-
temperature overrides for DQN. PPO sampling behavior remains unchanged, and
PPO/DQN optimizer resumes cannot be interchanged accidentally.

Defaults are eight emulator workers, batch 64, a bounded **50,000-transition**
buffer (409.6 MB for the two raw screen-stack arrays, plus small metadata),
five-step returns, discount 0.997, learning rate 0.0001, score scale 0.01 and
visible life boundaries. There is one optimizer update per 16 aggregate new
actions and a target-network copy every 2,000 updates. Replay priorities use
alpha 0.6; importance weights anneal from beta 0.4 to 1 over one million
aggregate actions. Warmup fills 10,000 transitions with random actions;
epsilon otherwise decays from 1 to 0.05 over one million actions after the
initial warmup allowance. There is no reward clipping, survival bonus,
intrinsic reward, scripted action sequence or training/evaluation action cap.

Checkpoints preserve the online network, independent target network, Adam
state, counters and RNG. Resume deliberately refills replay from new own
experience and restarts emulator episodes; it is not exact trajectory
continuation. The usual 10-game validation and reload-and-reproduce replay
gates apply. Target weights and optimizer remain in the resumable checkpoint;
watching the frozen policy requires only the online weights and its config.

All **213 regression tests** passed, including Double-Q action-selection/target-
evaluation arithmetic, 20-action updates, target isolation/synchronization,
greedy-policy round trips without RNG draws, episode-boundary/truncation
returns, malformed configuration rejection, real-emulator training/resume and
exact zero-update restoration of online, target, Adam and RNG state. Existing
PPO/Breakdown tests also passed; completed Breakdown artifacts are unchanged.

An isolated [four-worker integration run](results/defense/training/dqn-smoke-01/config.json)
started from random seed-97 weights and trained for **16,384 actions**. It used
an 8,192-transition buffer, 1,024-transition warmup and target copies every
256 updates; other learning defaults above were retained. Its
[ten complete greedy games](results/defense/training/dqn-smoke-01/checkpoint/evaluation.json)
averaged **328**, median **320**, best **360**, all stage 1 without a mission.
Its [verified greedy replay](results/defense/training/dqn-smoke-01/replay/replay.html)
reproduced **1,655** actions. Full online/target/optimizer state, logs and replay
are preserved. The run exited normally and is excluded from the collector.
This establishes end-to-end integration, not an advantage over PPO or progress
beyond the existing best. It is not initialized from the earlier PPO models.

`defense-dqn-22-fresh` now trains from random seed-97 weights with the full
defaults above, **not** from the small integration checkpoint. Its
[configuration](results/defense/training/dqn-22-fresh/config.json) preserves the
exact source hashes. This is an additional eight-worker learner, not another
32-worker PPO process; its fixed replay bound limits screen-storage growth.
Before launch, macOS reported 43% system-wide memory free. Resource use is
monitored alongside the three continuing PPO trials; no existing learner was
stopped while it was making fresh progress. The sole restarted collector now
includes this run and every previous source, but excludes the integration run.
The unchanged global-best PPO replay was also reloaded and all **2,580** actions
reproduced with the algorithm-aware loader before this launch. DQN has not yet
surpassed that model or established a stage clear.

Run 22's [first ten complete greedy games](results/defense/training/dqn-22-fresh/step-000000100000/evaluation.json)
at **100,000** fresh actions and **5,624** optimizer updates all scored **280**
(mean/median/best 280), with no stage advance or mission. Its
[first verified replay](results/defense/training/dqn-22-fresh/first-replay/replay.html)
reproduced **1,570** actions. Online/target weights and Adam state are preserved.
Training epsilon was still approximately **0.9145** at this checkpoint; greedy
evaluation contains none of that random exploration. This is an early baseline,
not an improvement over the integration run or PPO. Once the 50,000-transition
buffer filled, swap use remained near **5.3 GiB** and the other learners
continued progressing; MLX's reported DQN peak was about **296 MB** (excluding
the NumPy replay buffer and emulator processes). Resource use remains monitored.

At **400,000** actions, run 22 improved to
[mean 322, median 320, best 340](results/defense/training/dqn-22-fresh/step-000000400000/evaluation.json).
Its online/target/optimizer checkpoint and
[1,651-action verified greedy replay](results/defense/training/dqn-22-fresh/replay-340/replay.html)
are preserved. This is a small within-run improvement, still stage 1 and well
below the PPO models; it establishes no algorithm advantage.

At **900,000** actions, ordinary DQN reached
[mean 336, median 340, best 360](results/defense/training/dqn-22-fresh/step-000000900000/evaluation.json).
The full online/target/optimizer and
[1,672-action verified replay](results/defense/training/dqn-22-fresh/replay-360/replay.html)
are preserved. All ten games remained in stage 1; this remains a local
improvement, not a new global best or stage progression.

At **1,000,000**, ordinary DQN's
[mean 354, median 360, best 380](results/defense/training/dqn-22-fresh/step-000001000000/evaluation.json)
improved again. Its full checkpoint and
[1,695-action verified replay](results/defense/training/dqn-22-fresh/replay-380/replay.html)
are saved. These ten complete games still all ended in stage 1 without success.

At **1,200,000**, ordinary DQN reached
[mean 376, median 380, best 400](results/defense/training/dqn-22-fresh/step-000001200000/evaluation.json).
The full online/target/optimizer and
[1,703-action verified replay](results/defense/training/dqn-22-fresh/replay-400/replay.html)
are preserved. All ten games remained in stage 1, with no mission completion.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-reproduction \
  --artifacts runs/defense-dqn-reproduction/artifacts
# Later, resume with a fresh replay buffer from the saved online/target/Adam state:
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-continuation \
  --artifacts runs/defense-dqn-continuation/artifacts \
  --resume runs/defense-dqn-reproduction/latest
```

### Bootstrapped value exploration

The optional `rl.defense_dqn --bootstrap-heads 5` path adapts
[Bootstrapped DQN](https://arxiv.org/abs/1602.04621) and
[randomized prior functions](https://arxiv.org/abs/1806.03335). A training
worker draws one value head uniformly at a new **complete game**, retaining
it across ship losses. This tests more temporally consistent exploration
than changing random actions independently at every step. The weights still
learn during that game; the head identity, not the network parameters, stays
fixed. A configurable constant epsilon adds occasional uniform actions, and
the initial empty-replay warmup remains uniformly random.

This implementation uses shared convolutional features followed by independent
256-unit hidden and dueling output heads. A separate randomly initialized
network of the same architecture supplies fixed additive priors. The prior
parameters are frozen, excluded from Adam, saved with both online and target
networks, and restored exactly. These are random screen-to-value functions,
not game knowledge, an oracle, demonstrations or additional reward.

Each own n-step transition receives independent Bernoulli head memberships
**once when inserted**. Memberships persist through sampling and ring-buffer
reuse replaces them with the new transition's memberships. Empty memberships
are allowed. Every head's Double-Q target uses its own online action choice
and corresponding target head, including the same prior. Masked Huber losses
are averaged over active memberships with replay importance weights; a shared
priority is the mean absolute TD error over all heads. Life-terminal returns
still stop at visible loss, even though the behavior head persists for the game.

Evaluation is a fixed, deterministic **greedy mean of ensemble Q-values plus
priors**, not a favorable-head search. Complete-game evaluation, frozen-weight
action/screen/reward verification and stage/mission-first ranking are unchanged.
PPO, ordinary DQN and bootstrap optimizer resumes cannot be interchanged, nor
can the number of heads change on resume. Replay/masks refill from new own
experience; online/target/prior/optimizer and both RNG states are restored,
but new boot episodes draw new heads. The policy still sees only four raw
screen frames and training reward remains scaled visible score difference.

This is **not an exact paper reproduction**: shared visual layers, dueling
heads, prioritized n-step replay, the greedy-mean evaluation rule, our optimizer
and TRS-80 observations are explicit adaptations. With five heads, checkpoints
are substantially larger than ordinary DQN; long-run evaluation/checkpoint
frequency must account for available disk space. Default `--bootstrap-heads 0`
retains ordinary DQN behavior. The four existing production learners do not
change their loaded algorithms.

Tests cover persistent replay masks, head-specific Double-Q arithmetic,
masked head gradients, frozen priors, target synchronization, greedy ensemble
reload, invalid configurations, exact zero-update optimizer/RNG cloning,
real-emulator training/resume, and head persistence across ship losses.
The full suite passed **224 tests** after integration.

The [isolated check](results/defense/training/bootstrap-smoke-01/config.json)
completed **16,384** own actions and **960** updates, then evaluated ten complete
games: [mean/median/best all **140**](results/defense/training/bootstrap-smoke-01/checkpoint/evaluation.json),
all stage 1 losses. Its
[verified replay](results/defense/training/bootstrap-smoke-01/replay/replay.html)
reproduced **1,208** actions. The full online/target/prior/optimizer, configuration
and log are preserved. It exited normally and is excluded from the collector.
The ordinary four-worker DQN check had mean **328**, median **320**, best **360**
at the same action count: the bootstrap check is substantially worse, not an
early performance advantage. This short check establishes working integration,
not efficacy. GPU peak was approximately 454 MB, excluding replay/emulator memory.

A [read-only head-agreement check](results/defense/diagnostics/bootstrap-smoke-head-agreement.json)
reconstructed all 1,208 input stacks from that own verified replay and reproduced
every mean-policy action. Pairwise **raw-action** agreement among heads was
**14.59%**, while the mean policy chose LEFT on **998** steps. Head preferences
differed; this was not identical-head behavior on the selected trajectory.
It does **not** measure individual-head game performance, establish the cause
of the low score, or represent the new production run. Action aliases further
limit behavioral interpretation. No weights, prior scale, evaluation rule or
training data were changed; the production trial continues to its original gate.

The subsequent [matched zero-prior check](results/defense/training/bootstrap-no-prior-smoke-01/config.json)
changed **only** `bootstrap_prior_scale` from 1 to 0, plus its output paths.
Seed, architecture, independent heads, initial random-network generation,
membership probability, epsilon, replay, optimizer, 16,384-action budget,
960 updates, four workers and ten validation seeds all remained the same.
All **36 frozen prior tensors** were identical across the saved checks and the
zero-prior online/target networks; the zero scale disables their contribution,
not their construction. Subsequent learned weights and trajectories naturally
diverge as a result of the changed policy.

Its [ten complete games](results/defense/training/bootstrap-no-prior-smoke-01/checkpoint/evaluation.json)
had mean/median/best **280 / 280 / 280**, all stage 1 losses, versus
**140 / 140 / 140** with scale 1 on the same seeds. The
[verified replay](results/defense/training/bootstrap-no-prior-smoke-01/replay/replay.html)
reproduced **1,482** actions. Its full checkpoint, config and log are preserved;
it exited normally and is excluded from the collector. This favors zero prior
in this short **single-training-seed** comparison, but remains below the
ordinary DQN check's mean 328. It does not establish a long-run advantage or
justify changing the ongoing production trial before its scheduled evaluation.

To reproduce this check, use the short-check command below with
`--bootstrap-prior-scale 0` and distinct run/artifact directories.

`defense-dqn-24-bootstrap` starts fresh at counter zero with seed 97,
**five heads**, prior scale **1**, membership probability **0.5**, epsilon **0.01**
after a **10,000-transition random warmup**, eight workers, batch 64, capacity
50,000, n-step 5, discount 0.997 and target synchronization every 2,000 updates.
It does not load the smoke weights, another learner's encoder, replay traces
or native snapshots. All episode/training caps are zero. Ten-game evaluations
occur every **200,000** actions; each full checkpoint is approximately **91 MiB**,
so storage remains monitored. Its
[configuration](results/defense/training/dqn-24-bootstrap/config.json) is preserved.
This deliberately longer test of coherent exploration replaces stopped run 20;
it is not justified as already stronger. Runs 17, 22 and 23 remain active.
The single collector retains all historical sources and includes run 24, with
the same independent full-replay verification gate and unchanged global best.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-bootstrap-check \
  --artifacts runs/defense-bootstrap-check/artifacts --seed 97 \
  --envs 4 --batch-size 64 --capacity 8192 --warmup 1024 --target-every 256 \
  --steps 16384 --eval-every 16384 --eval-envs 4 --mlx-cache-mb 256 \
  --bootstrap-heads 5 --bootstrap-prior-scale 1 \
  --bootstrap-probability .5 --bootstrap-epsilon .01

# Independent unlimited production trial (fresh weights, standard DQN defaults):
venv/bin/python -u -m rl.defense_dqn --run runs/defense-bootstrap-reproduction \
  --artifacts runs/defense-bootstrap-reproduction/artifacts --seed 97 \
  --bootstrap-heads 5 --bootstrap-prior-scale 1 \
  --bootstrap-probability .5 --bootstrap-epsilon .01 --eval-every 200000
```

### Own-action-age archive experiment

`--curriculum-cells age --curriculum-age-interval 32` is an optional alternative
to score bins and coarse screen cells. It groups **actually reached** states
by visible stage and the learner's own action count since the last visible
ship-loss or stage boundary. A boot starts the counter at zero; each policy
action increments it. A real reset to an own saved state restores that state's
original counter, rather than inventing additional age or zeroing it at a new
training segment. No native payload is decoded or modified.

The motivation is to retain potentially useful states reached without an
immediate score increase. This changes **training reset selection only**:
the policy still receives the same four screen frames, the reward is still
only visible score change, and evaluation still starts from boot without an
archive. The counter is not a policy input, extra reward, hidden game timer,
position label, route or demonstration. It is also **not exact physical
survival or course progress**: visible loss can lag collision, and action
counts include intros/animations and may cover variable emulated time during
HUD settling. Later counter values need not correspond to better play.

Age-bin changes can trigger snapshots even at zero reward. The existing
lookback chooses the actual earlier same-life/stage snapshot and uses that
snapshot's age, score and baseline, not the later trigger's values. The
bounded archive keeps the largest age bins per visible stage, uses reservoir
sampling within each bin and samples stage/bin/state uniformly for a reset.
Defaults remain score bins; existing score/screen behavior and all live runs
remain unchanged unless a new learner explicitly selects this option.

All **218 regression tests** passed. New tests cover exact native/screen/age
equality at lag 128, bounded later-bin retention, counter clearing at visible
life/stage boundaries, saved-counter peer restore, malformed counters,
reward-free events and sharing without exposing snapshots to the model.
Full-game screen/reward comparisons also include age mode. Bookkeeping-only
stage-transition fixtures are unit tests, not evidence of an actual stage clear.

An isolated [four-worker check](results/defense/training/age-smoke-01/resume-config.json)
resumed run 12 at **10,447,616**, with lookback 128 and **16,384** new actions.
Relative to the earlier lookback-128 score-bin check, it changes the archive
criterion and its age interval/encoding, plus recorded source hashes and paths.
All **384** archive events had exact 128-action source/trigger offsets and
saved-age bin assignments. Four boot games completed; a restored segment
began generating own states but none had completed at the checkpoint.

Its [ten complete games](results/defense/training/age-smoke-01/checkpoint/evaluation.json)
averaged **8,456**, median **8,890**, best **10,460**, all stage 1, versus the
score-bin control's mean **9,304**, median **9,590**, best **10,480**. This is a
regression in that short comparison, not an improvement. Its
[2,552-action verified replay](results/defense/training/age-smoke-01/replay/replay.html),
full optimizer, config and log are preserved. It exited normally and is
excluded from the collector.

`defense-age-calibration-01` ([configuration](results/defense/training/age-calibration-01/resume-config.json))
tested the criterion at the normal **32-worker**
batch size, using run 21's stronger preserved parent at **11,750,144**, not the
four-worker smoke checkpoint. It retains rollout 256, batch 512, lookback 128,
eight boot-only workers and the parent's learning settings. The check permits
**131,072** new actions (4,096 per worker), so resets can actually be exercised
after complete boot games; merely filling archives would not test the proposed
mechanism. It used stopped run 21's slot and is excluded from the collector.

The check exited normally at **11,881,216**, with **32** complete boot games
and **seven** completed restored training segments. All **3,044** archive
events had correct source-age bins and 128-action offsets; **315** originated
in already-restored segments. Its [ten complete games](results/defense/training/age-calibration-01/checkpoint/evaluation.json)
averaged **10,387**, median **10,460**, best **10,480**, all stage 1 without a
mission. The [replay](results/defense/training/age-calibration-01/replay/replay.html)
verified **2,573** actions. Full logs, optimizer and replay are preserved. This
retains strong play but remains below the parent's ten-game mean **10,478**;
it is not evidence of improvement and has no matched 32-worker resumed control.

`defense-ppo-23-life-age` now continues from that checked calibration's
optimizer at **11,881,216**, with the same 32-worker settings, **unlimited**
training and 100,000-action evaluations. The
[configuration](results/defense/training/ppo-23-life-age/resume-config.json)
records its lineage; emulator episodes restart and archives refill from new
own experience. This longer trial tests whether reward-free later states help
progression, not a claimed success of the calibration. Runs 17, 20 and DQN 22
continue. The sole collector includes the new full trial and all old sources,
but neither small check. The shared best remains unchanged.

Run 23's [first ten complete games](results/defense/training/ppo-23-life-age/step-000011987712/evaluation.json),
at **11,987,712** (**106,496** actions after calibration), averaged **9,257**,
median **10,280**, best **10,360**, all stage 1 without a mission. Its
[verified replay](results/defense/training/ppo-23-life-age/first-replay/replay.html)
reproduced **2,507** actions. The full model/optimizer is preserved. This first
regular batch is below the calibrated parent, not an improvement; the longer
trial continues without replacing the stronger shared best.

By **12,184,320**, run 23 matched the shared best's **10,480** single-game
score, with [ten-game mean 9,710, median 10,460](results/defense/training/ppo-23-life-age/step-000012184320/evaluation.json).
All games remained in stage 1, with no mission completion. Its full optimizer
and [2,581-action verified replay](results/defense/training/ppo-23-life-age/replay-10480/replay.html)
are preserved separately. This recovers the score plateau but is still below
the starting calibration's mean; an equal best score does not promote the
global replay, and does not establish new progression.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-age-calibration-reproduction \
  --resume results/defense/training/ppo-21-long-lookback/step-000011750144 \
  --artifacts runs/defense-age-calibration-reproduction/artifacts \
  --curriculum-cells age --curriculum-age-interval 32 \
  --eval-every 131072 --steps 11881216

# Continue the preserved calibration with unlimited learning:
venv/bin/python -u -m rl.defense_train --run runs/defense-life-age-reproduction \
  --resume results/defense/training/age-calibration-01/checkpoint \
  --artifacts runs/defense-life-age-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Remaining validation: stage 2/3 controls and mission-success detection are
supported by disassembly and parser tests, but **not yet exercised by an
unmodified complete playthrough reaching those stages**. Validate them when a
learned policy reaches them; do not patch memory or supply scripted expert play
to manufacture a success. The screen parser is single-player only. If a new
binary changes its HUD or awards extra ships, its assumptions must be audited
again. Keep models and diagnostics tied to the recorded executable hash.
