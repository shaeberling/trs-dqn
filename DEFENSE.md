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

- Live progress: `runs/defense-ppo-08-long-rollout/status.json` and
  `runs/defense-ppo-09-curriculum-life/status.json`, each with an adjacent
  `metrics.jsonl`. Runs 04, 05, 06 and 07 have stopped cleanly; their complete logs
  and final resumable checkpoints are archived below.
- Historical checkpoints: `runs/defense-ppo-*/step-*/`, including optimizer,
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
help; both active learners continue with their original settings.

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
it was later archived as described above. Run 09 remains active.

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
and verified replays remain available, and both active runs continue unchanged.
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
and complete replay bundle are preserved; both active experiments continue.

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
Run 09 continues unchanged as the reference experiment. Both write isolated
artifacts, with the single verification collector watching both sources.

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
The experiment continues unchanged beyond this initial validation.

Remaining validation: stage 2/3 controls and mission-success detection are
supported by disassembly and parser tests, but **not yet exercised by an
unmodified complete playthrough reaching those stages**. Validate them when a
learned policy reaches them; do not patch memory or supply scripted expert play
to manufacture a success. The screen parser is single-player only. If a new
binary changes its HUD or awards extra ships, its assumptions must be audited
again. Keep models and diagnostics tied to the recorded executable hash.
