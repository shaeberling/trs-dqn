# Obstacle Run / Missile Defense: integration and training feasibility

The user-confirmed new game is the executable in `Missile_Defense.zip`.
It identifies itself as **Obstacle Run**, by Arno Puder (1983/84), and is
already present as `var/defense.cmd`. No emulator rebuild, disk controller,
new ROM, binary patch or duplicate game asset is needed.

Status: **screen-only PPO training is running independently of Breakdown**, with
parallel emulator workers, resumable checkpoints, complete-game validation and
automatic verified best-effort replays. See the commands and monitoring paths
below. A successful mission has not yet been verified.
The current standard-policy best is **460 points**, with **1,675** neural
actions exactly reverified; its ten-game mean is **416**, median **420**, all stage 1.
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

- Live progress: `runs/defense-ppo-06-life-boundary/status.json` and
  `runs/defense-ppo-07-curriculum/status.json`, each with an adjacent
  `metrics.jsonl`. Runs 04 and 05 have stopped cleanly; their complete logs
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
are preserved. Run 06 continues; the freed slot now runs the own-experience
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
Run 06 remains active for further training with its isolated archive;
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
counter **3,435,264** (median 420, best 440), still all stage 1. It remains
unchanged while the separate curriculum experiment runs.

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

`defense-ppo-07-curriculum` now resumes run 05's preserved 460-point checkpoint
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
reload; the shared 460-point best remains unchanged. Boot-game and restored
segment counters remain separate in the live log.

The second validation mean rose to **420** and the third to **438** (median
**440**, best **460**) after **303,104 additional actions**, at counter
**3,537,664**. The [third checkpoint](results/defense/training/ppo-07-curriculum/step-000003537664/state.json)
preserves model, optimizer and the complete ten-game evaluation. This improves
consistency on reused validation seeds, not fresh-test performance: every game
remained in stage 1, with no mission completed. The shared best single-effort
replay stays with the original verified 460-point model because the new best
score only ties it. Both active experiments continue unchanged.

Remaining validation: stage 2/3 controls and mission-success detection are
supported by disassembly and parser tests, but **not yet exercised by an
unmodified complete playthrough reaching those stages**. Validate them when a
learned policy reaches them; do not patch memory or supply scripted expert play
to manufacture a success. The screen parser is single-player only. If a new
binary changes its HUD or awards extra ships, its assumptions must be audited
again. Keep models and diagnostics tied to the recorded executable hash.
