# Obstacle Run / Missile Defense: integration and training feasibility

The user-confirmed new game is the executable in `Missile_Defense.zip`.
It identifies itself as **Obstacle Run**, by Arno Puder (1983/84), and is
already present as `var/defense.cmd`. No emulator rebuild, disk controller,
new ROM, binary patch or duplicate game asset is needed.

Status: **Defense training has resumed after the user freed disk space**
(23 GiB available at restart). The full **276-test** suite now passes, including
the supervisor checks previously blocked by the unchanged 5 GiB safeguard.
Bootstrap DQN 24, PPO 29 and frozen-memory PPO 30 later retired after depth
plateaus or sustained regression, with all results preserved.
Training remains independent of Breakdown, with complete-game
validation and automatic verified best-effort replays. No history was deleted.
Full DQN trials 31/32 now compare persistent random exploration with the
unchanged control. Their first six full-run rounds disagree on mean-score
ranking; all games still lost in stage 1.
A bounded matched check of 5% versus 25% nominal persistent exploration
finished without a new stage; the higher rate scored worse. Neither full
run's settings changed.
Bounded follow-ups combine persistent exploration with their own newly
reached-state resets. Both have finished without a new stage; the 32-action
lookback improves the bounded comparison's mean and archive coverage, but
does not solve the shared failure pattern.
Full trials 33/34 now continue the verified shorter-lookback reset arm and
its matched high-exploration no-reset comparator, retaining complete-game
evaluation and automatic verified replay collection.
Their first full-run round favors resets by 62 mean points, down from the
bounded check's 224; all twenty games still lost in stage 1.
A successful mission has not yet been verified.
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

- Live unlimited progress: `runs/defense-dqn-31-persistent/status.json`,
  `runs/defense-dqn-32-persistent-control/status.json`,
  `runs/defense-dqn-33-persistent-resets/status.json` and
  `runs/defense-dqn-34-persistent-rate-control/status.json`, each with an adjacent
  `metrics.jsonl`. Earlier trials have stopped cleanly; their outcomes and
  archived resumable checkpoints are recorded below. Confirm a status file's
  PID is still alive before treating it as evidence of a running learner.
- Completed bounded persistent-exploration comparison:
  `runs/defense-persistent-control-01/status.json` and
  `runs/defense-persistent-memory-01/status.json`. These are training checks,
  excluded from the global collector. The latter name refers to holding a
  random action, **not** a recurrent neural network.
- Completed bounded exploration-rate comparison:
  `runs/defense-persistent-rate-control-01/status.json` and
  `runs/defense-persistent-rate-high-01/status.json`, with adjacent logs.
  These short checks are excluded from the global collector.
- Completed bounded persistent-exploration/own-reset follow-up:
  `runs/defense-persistent-reset-calibration-01/status.json` (lookback 128,
  finished) and `runs/defense-persistent-reset-short-calibration-01/status.json`
  (lookback 32, finished), with adjacent logs. Both sources are excluded from
  the global collector.
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

After the storage pause, both original learner processes were confirmed gone
and every pause-checkpoint file was checked against its archived copy before
restart. DQN resumed at **7,383,056** actions and PPO at **15,674,112**.
Learning settings remain unchanged; newer optional features retain their
disabled/default behavior. The preserved
[DQN restart configuration](results/defense/training/dqn-24-bootstrap/resume-after-storage-config.json)
and [PPO restart configuration](results/defense/training/ppo-29-value-weight/resume-after-storage-config.json)
record the exact source hashes and ancestry. The
[259-test result](results/defense/diagnostics/post-storage-regression-tests.txt)
passed after space was freed. Both processes were observed advancing counters;
the original sole collector remained live, with short experiments excluded.

Weights, optimizer, DQN targets/priors and saved RNG states are restored, but
games restart from boot, DQN refills its replay buffer from new own experience,
and PPO refills its own-state archive. This is not exact trajectory or replay-
buffer continuation. Initial training-game scores during random DQN warmup are
not frozen-policy validation results. The stronger shared best is preserved.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-24-bootstrap \
  --artifacts runs/defense-dqn-24-bootstrap/artifacts \
  --resume results/defense/training/dqn-24-bootstrap/pause-checkpoint-000007383056 \
  --steps 0
venv/bin/python -u -m rl.defense_train --run runs/defense-ppo-29-value-weight \
  --artifacts runs/defense-ppo-29-value-weight/artifacts \
  --resume results/defense/training/ppo-29-value-weight/pause-checkpoint-000015674112 \
  --steps 0
```

These are the recorded restart commands, not commands to launch duplicate
learners while those runs are already active. Use the latest preserved
checkpoint and an unoccupied run directory for a later continuation.

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
  --source runs/defense-ppo-25-matched-history/artifacts \
  --source runs/defense-dqn-26-own-resets/artifacts \
  --source runs/defense-ppo-27-matched-history-low-lr/artifacts \
  --source runs/defense-dqn-28-large-replay/artifacts \
  --source runs/defense-ppo-29-value-weight/artifacts \
  --source runs/defense-ppo-30-frozen-memory/artifacts \
  --source runs/defense-dqn-31-persistent/artifacts \
  --source runs/defense-dqn-32-persistent-control/artifacts \
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

The later `rl.defense_alias_probe` extends this to whole **verified own-policy
replays**, including recurrent memory carried across ship losses. It reconstructs
boot-padded screen stacks at the saved stride and must reproduce every original
sampled action using the original RNG before reporting statistics. Source hashes
are checked before and after. This is replay analysis, **not a new native game
evaluation**; the original native verification is retained in the report.
Float32 sampling is unchanged; entropy uses float64-renormalized probabilities
to remove small softmax rounding errors (maximum observed sum error below
0.000005 in these traces).

| Selected replay | Raw entropy (nats) | Grouped entropy (nats) | Alias share | Movement probability |
| --- | ---: | ---: | ---: | ---: |
| [Original best, 2,580 actions](results/defense/diagnostics/aliases-global-best-8342272.json) | 0.37958 | 0.25838 | 31.9% | 55.5% |
| [Frozen memory at 335,872, 2,494 actions](results/defense/diagnostics/aliases-frozen-memory-335872.json) | 0.61472 | 0.47975 | 22.0% | 57.5% |

Grouping merges forward-fire IDs 9–17 only; it describes the stage-one command
handler, not necessarily distinct physical outcomes. In the four 128-action
windows before **visible** ship-loss reports, alias entropy shares ranged from
15.3–25.9% for the original best and 22.0–25.4% for the recurrent replay.
The whole trajectories include animations, and visible loss can lag collision.
These two selected games have different seeds and trajectories: this is not a
matched-state causal comparison or a fresh success-rate estimate. Neither
trace shows redundant firing as most of its action entropy. This does not rule
out benefits from a different action set, but does not establish aliasing as
the dominant bottleneck here; controls and training remain unchanged. No
diagnostic frames/actions are used as training data or promotion candidates.

```bash
venv/bin/python -m rl.defense_alias_probe \
  results/defense/training/ppo-30-frozen-memory/first-replay \
  --output runs/defense-alias-reproduction.json
```

### Shared failure location: screen evidence

The user's observation that the policies fail at the same place prompted a
[four-policy, sixteen-life comparison](results/defense/diagnostics/shared-loss-01/report.json).
`rl.defense_loss_probe` reads only preserved, hash-checked, originally verified
own-policy traces. It reconciles each life total with recorded visible score
increments and renders the original screen bytes, without running new games,
reading private RAM, generating demonstrations, or changing training.

| Selected policy replay | Points earned on each of its four lives | Screen comparison |
| --- | --- | --- |
| Original global best | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-1-losses.png) |
| PPO 12 best effort | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-2-losses.png) |
| Frozen-memory PPO 30, 933,888 | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-3-losses.png) |
| Independently trained bootstrap DQN 24 | 2,550 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-4-losses.png) |

Visual inspection shows the same broad horizontal barrier approaching the
ship, with a large opening at the right, while these ships remain near the
center/left. Some frames show small breaks in the barrier; this is not proof
that one particular route or action is required. The selected actions differ
across policies and lives, but none of these traces negotiates this obstacle.
This supports a **shared behavioral bottleneck**, not just similar final
scores. It is still a selected-replay observation, not a population failure
rate, measured course index, or proof of the exact collision mechanism.

Rows of each sheet are lives; columns are 64, 32, 8 and 1 decisions before
an alignment marker. For the first three lives, the marker is the first sampled
frame in the last 128 with more than half the non-HUD cells solid white.
These flashes precede the visible life decrement by **11–18 decisions**.
They are **not exact physical collision timestamps**; even a preceding frame
can already be in the death animation. No such flash is sampled on the final
life (terminal settling advances the original animation), so that row explicitly
uses the visible loss endpoint instead. The sheets do not pretend these
different markers are precisely time-aligned collision events.

The records also show large **750–770** and **1,500–1,520** visible-score
increments shortly before this failure. A plausible learning explanation is
that the policies reliably collect the earlier reward but have not discovered
the longer sequence needed to continue past the barrier. This remains a
hypothesis, not a demonstrated cause. Movement choices still account for
34–72% of the selected 64-decision windows; action counts alone do not establish
effective displacement, and "the agent never moves" would be inaccurate.

The training implication is to judge subsequent experiments by escaping this
failure pattern and reaching new stages, not by another tiny increase in mean
score at the same 10,480 ceiling. Earlier lookback, longer-return, action-timing
and parameter-noise experiments are already documented below and did not clear
it; simply repeating them is not a new diagnosis. The live frozen-memory trial
continues as a controlled test, with no hand-coded right-turn rule, screen-derived
collision reward, or replay examples added to learning.

All **269 regression tests** pass. The new read-only checks cover synthetic
loss-window boundaries, absent-flash fallback, score reconciliation, invalid
traces, original-source immutability and checksum rejection, plus exact
feedforward/recurrent action reconstruction in the separate alias diagnostic.
The [full test log](results/defense/diagnostics/shared-loss-regression-tests.txt)
is preserved. No learner or environment behavior changed in this diagnostic.

```bash
venv/bin/python -m rl.defense_loss_probe \
  results/defense/learned/best \
  results/defense/training/ppo-12-lookback/best-effort \
  results/defense/training/ppo-30-frozen-memory/replay-peak-933888 \
  results/defense/training/dqn-24-bootstrap/replay-10410 \
  --output runs/defense-shared-loss-reproduction
```

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
The fresh-seed PPO lineage continued independently until the later plateau
described below.

Run 17 eventually stopped cleanly at **9,289,728** actions, with **3,969**
complete boot games, **2,649** restored segments and **92** complete ten-game
validations. There were **27** evaluations after its 6,504,448-action best;
none exceeded that checkpoint's mean or best score. The last ten means ranged
from **548 to 588**. Its final evaluation at **9,207,808** averaged **580**,
median **590**, best **600**. Neither training nor evaluation reached stage 2
or a mission. The [final optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/final-checkpoint/state.json)
and [losslessly compressed full log](results/defense/training/ppo-17-fresh-seed/metrics.jsonl.gz)
are preserved; decompression was byte-checked against the unchanged local log.
The earlier full 2,890-point checkpoint and verified replay remain available.
This score/depth plateau, not elapsed time, prompted releasing its compute
for the continuing timing and value-learning experiments.

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

### Frozen action-rate and visual-history comparison

The earlier shorter-action result needs an important qualification. A
**frozen-parent history-spacing probe** now separates action cadence from
the screen-history span. `rl.defense_temporal_probe` reuses the generic raw-frame
history wrapper with Defense's evaluator. It changes no weights, rewards or
categorical sampling rule, supplies no actions, and is restricted to the ten
reused primary validation seeds. Results are explicitly evaluation-only and
ineligible for promotion. Complete games are uncapped; HUD/terminal settling
adds variable time beyond the nominal history span.

All rows below use run 12's **10,447,616** checkpoint and seeds 10000–10009:

| Action interval / frame stride | Nominal four-frame span | Mean | Median | Best |
| --- | ---: | ---: | ---: | ---: |
| [100,000 / 1, original validation](results/defense/training/ppo-12-lookback/step-000010447616/evaluation.json) | 300,000 | 10,474 | 10,480 | 10,480 |
| [50,000 / 1, frozen control](results/defense/validation/step-10447616-tstates-50000-stride-1.json) | 150,000 | 7,939 | 7,760 | 10,310 |
| [50,000 / 2, frozen probe](results/defense/validation/step-10447616-tstates-50000-stride-2.json) | 300,000 | 10,433 | 10,430 | 10,480 |

Intervals/spans are CPU T-states. All 30 games ended in stage 1 without a
mission. The repeated stride-1 probe reproduced **all ten complete game
records exactly** from the earlier 50,000-T-state evaluation, not merely its
summary. With stride 2, all ten paired scores increased; mean difference was
**+2,494**. The checkpoint hash remained unchanged throughout both probes.
This recovers most of the frozen model's original score consistency at the
faster action rate, but does not demonstrate deeper progression, reliable
winning, or learning at that rate. It also does not convert the earlier small
short-action training regression into a success.

This motivates a subsequent **training** experiment that preserves visual
history and accounts for the changed action duration in its credit/reset
horizons. Such training is not implemented by this probe. The production
learners and shared best remain unchanged. **227 regression tests** passed,
including real Defense frame-spacing/sampling tests, input-seed restrictions,
the no-cap/no-promotion contract and existing generic temporal-history tests.

```bash
venv/bin/python -m rl.defense_temporal_probe \
  results/defense/training/ppo-12-lookback/step-000010447616/model.safetensors \
  --tstates 50000 --stride 2 --envs 4 --output results/defense/validation/new-history-probe.json
# For the narrow-history control, use --stride 1 and a distinct output path.
```

### Trainable observation spacing

The frozen Defense timing probe now uses the environment's native spacing
instead of the wrapper, allowing **greedy DQN** as well as categorical PPO
without changing either policy's action rule. It records original and probe
spacing separately, retains the reused-seed restriction, and cannot promote
results. All **237 regression tests** passed. A
[PPO parity check](results/defense/validation/native-probe-ppo-parity.json)
reproduced all ten earlier wrapper game records exactly.

For ordinary DQN's frozen **1,700,000** checkpoint, the same ten seeds gave:

| Action interval / frame stride | Mean | Median | Best |
| --- | ---: | ---: | ---: |
| [100,000 / 1, baseline](results/defense/validation/dqn-1700000-timing-baseline.json) | 1,372 | 1,470 | 1,670 |
| [50,000 / 1](results/defense/validation/dqn-1700000-tstates-50000-stride-1.json) | 601 | 560 | 1,370 |
| [50,000 / 2](results/defense/validation/dqn-1700000-tstates-50000-stride-2.json) | 924 | 540 | 2,220 |

The baseline reproduced every saved game record exactly. All thirty games
ended in stage 1 without a mission, and weights remained unchanged. Wider
history partially recovers the mean at the faster cadence, but is still below
the original timing and has a lower median than either control. Its higher
single score is diagnostic only, not a new trained or promoted best. This
mixed result does not currently justify a separate faster-action DQN run;
ordinary DQN continues at its original timing while PPO tests learning at the
faster cadence.

Both Defense trainers now accept `--observation-stride` (positive integer,
default **1**). The policy still receives exactly four visible frames. Stride
2 retains seven consecutive frames and selects indices 0, 2, 4, 6; it adds
neither hidden-state input nor an action controller. Snapshot capture retains
all intermediate frames so resumed stacks are exact. Cross-stride or malformed
snapshots are rejected before changing emulator state, including shared score,
screen and age archives. Checkpoint evaluation, recording and replay verification
use the saved stride; legacy checkpoints default to 1.

The existing global best was reloaded under this implementation and reproduced
all **2,580** actions, screens and rewards at its original stride, with the
same **10,480**-point stage-1 result. Real-emulator tests compare stride 2
against the frozen probe wrapper, exercise snapshot continuation and peer
sharing, and check vector workers and both trainers' optimizer resumes.
All **236 regression tests** passed, including the storage checks below.
This is infrastructure for a training experiment, not evidence of a new stage.

Immutable Defense checkpoint/replay copies can be inventoried with
`python -m rl.defense_storage`; `--apply` replaces byte-identical duplicates
with macOS APFS copy-on-write clones. Every path and content hash is retained,
with independent future writes (not hard links). Live `latest`/`best` paths
and logs are excluded. The first pass replaced **393** duplicate files across
**278** groups, with about **1.6 GiB** more free space measured afterward;
concurrent training also changes free space. No historical checkpoint was
deleted, and the existing disk-space safety threshold was not lowered.

### Matched-history learning check and longer trial

The [isolated four-worker check](results/defense/training/matched-history-smoke-01/resume-config.json)
resumed the original run-12 optimizer at **10,447,616**, not a timing-probe
trajectory. It trained **32,768** new actions at 50,000 T-states with stride 2,
ending at **10,480,384**. Relative to the earlier 16,384-action, 100,000-T-state
four-worker control, rollout and batch doubled to **512**, lookback doubled
to **64**, gamma became **sqrt(0.997)** and GAE lambda **sqrt(0.99)**. These
preserve nominal visual, rollout, reset and discount horizons and the number
of minibatches per rollout. Variable HUD settling, new decision opportunities,
RNG use and changed optimization data prevent an exact game-time equivalence.

Four complete boot games and two restored segments finished during training.
All **326** archive events had exact 64-action source/trigger offsets; **60**
originated in restored segments. The check's
[ten complete evaluations](results/defense/training/matched-history-smoke-01/checkpoint/evaluation.json)
averaged **9,685**, median **10,415**, best **10,480**, all stage-1 losses.
This is below the frozen same-timing parent's mean **10,433** and the old
100,000-T-state short control's **9,929**, though above the earlier narrow-history
short-action check's **5,261**. It is not a controlled single-setting ablation
or a demonstrated improvement. Its
[5,089-action verified replay](results/defense/training/matched-history-smoke-01/replay/replay.html),
full optimizer and log are preserved; it exited normally and is excluded
from the collector.

`defense-ppo-25-matched-history` now tests this timing/history combination at
**32 workers**, resuming the original **10,447,616** parent directly, **not**
the short-check checkpoint. It uses rollout 512, batch 512, eight boot-only
workers, score-bin sharing at probability 0.5, lookback 64 and the adjusted
discounts above. Training and games have no action or wall-clock cap; complete
ten-game evaluations occur every approximately 200,000 actions. Batch size
stays 512 for memory headroom, so the larger rollout has more minibatches
than the original normal-size trial. The
[saved configuration](results/defense/training/ppo-25-matched-history/resume-config.json)
records all settings and source hashes. It replaces stopped run 23's compute
slot. The sole collector was restarted with stride-aware verification and this
full trial added; all old sources remain, and small probes remain excluded.
No stage-2 or mission success is claimed from launching it.

Its first [ten complete evaluations](results/defense/training/ppo-25-matched-history/step-000010660608/evaluation.json),
at **10,660,608** (**212,992** new actions), averaged **9,411**, median **9,590**,
best **10,440**. All ended in stage 1 without a mission. Its full optimizer
checkpoint and [5,061-action verified replay](results/defense/training/ppo-25-matched-history/first-replay/replay.html)
are preserved. This first batch is below the frozen timing-matched parent and
the short learning check; the longer trial continues, without replacing the
stronger shared best.

Run 25's second batch at **10,857,216** raised its own best to **10,460**,
but its mean fell to **8,750** (median **8,155**). The full
[checkpoint](results/defense/training/ppo-25-matched-history/step-000010857216/evaluation.json)
and [5,116-action verified replay](results/defense/training/ppo-25-matched-history/replay-10460/replay.html)
are preserved. The five successive validation means were **9,411 → 8,750 →
7,910 → 2,562 → 326**, with final median **300**, best **440** at **11,463,424**.
No complete game reached stage 2 or a mission.

The trial stopped cleanly at **11,512,576**, after **1,064,960** new actions,
**193** complete boot games and **91** restored segments. Its
[final optimizer checkpoint](results/defense/training/ppo-25-matched-history/final-checkpoint/state.json)
and [full log](results/defense/training/ppo-25-matched-history/metrics.jsonl)
are preserved alongside its first/strongest-mean and best-effort checkpoints.
This sustained loss of competence motivated stopping the configuration, not
a wall-clock limit. A separate half-learning-rate check starts from the
original strong parent, never the collapsed model; a learning-rate explanation
is a hypothesis, not a diagnosed cause of this regression.

The [half-learning-rate short check](results/defense/training/matched-history-low-lr-smoke-01/resume-config.json)
differs from the earlier four-worker matched-history check **only** in learning
rate (**0.00025 → 0.000125**) and output paths. Both start from the original
10,447,616 parent and train 32,768 new actions to **10,480,384**. The new
[ten-game result](results/defense/training/matched-history-low-lr-smoke-01/checkpoint/evaluation.json)
was mean **10,152**, median **10,450**, best **10,480**, versus the original
check's mean **9,685**, median **10,415**, best **10,480**. All games remained
stage-1 losses. This short comparison improves retention but remains below
the frozen timing-matched parent's mean **10,433**; it does not establish
deeper progress or explain the longer trial's failure.

The new check completed four boot games; no restored segment had finished
at its checkpoint. It recorded **329** archive events, all with exact
64-action lookback, **30** from ongoing restored segments. Its
[4,987-action verified replay](results/defense/training/matched-history-low-lr-smoke-01/replay/replay.html),
full optimizer and log are preserved, and it exited normally. It is excluded
from the collector and is not used as the parent of the longer trial.

`defense-ppo-27-matched-history-low-lr` now runs at the normal 32-worker size,
starting directly from the same original **10,447,616** parent. Its
[configuration](results/defense/training/ppo-27-matched-history-low-lr/resume-config.json)
differs from run 25's **only** in learning rate (0.000125) and output paths.
It retains unlimited learning/games, 50,000-T-state actions, stride 2,
rollout/batch 512, adjusted gamma/lambda, eight boot-only workers and lookback
64. It replaces stopped run 25's compute slot; DQN runs 22, 24 and 26 continue.
The larger rollout had increased minibatches per nominal game-time window;
halving the rate tests a smaller update size, not an exact optimizer-budget
equivalence because Adam and KL-based epoch stopping are nonlinear. The sole
collector includes run 27, with all older sources retained and small checks
excluded. No longer-run improvement is claimed from the short result alone.

Run 27's [first ten complete evaluations](results/defense/training/ppo-27-matched-history-low-lr/step-000010660608/evaluation.json),
at **10,660,608** (**212,992** new actions), averaged **9,933**, median
**10,385**, best **10,480**, all stage-1 losses. At the same first checkpoint,
run 25 averaged **9,411**, median **9,590**, best **10,440**. The lower-rate
trial therefore retains more score in this first batch, but neither shows
new depth, and long-run stability remains unproven. Its full optimizer and
[5,044-action verified replay](results/defense/training/ppo-27-matched-history-low-lr/first-replay/replay.html)
are preserved separately; the equal best score does not replace the shared
global replay.

At **11,250,432** (**802,816** new actions), its fourth batch averaged
[10,237, median 10,370, best 10,480](results/defense/training/ppo-27-matched-history-low-lr/step-000011250432/evaluation.json).
The full optimizer checkpoint is preserved. This compares with run 25's
same-counter mean **2,562**; the lower-rate variant has retained substantially
more score through these four batches. All ten games still lost in stage 1,
so this is retention, not new depth or proof of lasting stability.

The fifth batch, at **11,463,424**, fell to
[mean 6,929, median 7,900, best 10,460](results/defense/training/ppo-27-matched-history-low-lr/step-000011463424/evaluation.json).
That full checkpoint is preserved as well. Lowering the rate delayed the
earlier run's regression but has not eliminated it; none of these games
reached stage 2. One declining batch is not yet grounds to claim a permanent
plateau or to retire this trial.

The longer run did not recover its peak. It stopped cleanly at
**24,488,704** inherited actions: **14,041,088** new actions, **2,267** new
boot games, **1,461** restored segments and **70** completed ten-game
validations. Its strongest mean remained **10,237** at 11,250,432; none
reached stage 2 or a mission. The last complete batch at **24,455,936**
averaged **6,559**, median **6,335**, best **10,080**. The
[final checkpoint](results/defense/training/ppo-27-matched-history-low-lr/final-checkpoint/state.json),
[last validated checkpoint](results/defense/training/ppo-27-matched-history-low-lr/step-000024455936/evaluation.json)
and [losslessly compressed full log](results/defense/training/ppo-27-matched-history-low-lr/metrics.jsonl.gz)
are preserved alongside its earlier peak and verified best replay. This
extended depth plateau motivated replacing the run, not a wall-clock limit.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-matched-history-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-matched-history-reproduction/artifacts \
  --envs 32 --rollout 512 --batch-size 512 --tstates 50000 --observation-stride 2 \
  --gamma 0.9984988733093293 --gae-lambda 0.99498743710662 \
  --curriculum-lookback 64 --curriculum-boot-envs 8 --mlx-cache-mb 512 \
  --eval-envs 8 --eval-every 200000 --steps 0
# For run 27's lower-rate variant, use a distinct run/artifact directory and
# add --learning-rate 0.000125, keeping the original parent above.
```

### PPO value-loss weight comparison

`rl.defense_train --value-coefficient` controls the weight on **half the mean
squared value error** in PPO's existing joint loss. The default remains **0.5**,
so the complete value term is still 0.25 times mean squared error. A value of
0.1 makes that term 0.05 times mean squared error. The actor objective, entropy
coefficient, advantage/return calculation, reward, observations, action set
and evaluation policy are unchanged. This is not a reward bonus or a second
source of supervision. SIL's separate loss, when enabled, is unaffected.

The option is saved and inherited on resume, can be explicitly overridden,
and defaults to the old weight for older checkpoints. Current live learners
are not silently reconfigured. Source hashes now identify the Defense trainer,
PPO loss implementation and unchanged network in new configuration records.

This tests whether reducing the critic's contribution to the shared encoder
helps retain useful policy behavior. Rising value error around run 27's
regression motivates the comparison but does not establish causality; error
can also be a consequence of changed behavior. Adam's existing moments and
joint gradient clipping mean reducing this coefficient is not equivalent to
scaling a separate critic learning rate.

All **252 regression tests** passed. Targeted checks verify exact default loss
and every gradient against the previous formula (including reference-KL mode),
the isolated coefficient's effect on value/shared gradients without changing
actor-head gradients, invalid arguments, real-emulator default parity and
resume inheritance/override/legacy fallback. Paired short training checks use
the same original parent and settings, changing only the coefficient; their
sources are excluded from the shared best collector.

The completed matched pair used four workers, 512-step rollouts, batch 512,
50,000-T-state actions, stride 2, learning rate 0.000125, the adjusted gamma/
lambda and 64-action lookback of the lower-rate timing check above. Both
started directly from original parent **10,447,616** and collected **32,768**
new actions, ending at **10,480,384**. Configuration differences are only the
coefficient and output paths; no evaluation trajectory is training data.

| Value coefficient | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0.5 control](results/defense/training/value-control-01/checkpoint/evaluation.json) | 10,152 | 10,450 | 10,480 | [4,987](results/defense/training/value-control-01/replay/replay.html) |
| [0.1 reduced](results/defense/training/value-small-01/checkpoint/evaluation.json) | 8,679 | 7,940 | 10,460 | [5,084](results/defense/training/value-small-01/replay/replay.html) |

All twenty validation games lost in stage 1. The reduced weight scored lower
on nine of the ten paired seeds, higher on one, with a mean difference of
**−1,473**. It is not being promoted to a longer run. This negative short
comparison does not prove why PPO regressed, or rule out every other critic
configuration. The control additionally reproduced the earlier lower-rate
check's model bytes, all **26 optimizer arrays**, counters/RNG and all ten
game records **exactly**; see its [parity record](results/defense/training/value-control-01/parity.json).

Both checks exited normally; full models, optimizer states, logs, settings
and verified replays are preserved. Each completed four new boot games during
training; the control completed no restored segments and the reduced variant
completed two. Those restored segments are not counted as complete games.
The opposite-direction **1.0** coefficient check used the same parent and
settings, recorded [here](results/defense/training/value-large-01/resume-config.json).
Its [ten complete games](results/defense/training/value-large-01/checkpoint/evaluation.json)
averaged **10,449**, median **10,450**, best **10,480**. That is **297** more
than the matched control's mean, still entirely stage-1 losses. It completed
four new boot games and no restored segments during training, exited normally,
and preserved its full model/optimizer/log plus a
[5,104-action verified replay](results/defense/training/value-large-01/replay/replay.html).
This short result supports a longer comparison, not a claim of new depth,
durable stability or a diagnosed cause of the earlier regression.

`defense-ppo-29-value-weight` now runs at the normal **32-worker** size,
starting directly from original parent **10,447,616**, not from the short
check. Its [configuration](results/defense/training/ppo-29-value-weight/resume-config.json)
matches run 27's learning settings except for **value coefficient 1.0**
instead of the old implicit 0.5: learning rate 0.000125, 512-step rollouts,
batch 512, 50,000-T-state actions, stride 2, adjusted gamma/lambda, shared
score archive, eight boot-only workers and 64-action lookback. Training and
evaluation are uncapped; ten-game evaluation occurs every 200,000 actions.
It replaces stopped run 27's slot. The sole collector includes this source
and all historical sources; short checks remain excluded.

To reproduce, use run 27's command above with distinct output paths and
`--learning-rate 0.000125 --value-coefficient 1`.

Run 29's [first ten complete games at 10,660,608](results/defense/training/ppo-29-value-weight/step-000010660608/evaluation.json)
(**212,992** new actions) averaged **10,210**, median **10,460**, best
**10,480**. The full optimizer checkpoint and
[5,044-action verified replay](results/defense/training/ppo-29-value-weight/first-replay/replay.html)
are preserved. Its mean exceeds run 27's same-counter 9,933 by 277, but all
games still lost in stage 1. This is early retention, not proof of durable
stability or a depth gain, and the tied best does not replace the global replay.

Run 29 later reached [mean **10,461**, median **10,460**, best **10,480** at
11,250,432](results/defense/training/ppo-29-value-weight/step-000011250432/evaluation.json).
Its full optimizer checkpoint is preserved. Subsequent means fluctuated and
fell to **8,105** at **15,248,128**; all remained stage-1 losses. It subsequently
paused cleanly for disk pressure at **15,674,112** inherited actions
(**5,226,496** new actions). All **26** complete validation batches stayed in
stage 1; the last mean was **7,879** at **15,657,728**. The
[pause checkpoint](results/defense/training/ppo-29-value-weight/pause-checkpoint-000015674112/state.json),
[validation history](results/defense/training/ppo-29-value-weight/validation-at-000015674112.json)
and [losslessly compressed complete log](results/defense/training/ppo-29-value-weight/metrics-at-000015674112.jsonl.gz)
are preserved. This is not evidence that more compute has solved progression.

After storage recovery, run 29's
[first resumed validation at **15,887,104**](results/defense/training/ppo-29-value-weight/step-000015887104/evaluation.json)
averaged **8,966**, median **9,935**, best **10,440**, all ten stage-1 losses.
That is above its last pre-pause mean of 7,879, but below its preserved peak
10,461 and the shared best's score. The full optimizer checkpoint is preserved;
the continuation did not replace the global replay.

Run 29 subsequently retired cleanly at **18,279,168** inherited actions,
**7,831,552** new actions beyond its original parent. All **38** complete
ten-game validations stayed in stage 1; peak mean remained 10,461. The final
validation at **18,082,560** averaged **8,416**, median **8,595**, best **10,220**.
Its [last validation checkpoint](results/defense/training/ppo-29-value-weight/step-000018082560/evaluation.json),
[final optimizer checkpoint](results/defense/training/ppo-29-value-weight/final-checkpoint-000018279168/state.json),
[retirement record and curve](results/defense/training/ppo-29-value-weight/retirement-000018279168.json)
and [complete compressed log](results/defense/training/ppo-29-value-weight/metrics-at-000018279168.jsonl.gz)
are preserved. The stopped process was confirmed gone. Sustained regression
and no new stage—not a wall-clock limit—motivated releasing its compute for
the controlled persistent-exploration comparison below. Its earlier best replay
and every previous checkpoint remain available.

### Recurrent screen-history experiment

`rl.defense_train --recurrent-hidden 128 --sequence-length 32` adds a GRU to
the existing CNN's 256-feature output, with residual actor/value projections.
The motivating question is whether learned memory of earlier visible screens
helps beyond the four-frame input. Memory has been studied in
[Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527); this implementation
is **residual GRU PPO**, not a reproduction of that paper's LSTM DQN. We have
not established that insufficient history causes the current stage-1 plateau.

Only the policy's own screen history enters the recurrent state. There are
no game-private inputs, position labels, routes, oracle actions or extra
rewards. Memory starts at zero on boot or a real environment reset (including
a reset to an opaque own-reached state), not at a visible ship loss. Evaluation
always starts from boot; parallel games have independent memory keyed by game
identity, and evaluation never advances the learner's carried memory.

Training shuffles contiguous within-worker sequences, not individual frames.
The sequence length controls truncated backpropagation, while inference memory
can persist for the full game. Initial sequence states are detached from the
gradient; no burn-in is implemented. Carried states can therefore be stale
after an optimizer update. Resuming restores weights, optimizer and action RNG,
but emulator episodes and neural memory restart from boot. Own-state archives
do not contain saved neural memory. These are experiment limitations, not
claims of exact trajectory continuation.

`--initialize-policy` copies the entire compatible **own learned** feedforward
PPO into the residual base. The memory output projections start at zero,
preserving the parent's initial logits and values; the optimizer and action
counters start fresh. The base remains trainable. Saved provenance includes
the parent's model/configuration hashes and its **10,447,616** pretraining
actions: this is transfer from our own policy, not learning from scratch or
from demonstrations. `--memory-scale 0` supplies a matched memory-disabled
control with the same initialization and sequence batching. Default
`--recurrent-hidden 0` preserves the original feedforward path.

The [zero-update initializer](results/defense/training/recurrent-initial-01/checkpoint/state.json)
reproduced **all ten parent game records exactly** on reused validation seeds
10000–10009: mean **10,474**, median/best **10,480**, all stage-1 losses.
Its [replay](results/defense/training/recurrent-initial-01/replay/replay.html)
reverified **2,553** neural actions. This proves initial-policy preservation,
not an improvement. The initializer's checkpoint, fresh optimizer, provenance,
[parity record](results/defense/training/recurrent-initial-01/parity.json) and
complete-game evaluation are preserved separately from the shared best.

Regression tests cover sequence ordering, zero-residual identity, within-sequence
resets, per-game parallel memory, memory learning, disabled-memory gradients,
bootstrap/evaluation isolation, native full-game serial/parallel agreement,
saved-policy replay verification, invalid modes and real training/resume.
SIL and actor-noise modes are rejected with recurrent training until separately
implemented and tested.

The two **32,768-action** checks completed normally, using the same initializer,
four workers, 256-step rollouts, batch 512, 32-step sequences, learning rate
0.000125, entropy 0.002, value coefficient 0.5, 100,000-T-state actions and
stride 1. Shared own-score resets used probability 0.5, one boot-only worker
and lookback 32. Configuration differs only in memory scale and output paths.

| Memory | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [Enabled](results/defense/training/recurrent-memory-01/checkpoint/evaluation.json) | 9,499 | 10,300 | 10,440 | 2,462 |
| [Disabled control](results/defense/training/recurrent-control-01/checkpoint/evaluation.json) | 10,169 | 10,395 | 10,480 | 2,489 |

Both evaluated ten complete boot games on reused seeds 10000–10009; every
game lost in stage 1. Memory was **670 points worse in mean** than its matched
control and both regressed from the untouched parent's 10,474. This short
comparison does not establish a benefit from recurrence. The enabled check
completed **12** boot games and **one** restored segment during learning;
the control completed **10** boot games and **four** restored segments.
All checkpoints, optimizers, configurations, full logs and separately verified
replays are preserved. Neither short check enters the shared collector.

The initial full suite passed **258 tests**. After strengthening memory tests,
the **259-test** suite passed all seven recurrent tests and the other
non-supervisor checks, but **three supervisor checks failed** when actual free
disk space fell below **5 GiB**. The safety threshold and tests were not
weakened. No unlimited recurrent run was launched during that pause. After the
user freed space, all **259 tests passed**; the log is linked in the restart
section above.

A larger matched comparison then used the normal **32 workers** and
**131,072** new actions per arm, starting from the original zero-update
initializer, not either four-worker short-check result. Both used eight
boot-only workers, the same 256-step rollouts and 32-step sequences, batch 512,
and unchanged learning/timing settings. Each evaluated ten complete games on
the same reused validation seeds; all were stage-1 losses.

| Memory, 32-worker check | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [Enabled](results/defense/training/recurrent-memory-calibration-01/checkpoint/evaluation.json) | 10,032 | 10,430 | 10,480 | 2,521 |
| [Disabled control](results/defense/training/recurrent-control-calibration-01/checkpoint/evaluation.json) | 10,300 | 10,445 | 10,480 | 2,560 |

The enabled run completed **32** boot games and **12** restored training
segments; the control completed **32** boot games and **11** restored segments.
Both exited normally. Their configurations differ only in memory scale and
output paths, as recorded in the
[paired comparison](results/defense/training/recurrent-memory-calibration-01/comparison.json).
Memory was **268 points lower in mean**, and both remained below the
untouched parent's 10,474. This does not establish a benefit from this recurrent
setup at either tested size; it does not rule out all recurrent methods.
Neither check was promoted to an unlimited run or added to the collector.
Full model/optimizer checkpoints, configurations, logs and verified replays
are preserved, with the parent's 10,447,616 pretraining actions recorded in
both lineages. These are reused-seed tuning results, not fresh success rates.

To reproduce this larger pair, use the two bounded commands below with
`--envs 32 --curriculum-boot-envs 8 --steps 131072 --eval-every 131072
--eval-envs 8 --mlx-cache-mb 512` and new run/artifact paths. The two checks
still start from the same saved initializer; only the control sets memory
scale to zero.

```bash
# Save an exact-policy recurrent initializer without learning:
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-initial-reproduction \
  --artifacts runs/defense-recurrent-initial-reproduction/artifacts \
  --initialize-policy results/defense/training/ppo-12-lookback/step-000010447616 \
  --initialize-only --recurrent-hidden 128 --sequence-length 32 \
  --envs 4 --rollout 256 --batch-size 512 --learning-rate .000125 \
  --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 1 \
  --curriculum-lookback 32 --eval-envs 4 --eval-every 32768 --mlx-cache-mb 256

# Matched bounded checks; initialize-only is deliberately not inherited:
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-memory-reproduction \
  --artifacts runs/defense-recurrent-memory-reproduction/artifacts \
  --resume results/defense/training/recurrent-initial-01/checkpoint --steps 32768
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-control-reproduction \
  --artifacts runs/defense-recurrent-control-reproduction/artifacts \
  --resume results/defense/training/recurrent-initial-01/checkpoint \
  --memory-scale 0 --steps 32768
```

#### Frozen own-policy residual experiment

`--freeze-recurrent-base` freezes the own initialized CNN, feature layer and
original actor/value heads **before Adam is constructed**. Only the GRU and
its actor/value residual heads receive gradients and optimizer slots. Complete
checkpoints still save every frozen weight, so evaluation/replay needs no
external base model. The default remains fully trainable; fresh frozen mode
requires compatible `--initialize-policy` and enabled memory. Changing the
frozen set during optimizer resume is rejected rather than silently mixing
incompatible optimizer states.

This tests whether a fixed base can retain a useful starting representation
while a learned residual adapts it. A related decomposition appears in
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
That work combines conventional continuous control with a learned correction;
ours is **not a reproduction**: it adds categorical-logit/value residuals over
our own RL-trained neural policy. No conventional controller, action oracle,
demonstration or hidden game information enters this experiment. Freezing
guarantees unchanged base parameters, **not unchanged overall behavior or a
performance improvement**; the learned residual can still make play worse.

The [frozen initializer](results/defense/training/recurrent-frozen-initial-01/checkpoint/state.json)
has **byte-identical model weights and identical policy RNG** to the original
recurrent initializer. All **18** retained optimizer arrays exactly equal
their original zero-step values; the original **42-array** optimizer's base
slots are absent. The [parity record](results/defense/training/recurrent-frozen-initial-01/parity.json)
checks the saved inference settings too. This reuses the prior verified
initializer's identity; it is not a newly measured ten-game evaluation.
Provenance retains the own feedforward parent's **10,447,616** training actions.

All [**262 regression tests passed**](results/defense/diagnostics/frozen-recurrent-regression-tests.txt).
New checks establish exact base immutability while memory weights learn,
absence of base gradients/optimizer slots, complete checkpoint serialization,
native saved-policy replay reproduction, inherited freezing on real training
resume, incompatible-mode rejection and unchanged default initialization.

`defense-recurrent-frozen-calibration-01` completed **131,072** new actions with
32 workers, eight boot-only workers and the same learning settings as the
previous memory-enabled calibration. It starts from the exact-weight frozen
initializer, not the already-trained short result. Its
[ten complete evaluations](results/defense/training/recurrent-frozen-calibration-01/checkpoint/evaluation.json)
averaged **10,453**, median **10,460**, best **10,480**, all stage-1 losses.
That is **421** above the matched unfrozen-memory mean 10,032, and **153** above
the memory-disabled control's 10,300, but still **21 below** the untouched
parent's 10,474. This is better short-run retention, not new stage reach or
proof that recurrence improves on the original parent.

It completed **32** boot games and **10** restored segments during training,
then exited normally. The full checkpoint, optimizer, configuration, log and
[2,551-action verified replay](results/defense/training/recurrent-frozen-calibration-01/replay/replay.html)
are preserved. The [post-training comparison](results/defense/training/recurrent-frozen-calibration-01/comparison.json)
checks all **12** base arrays against the original feedforward parent:
**exactly unchanged**, while all **eight** memory/residual arrays changed.
The optimizer still has **18** arrays and no base slots. Both recurrent arms'
learning and environment settings match; freezing changes the trainable
parameter set, and all seeds are reused validation seeds, not fresh tests.
The bounded check remains excluded from the collector.

The improved retention supports an exploratory longer trial,
`defense-ppo-30-frozen-memory`, continuing this calibration's full optimizer
at **131,072** (plus the inherited base's **10,447,616** pretraining actions).
The [configuration](results/defense/training/ppo-30-frozen-memory/resume-config.json)
keeps 32 workers, 256-step rollouts, batch 512, 128 memory units, 32-step
sequences, learning rate 0.000125, entropy 0.002, value coefficient 0.5,
gamma 0.997, lambda 0.99, 100,000-T-state actions, stride 1, life boundaries,
shared own-score resets, eight boot-only workers and lookback 32. Training
and games are uncapped; ten-game validation runs every 200,000 actions.
Episodes, neural memory and own-state archives restart, not the optimizer.
Runs 24 and 29 continue independently. The sole collector was cleanly
restarted with the recurrent loader and this new full-run source, retaining
all historical sources and excluding all short checks. It still requires
frozen-policy replay verification before replacing the shared best.

Run 30's [first full-run validation at **335,872**](results/defense/training/ppo-30-frozen-memory/step-000000335872/evaluation.json)
(**204,800** actions beyond calibration) averaged **10,464**, median/best
**10,480**, all ten stage-1 losses. Its
[2,494-action verified replay](results/defense/training/ppo-30-frozen-memory/first-replay/replay.html)
and full optimizer checkpoint are preserved. An additional
[saved-weight check](results/defense/training/ppo-30-frozen-memory/step-000000335872/base-immutability.json)
confirms all **12** base arrays remain exactly equal to the original own
feedforward parent and the optimizer still excludes their slots. Since the
calibration, training completed **55** additional boot games and **17** restored
segments. This is continued retention, slightly above calibration's mean
10,453 but below the original parent's 10,474; it is not a stage clear. The
tied best score does not replace the global replay. Unlimited learning continues.

The [next four validation rounds](results/defense/training/ppo-30-frozen-memory/validation-through-000001138688.json)
had means **10,464**, **10,382**, **10,476** and **10,464**. The then-peak at
**933,888** is preserved as a
[full optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000000933888/evaluation.json).
Independent frozen re-evaluation exactly reproduced **all ten game records**,
and its [2,551-action replay](results/defense/training/ppo-30-frozen-memory/replay-peak-933888/replay.html)
reproduced every neural action, reward and screen from boot. All twelve base
arrays remain exactly equal to the original parent; no base optimizer slots
were introduced. Mean 10,476 is only **two points** above that parent's 10,474,
and below run 21's earlier 10,478. All games still end in stage 1, with no
mission. This separately preserved evaluation-only replay does not replace
the tied global best or supply training data.

Reviewing the [first eighteen complete validation rounds](results/defense/training/ppo-30-frozen-memory/validation-through-000003735552.json)
identified a later mean-score peak at **1,335,296**: **all ten games scored
10,480**, all ending in stage 1 without a mission. That previously unarchived
[full optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000001335296/evaluation.json)
is now preserved. Independent frozen re-evaluation reproduced **every one of
the ten complete game records exactly**, and the separate
[2,577-action replay](results/defense/training/ppo-30-frozen-memory/replay-peak-1335296/replay.html)
reproduced every neural action, screen and reward from boot. All twelve base
arrays still exactly match the original own feedforward parent, with no base
optimizer slots. The mean is six points above that parent's 10,474, not a
new single-game ceiling or stage clear. These are repeatedly used validation
seeds, not a fresh success-rate test. Later means varied again (10,328 at
3,735,552); all eighteen rounds stayed in stage 1. The stable shared replay
remains unchanged on a tied best; this selected peak is separately available
and excluded from training data and automatic evaluation-probe promotion.

PPO 30 subsequently stopped cleanly at **6,340,608** after **31** complete
ten-game validation rounds, all stage-1 losses, and no later-stage event in
training. It had not exceeded the score/depth ceiling first achieved at
1,335,296 after another **5,005,312** actions. Its last validation at
**6,332,416** again scored 10,480 in all ten games; this is consistent replay
of the same ceiling, not stage progression. Retirement releases compute for
the matched persistent-exploration/own-reset experiment, not because of a
wall-clock limit. All earlier peaks and verified replays remain intact.

The [full final checkpoint](results/defense/training/ppo-30-frozen-memory/final-checkpoint-000006340608/state.json),
[last complete evaluation and optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000006332416/evaluation.json),
[31-round retirement record](results/defense/training/ppo-30-frozen-memory/retirement-000006340608.json)
and [complete compressed log](results/defense/training/ppo-30-frozen-memory/metrics-at-000006340608.jsonl.gz)
are preserved. Final cumulative counts are **1,884** boot games and **1,291**
restored segments. The final post-update checkpoint has not itself been
evaluated and is not substituted for the independently verified selected peak.
A [read-only final parameter check](results/defense/training/ppo-30-frozen-memory/final-checkpoint-000006340608/base-immutability.json)
confirms all twelve base arrays remain byte-identical to the original own
feedforward parent, with only eighteen non-base optimizer arrays.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-recurrent-frozen-calibration-reproduction \
  --artifacts runs/defense-recurrent-frozen-calibration-reproduction/artifacts \
  --resume results/defense/training/recurrent-frozen-initial-01/checkpoint \
  --steps 131072

# Continue the verified calibration without a training/action limit:
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-frozen-memory-reproduction \
  --artifacts runs/defense-frozen-memory-reproduction/artifacts \
  --resume results/defense/training/recurrent-frozen-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
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
demonstrations, stored emulator snapshots or pretrained features. The default
has no state resets; the optional own-state mode below collects its own new
states during training. Uniform random
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

At **1,300,000**, ordinary DQN's best increased to **420**, although its
[mean 366 and median 360](results/defense/training/dqn-22-fresh/step-000001300000/evaluation.json)
fell from the previous batch. Its full checkpoint and
[1,785-action verified replay](results/defense/training/dqn-22-fresh/replay-420/replay.html)
are preserved. All ten games remained in stage 1 without a mission; a higher
single effort does not imply a better mean or deeper progression.

At **1,400,000**, ordinary DQN reached
[mean 458, median 470, best 500](results/defense/training/dqn-22-fresh/step-000001400000/evaluation.json).
Its full online/target/optimizer checkpoint and
[1,963-action verified replay](results/defense/training/dqn-22-fresh/replay-500/replay.html)
are preserved. All ten complete games still ended in stage 1 without a mission.

At **1,600,000**, ordinary DQN reached
[mean 991, median 995, best 1,390](results/defense/training/dqn-22-fresh/step-000001600000/evaluation.json),
with a [2,239-action verified replay](results/defense/training/dqn-22-fresh/replay-1390/replay.html).
At **1,700,000**, it improved to
[mean 1,372, median 1,470, best 1,670](results/defense/training/dqn-22-fresh/step-000001700000/evaluation.json),
with a [2,201-action verified replay](results/defense/training/dqn-22-fresh/replay-1670/replay.html).
Both full online/target/optimizer checkpoints are preserved. All twenty games
remained stage-1 losses. The next 1,800,000-action batch regressed to mean
392, median 400, best 440: improvement is not monotonic, and these milestones
do not replace the stronger PPO global best.

At **2,000,000**, DQN's
[ten-game mean was 1,187, median 1,260, best 2,060](results/defense/training/dqn-22-fresh/step-000002000000/evaluation.json).
The [2,089-action verified replay](results/defense/training/dqn-22-fresh/replay-2060/replay.html)
and full optimizer/target checkpoint are preserved. This is a new single-game
best for the independent DQN lineage, but its mean is below the 1,700,000-action
checkpoint's 1,372. All ten games remained stage-1 losses.

Its later **2,200,000** checkpoint improved the ten-game mean to
[1,467, median 1,270, best 2,020](results/defense/training/dqn-22-fresh/step-000002200000/evaluation.json).
That full online/target/optimizer checkpoint is also preserved. The lower
single-game best did not replace its existing 2,060-point replay, and no game
reached stage 2 or a mission.

Run 22 subsequently stopped after **3,030,192** actions, **188,761** updates
and **1,784** complete training games. Its [final resumable checkpoint](results/defense/training/dqn-22-fresh/final-checkpoint/state.json)
and [entire training log](results/defense/training/dqn-22-fresh/metrics.jsonl)
are preserved alongside the best-effort and peak-mean checkpoints above.
Across **30** complete validation batches, none reached stage 2 or a mission.
The [last batch at 3,000,000](results/defense/training/dqn-22-fresh/step-000003000000/evaluation.json)
averaged **356**, median **360**, best **380**. The six final validation means
were 324, 402, 362, 372, 376 and 356. This persistent regression, rather than
a wall-clock limit, motivated replacing its compute slot with run 28 below.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-reproduction \
  --artifacts runs/defense-dqn-reproduction/artifacts
# Later, resume with a fresh replay buffer from the saved online/target/Adam state:
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-continuation \
  --artifacts runs/defense-dqn-continuation/artifacts \
  --resume runs/defense-dqn-reproduction/latest
```

### Persistent random exploration

The shared-barrier replay diagnostic motivates testing **temporal persistence**
in exploration, without supplying a route or using the diagnostic frames as
training examples. [Temporally-Extended Epsilon-Greedy Exploration](https://arxiv.org/abs/2006.01782)
studies randomly sampled actions held for sampled durations. Our optional
ordinary-DQN adaptation is not a reproduction of its Atari agents: it uses
this repository's existing network, score-only reward and primitive-step
replay, with bounded durations and an occupancy-calibrated exploration rate.

`--exploration-max-repeat 1` (the default) executes the **unchanged** independent
epsilon-greedy branch, including its original RNG draws. A larger value enables
`rl.persistent_exploration`: sample uniformly among all existing actions, then
hold that action for a random number of original environment steps. For the
first comparison, durations are 1–64 with probability proportional to
`length**(-1.5)`. No observation, obstacle detector, location, score threshold,
or game-private state selects the start, action or duration. All directions,
firing aliases and no-op remain eligible; there is no preference for right.

Each held primitive step still produces its own screen, visible-score reward,
n-step replay transition and normal update opportunity. This is **not** a new
evaluation action repeat or a change in emulation speed. Visible life loss,
termination and truncation cancel outstanding holds. Replay warmup retains
the old independent uniform random actions. Saved online/target/optimizer and
replay semantics are unchanged. An independent exploration RNG is saved and
restored; active holds and local diagnostic counters clear when resume boots
new games. Bootstrap-head combinations are currently rejected explicitly.

To separate longer sequences from simply adding more random actions, epsilon
denotes a nominal exploratory-step fraction. If mean duration is `m`, an idle
worker starts a random hold with probability `epsilon / (m*(1-epsilon)+epsilon)`.
This is the renewal-process occupancy formula, not the paper's unadjusted
option-start epsilon. Here `m = 6.17855` and epsilon 0.05 gives start probability
**0.00844648**. Stationary uninterrupted occupancy is 5%; episode/life cuts and
changing epsilon can change realized occupancy. Logs and checkpoints retain
actual exploratory steps, duration/action counts and cancelled future steps.

Greedy complete-game validation remains **entirely learned, at the original
100,000-T-state decision cadence**. The evaluation loader never instantiates
the training exploration helper. No reward bonus, demonstration, hidden RAM,
scripted route, or evaluation-only action override is introduced.

The first paired calibration starts both arms from the same preserved ordinary
DQN-28 peak at **6,100,000**, not from either earlier short learning-rate check.
Each collects **131,072** new primitive actions and ends at **6,231,072**.
Both retain eight workers, compact capacity 200,000, batch 64, learning rate
0.0001, n-step 5, gamma 0.997, life learning boundaries, 10,000-entry replay
warmup, update interval 16, target interval 2,000, epsilon 0.05 and no own-state
resets. Each refills replay with its own newly collected experience. The
[control configuration](results/defense/training/persistent-control-01/resume-config.json)
and [persistent configuration](results/defense/training/persistent-repeat-01/resume-config.json)
differ only in maximum repeat, output paths and exploration provenance.
They share the same reused ten validation seeds; neither is a fresh test or
eligible source for the global collector. A complete stage clear remains the
objective, not a higher training score under random exploratory actions.

All **274 regression tests** pass. The five added tests cover the duration
distribution and occupancy calibration, exact holds and boundary cancellation,
RNG reproducibility, invalid CLI combinations, real native training and resume,
and byte-for-byte array equality of default versus explicitly disabled runs
(online, target and optimizer, with identical counters/RNG). A 384,000-decision
synthetic check exercises the nominal 5% occupancy and long-duration tail;
it is a sampler test, not game-performance evidence. The
[complete test log](results/defense/diagnostics/persistent-regression-tests.txt)
is preserved. Existing learners continue using their originally loaded code;
new checkpoints record the exact new trainer/helper source hashes.

Both bounded arms exited normally and independently verified their best replay:

| Calibration | Ten-game mean | Median | Best | Verified greedy replay |
| --- | ---: | ---: | ---: | --- |
| [Independent-action control](results/defense/training/persistent-control-01/checkpoint/evaluation.json) | 10,008 | 10,245 | 10,350 | [2,414 actions](results/defense/training/persistent-control-01/replay/replay.html) |
| [Persistent random exploration](results/defense/training/persistent-repeat-01/checkpoint/evaluation.json) | 10,240 | 10,180 | 10,440 | [2,485 actions](results/defense/training/persistent-repeat-01/replay/replay.html) |

All twenty games lost in stage 1. The [paired mean gain of **232**](results/defense/training/persistent-repeat-01/comparison.json)
comes largely from **one 2,370-point difference**: five seeds improved and five
worsened, and the persistent median is lower. Both means remain below the
untouched parent's 10,290. This is weak early evidence, not a demonstrated
exploration advantage or barrier passage. The control completed **57** new
boot games during training, versus **56** for persistence; neither used resets.
After warmup, persistence recorded **5,798 / 121,040** exploratory primitive
steps (**4.790%**), 929 sampled holds and 264 future steps cancelled at boundaries.
Full online/target/optimizer/RNG checkpoints, configurations, logs and verified
replays are preserved for both arms. Training games and exploratory actions
are not substituted for greedy validation results.

A [screen-only check of these two selected calibration replays](results/defense/diagnostics/shared-loss-persistent-01/report.json)
also does **not** establish passage through the diagnosed obstacle sequence.
The [control's life totals](results/defense/diagnostics/shared-loss-persistent-01/policy-1-losses.png)
are **2,600 / 2,580 / 2,600 / 2,570**; the
[persistent model's totals](results/defense/diagnostics/shared-loss-persistent-01/policy-2-losses.png)
are **2,600 / 2,600 / 2,600 / 2,640**. Some windows show explosion graphics near
the preceding center-gap barrier while the broad barrier is still well above
the ship; they are not all identical to the older best's later approach.
The same flash-alignment limitations apply, and no exact collision cause or
course index was measured. In particular, one 2,640-point life is not proof
of crossing the barrier. These are read-only views of already verified games,
not additional evaluations, learning inputs or a change to the running trial.

To test durability with a matched baseline, unlimited full trials now continue
each arm's own **6,231,072** checkpoint:
[DQN 31 persistent configuration](results/defense/training/dqn-31-persistent/resume-config.json)
and [DQN 32 control configuration](results/defense/training/dqn-32-persistent-control/resume-config.json).
They keep all learning settings, change the validation interval to 200,000,
and remove the training action cap; full games remain uncapped. Each restores
its online/target/optimizer and RNG states, boots new games and refills its
own replay buffer, so this is not exact continuation of prior trajectories.
They are not independently initialized replicates. Frozen-memory PPO 30 and
DQN 24 subsequently retired with all state preserved.
The sole collector was stopped cleanly,
confirmed gone, then restarted with both full-run sources and all 26 historical
sources. Short calibration sources remain excluded. Any new global best still
requires independent frozen-policy replay verification.

The [first full-run comparison at **6,431,072**](results/defense/training/dqn-31-persistent/comparison-at-000006431072.json)
adds **200,000** actions per arm after calibration (**331,072** after the common
original parent). It reverses the short-run mean advantage:

| Full trial | Ten-game mean | Median | Best | Verified greedy replay |
| --- | ---: | ---: | ---: | --- |
| [DQN 31 persistent](results/defense/training/dqn-31-persistent/step-000006431072/evaluation.json) | 9,373 | 10,280 | 10,310 | [2,483 actions](results/defense/training/dqn-31-persistent/first-replay/replay.html) |
| [DQN 32 control](results/defense/training/dqn-32-persistent-control/step-000006431072/evaluation.json) | 10,278 | 10,280 | 10,280 | [2,545 actions](results/defense/training/dqn-32-persistent-control/first-replay/replay.html) |

All twenty complete games lost in stage 1. Persistence is **905 points lower
on average**, with two higher, four lower and four tied seeds. Its exploratory
step fraction during this continuation was **5.212%** (9,902 / 189,968 after
warmup), not a sudden increase in random-action volume. The learners completed
**89** and **88** new boot games respectively, with no restored segments.
Both full online/target/optimizer/RNG checkpoints and independently verified
replays are preserved. The earlier, higher-scoring calibration replays are
unchanged. This does not establish that persistence helps; both trials continue
unchanged for further matched rounds, without treating a single batch as a
success or a definitive rejection. Neither replaced the shared 10,480 best.

The [second matched round at **6,631,072**](results/defense/training/dqn-31-persistent/comparison-at-000006631072.json)
adds **400,000** actions per arm after calibration (**531,072** after the common
parent). [Persistence](results/defense/training/dqn-31-persistent/step-000006631072/evaluation.json)
averaged **10,344**, median **10,350**, best **10,430**;
[control](results/defense/training/dqn-32-persistent-control/step-000006631072/evaluation.json)
averaged **10,042**, median **10,220**, best **10,360**. The mean difference is
now **+302**, with eight higher and two lower paired seeds, reversing the
first round's −905. All twenty games still lost in stage 1. Persistence is
54 points above the untouched parent's mean 10,290, but this fluctuating
comparison is not a durable advantage or a depth gain.

Both full checkpoints and their independently verified
[2,538-action persistent replay](results/defense/training/dqn-31-persistent/replay-10430/replay.html)
and [2,579-action control replay](results/defense/training/dqn-32-persistent-control/replay-10360/replay.html)
are preserved. Since calibration, they completed **174** and **170** new boot
games respectively, with no restored segments. The cumulative measured
persistent exploratory-step fraction was **4.930%**. Earlier checkpoints and
higher-scoring calibration replays remain untouched. Both learners continue
unchanged, and the shared best is still the original verified 10,480 replay.

The [third paired round at **6,831,072**](results/defense/training/dqn-31-persistent/comparison-at-000006831072.json)
adds **600,000** actions per arm after calibration. Persistence averaged
**10,303**, median **10,305**, best **10,420**; control averaged **10,188**,
median **10,240**, best **10,260**. The mean difference is **+115**. All twenty
games again lost in stage 1. Both full checkpoints and complete game records
are preserved; neither beat its earlier archived best replay, so those verified
replays remain the references. This is no stage-depth improvement.

At the [fourth round, **7,031,072**](results/defense/training/dqn-31-persistent/comparison-at-000007031072.json),
persistence reached a new personal peak mean **10,450**, median **10,455**,
best **10,480**. Its independently verified
[2,526-action replay](results/defense/training/dqn-31-persistent/replay-10480/replay.html)
is preserved with the full online/target/optimizer/RNG checkpoint. Control
averaged **10,199**, median/best **10,220**. All ten paired differences favor
persistence, averaging **+251**, but all twenty games still lost in stage 1.
The runs have completed **342** and **343** new boot games since calibration.
This matches the global score ceiling without passing it or establishing a
stage clear; the existing shared replay remains unchanged.

The [six-round comparison through **7,431,072**](results/defense/training/dqn-31-persistent/comparison-at-000007431072.json)
shows another reversal. At 7,231,072 the means were **10,195** persistent and
**9,866** control (+329). At 7,431,072 they were **10,239** persistent versus
**10,304** control (−65), with four higher and six lower paired persistent
scores. Persistent median/best were **10,250 / 10,320**; control median/best
were **10,320 / 10,440**, a new control-run personal best. Its independently
verified [2,588-action replay](results/defense/training/dqn-32-persistent-control/replay-10440/replay.html)
and both complete checkpoints are preserved. The earlier persistent 10,480
replay remains unchanged. Since calibration, the runs completed **508** and
**516** new boot games respectively. All **120** evaluation games across these
six reused-seed rounds per arm lost in stage 1. The evidence remains mixed on
mean score and entirely negative on a new stage; it is not a fresh success rate.

```bash
# Use distinct run/artifact paths for each arm. Set repeat to 1 for the control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reproduction \
  --artifacts runs/defense-persistent-reproduction/artifacts \
  --resume results/defense/training/dqn-28-large-replay/step-000006100000 \
  --steps 6231072 --eval-every 131072 \
  --exploration-max-repeat 64 --exploration-exponent 1.5

# Unlimited continuation of the verified persistence calibration:
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-full-reproduction \
  --artifacts runs/defense-persistent-full-reproduction/artifacts \
  --resume results/defense/training/persistent-repeat-01/checkpoint \
  --steps 0 --eval-every 200000
```

### Persistent exploration-rate comparison

The [saved duration counters at 6,631,072](results/defense/diagnostics/persistent-duration-6631072.json)
show **3,240** sampled holds over **174** new boot games, but only **131**
sampled durations of at least 32 steps and **53** of at least 48. These counts
include firing/no-op actions and animations; life/episode boundaries can
shorten a sampled hold. The saved marginals cannot identify long **movement**
holds or where they occurred. They do not prove insufficient exploration is
the cause, or that any particular duration would solve the barrier.

They motivate a controlled intensity check rather than assuming the 5% arm
already tried many long maneuvers at the relevant states. Both new bounded
arms start from the **same** preserved DQN-31 checkpoint at **6,631,072**
(mean 10,344), with identical online/target/optimizer and saved RNG ancestry.
Each adds **131,072** actions, ending at **6,762,144**. Their
[control configuration](results/defense/training/persistent-rate-control-01/resume-config.json)
and [higher-rate configuration](results/defense/training/persistent-rate-high-01/resume-config.json)
differ only in paths and **epsilon-final 0.05 versus 0.25**. Both use persistent
durations 1–64 with exponent 1.5, eight workers, compact capacity 200,000,
unchanged learning settings and no own-state resets. Replay refills from new
own experience after boot; no saved evaluation trajectory is loaded.

Under the existing occupancy formula, idle start probabilities are
**0.00844648** and **0.05118847**. Realized fractions can differ because of
boundary cancellation. All action IDs remain uniformly eligible; no direction,
position, obstacle, score threshold or route triggers exploration. This tests
**more random exploratory steps**, not a longer duration distribution or a
change to score-only rewards. Both frozen evaluations remain pure learned
greedy play at the ordinary cadence, on the same ten reused validation seeds.
The full low-rate run 31 and its independent-action control 32 continue
unchanged. Retired bootstrap run 24 released compute; neither bounded source
is included in the shared collector. Existing tested code implements this
comparison without a learner/environment change.

The [completed paired result](results/defense/training/persistent-rate-high-01/comparison.json)
does **not** support increasing the rate on this evidence:

| Nominal persistent fraction | Measured fraction after warmup | Ten-game mean | Median | Best | Verified replay |
| --- | ---: | ---: | ---: | ---: | --- |
| 5% control | 4.542% | 10,324 | 10,280 | 10,480 | [2,509 actions](results/defense/training/persistent-rate-control-01/replay/replay.html) |
| 25% higher rate | 23.869% | 10,115 | 10,110 | 10,260 | [2,458 actions](results/defense/training/persistent-rate-high-01/replay/replay.html) |

The higher rate is **209 points lower on average**, with nine lower paired
scores and one tie. All twenty complete validation games lost in stage 1.
The arms completed **56** and **65** new boot training games respectively,
also without a later-stage or mission event. Both stopped normally at their
predeclared action budget; their full online/target/optimizer/RNG checkpoints,
compressed complete logs and independently verified replays are preserved.
This is one bounded paired comparison on reused validation seeds, not a
definitive rejection of persistent exploration. Neither becomes a new unlimited
trial, and the global best remains unchanged.

A [read-only loss comparison](results/defense/diagnostics/shared-loss-persistent-rate-01/report.json)
reinforces the user's observation. The
[control's four lives](results/defense/diagnostics/shared-loss-persistent-rate-01/policy-1-losses.png)
each scored **2,620**, with the ship near the center/left as the broad barrier
with an opening on the right approaches. The
[higher-rate model](results/defense/diagnostics/shared-loss-persistent-rate-01/policy-2-losses.png)
scored **2,500 / 2,620 / 2,600 / 2,540** across its lives, including losses
around the preceding center-gap barrier. Neither shows passage through this
obstacle sequence. These images align to visible flashes or the final visible
loss, not exact collision timestamps; they do not establish a required route
or the underlying learning failure. No replay data was used for training.

```bash
# Use separate output paths and epsilon-final 0.05 for the matched control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-rate-reproduction \
  --artifacts runs/defense-persistent-rate-reproduction/artifacts \
  --resume results/defense/training/dqn-31-persistent/step-000006631072 \
  --steps 6762144 --eval-every 131072 --epsilon-final 0.25
```

### Persistent exploration with own-state resets

The higher-rate trial above mostly repeated full boot games without escaping
the common obstacle sequence. A bounded follow-up tests whether reusing states
reached during **its own new training** gives sustained random exploration more
opportunities from later positions. This is a hypothesis about training-state
coverage, not a claim that the archive necessarily concentrates on a specific
obstacle or contains a solution.

`defense-persistent-reset-calibration-01` starts from the **same 6,631,072**
online/target/optimizer/RNG parent as both rate checks, **not** from the
higher-rate trial's final weights. It adds **131,072** actions with nominal
persistent exploration 25%, lengths 1–64 and exponent 1.5. Its comparator is
the already completed `persistent-rate-high-01`, so the changed treatment is
the own-state reset setup: probability **0.5**, sharing, **two of eight**
workers permanently boot-only, and **128-action** archive lookback. All other
learning, observation, action timing, replay capacity and evaluation settings
match. This is one sequential paired lineage, not simultaneous independent
replicates. Wall-clock throughput is not its outcome measure.

The archive begins empty and uses the existing bounded visible-score cells.
Only newly reached opaque native snapshots are shared inside this run; none
are decoded, fabricated or taken from an evaluation replay or another learner.
Restored starting score is not credited as reward. Every primitive action
still comes from the learned policy or uniform, screen-independent random
exploration. The longer lookback is the existing generic archive setting,
not an obstacle detector or steering instruction. Boot games and restored
segments are reported separately, and all ten uncapped greedy evaluation
games start from boot. This short source is excluded from the global collector.

Two new regression tests cover this combination without changing production
code. A synthetic boundary fixture confirms held actions cancel both at a
visible life loss and at an own-state reset, replay ends on the actual terminal
screen, reset observations start new transitions, and only new score increments
are recorded. A real-emulator test exercises compact replay, new archive
creation, a reserved boot worker, restored segments and optimizer updates.
Its deliberately truncated episodes test plumbing, not successful gameplay.
All seven focused persistent-exploration tests pass. The full **276-test**
suite also [passes](results/defense/diagnostics/persistent-reset-regression-tests.txt).

An [interim training-only archive audit at **6,698,992**](results/defense/diagnostics/persistent-reset-interim-6698992.json) found that retained
save events had source progress at most **170** points within a life, despite
trigger progress reaching **2,620**. Looking back 128 actions rewinds across
the large score jumps. This does not by itself identify obstacle positions or
prove those states are unhelpful, but it cautions against claiming that this
configuration provides near-failure practice. A second bounded arm,
`defense-persistent-reset-short-calibration-01`, therefore starts from the same
6,631,072 parent with identical settings except **lookback 32** and output
paths. The intended comparison includes both actual archive coverage and
complete from-boot evaluation against lookback 128 and the existing no-reset
high-exploration comparator. Neither reset arm imports the other's experience.

The [lookback-128 trial has finished](results/defense/training/persistent-reset-calibration-01/comparison.json)
at **6,762,144**, with ten complete greedy games averaging **10,153**, median
**10,130**, best **10,310**, all stage-1 losses. That is **+38** over the
no-reset comparator's 10,115, with six higher and four lower paired scores,
and remains below the common parent's 10,344. Its best
[2,596-action replay](results/defense/training/persistent-reset-calibration-01/replay/replay.html)
was independently reproduced from frozen weights; full online/target/optimizer/
RNG state and complete compressed logs are preserved. It stopped normally at
the declared budget and is not promoted to an unlimited run.

Training completed **50** new boot games and **39** restored segments, all
without a later-stage event. Both reserved workers stayed boot-only. All
**1,639** archive events had exactly 128-action lookback; the **127** retained
save events reached at most **190** points of source within-life progress,
while triggers reached 2,620. This confirms the earlier coverage limitation
through the end of this check, without claiming score measures course distance.
The shorter-lookback result follows below.

The [32-action-lookback arm](results/defense/training/persistent-reset-short-calibration-01/comparison.json)
also stopped normally at **6,762,144**. Its ten complete greedy games averaged
**10,339**, median **10,340**, best **10,390**, all stage-1 losses. Relative to
the no-reset high-exploration comparator this is **+224**, with all ten paired
scores higher. Relative to lookback 128 it is **+186**, with seven higher and
three tied. It is still **5 points below the common parent's mean**, with no
stage-depth gain. This one sequential paired lineage on reused seeds suggests
a useful setting to test further, not a demonstrated solution or fresh success
rate. Full weights/target/optimizer/RNG, compressed logs and its independently
verified [2,580-action replay](results/defense/training/persistent-reset-short-calibration-01/replay/replay.html)
are preserved. No global replay was replaced.

Training completed **54** new boot games and **30** restored segments, all
stage 1. Both boot-only workers remained boot-only. All **1,723** archive
events used exactly 32-action lookback; **191** retained save events reached
source within-life progress **2,570**, versus the long-lookback arm's **190**.
This establishes different saved-state coverage by visible score, not an
obstacle coordinate, exact collision timing, or a count of current archive
contents. The [read-only loss report](results/defense/diagnostics/shared-loss-persistent-reset-01/report.json)
records best-replay life scores of **2,500 / 2,570 / 2,620 / 2,620** for lookback
128 and **2,600 / 2,570 / 2,600 / 2,620** for lookback 32. These remain near the
old score ceiling, with no verified barrier passage or stage transition.

To test whether the bounded reset result lasts, full runs now continue each
arm's own **6,762,144** checkpoint:
[DQN 33 shorter-lookback reset configuration](results/defense/training/dqn-33-persistent-resets/resume-config.json)
and [DQN 34 no-reset configuration](results/defense/training/dqn-34-persistent-rate-control/resume-config.json).
Both retain nominal 25% persistent exploration, eight workers, compact replay
capacity 200,000, batch 64, learning rate 0.0001, n-step 5, discount 0.997,
life learning boundaries and the original 100,000-T-state/stride-1 timing.
Run 33 retains 0.5 shared own-state resets, two boot-only workers and lookback
32; run 34 retains no resets. Training is uncapped, with ten complete greedy
from-boot evaluations every 200,000 new actions, first at **6,962,144**.

Both restore their own online/target/optimizer and RNG states but refill
experience from newly booted games; run 33 also rebuilds its archive from
empty. Thus this is not exact continuation of trajectories, nor independent
random-seed replication. No evaluation replay or other run's snapshots are
training data. Trials 31/32 continue unchanged as the low-rate persistence/
independent-action comparison. Retiring PPO 30 releases capacity for these
four eight-worker DQN learners.

The sole replay collector was stopped cleanly and confirmed gone before
restarting with [all 30 full-run sources](results/defense/training/dqn-33-persistent-resets/collector-config.json),
including the new pair and all historical sources. Short calibrations remain
excluded. Any global replacement still requires an independently reproduced
frozen-policy replay and a higher stage/mission/score rank; the existing best
is never replaced just for a higher mean.

The [first full comparison at **6,962,144**](results/defense/training/dqn-33-persistent-resets/comparison-at-000006962144.json)
adds **200,000** actions per arm after calibration (**331,072** after the common
6,631,072 parent):

| Full trial | Ten-game mean | Median | Best | Independently verified replay |
| --- | ---: | ---: | ---: | --- |
| [DQN 33 own resets](results/defense/training/dqn-33-persistent-resets/step-000006962144/evaluation.json) | 10,259 | 10,430 | 10,480 | [2,564 actions](results/defense/training/dqn-33-persistent-resets/first-replay/replay.html) |
| [DQN 34 no resets](results/defense/training/dqn-34-persistent-rate-control/step-000006962144/evaluation.json) | 10,197 | 10,150 | 10,460 | [2,558 actions](results/defense/training/dqn-34-persistent-rate-control/first-replay/replay.html) |

All twenty complete games lost in stage 1. The reset advantage narrows to
**+62**, with eight higher and two lower paired scores, including one −1,350
outlier. This is not a durable advantage or a new stage; both means remain
below the common parent's 10,344. Matching the global single-game ceiling
does not replace the shared best. Both full online/target/optimizer/RNG
checkpoints, game records and verified replays are preserved.

The reset arm completed **83** new boot games and **45** restored segments;
control completed **99** new boot games, with none restored. Both had no
later-stage training event. Measured exploratory-action fractions were
**24.467%** and **24.388%**, respectively. All **2,753** reset-arm archive
events used 32-action lookback, the **241** retained save events reached
within-life source score **2,600**, and both reserved workers stayed boot-only.
These counts describe retained events over time, not current archive contents
or measured obstacle positions. Both learners continue unchanged for further
matched rounds; no new source data, reward or controller was introduced.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reset-reproduction \
  --artifacts runs/defense-persistent-reset-reproduction/artifacts \
  --resume results/defense/training/dqn-31-persistent/step-000006631072 \
  --steps 6762144 --eval-every 131072 --epsilon-final .25 \
  --curriculum-probability .5 --curriculum-share \
  --curriculum-boot-envs 2 --curriculum-lookback 128

# Unlimited continuation of the verified 32-action-lookback arm:
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reset-full-reproduction \
  --artifacts runs/defense-persistent-reset-full-reproduction/artifacts \
  --resume results/defense/training/persistent-reset-short-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
# For the matched comparator, use distinct output paths and resume
# results/defense/training/persistent-rate-high-01/checkpoint instead.
```

### Lossless compact training replay

`rl.defense_dqn --compact-replay` optionally replaces dense duplicated screen
stacks with exact frame identities and reference-counted byte storage. Each
observation and next-observation still records its **four exact frames**;
there is no inference from temporal adjacency, worker order, n-step distance,
life boundaries or restored-state history. Byte-key equality resolves hash
collisions without approximation. When ring slots are overwritten, unreferenced
frames are released and their IDs reused; sampled arrays are independent,
writable copies. No native state is inspected and no RNG draw is added.

Actions, rewards, discounts, priorities, sampling weights, bootstrap memberships,
model inputs and update rules are unchanged. The option supports both ordinary
and bootstrap DQN, defaults off, and is inherited on optimizer resume. Replay
still refills from new experience after resume; it is not a loaded demonstration
archive. Configuration records the storage implementation hash. Logged storage
figures distinguish frame payload and index bytes from the equivalent dense
arrays; these figures **exclude Python metadata and the other replay/model
allocations**, and are not total process-memory measurements.

All **246 regression tests** passed. New checks exercise ring wraparound,
repeated boot frames, exact reference counts, sampling without aliases,
arbitrary history spacing, real own-state resets, n-step terminal/truncation
handling, identical priority trees, bootstrap masks and RNGs. Real-emulator
dense/compact training produced **exactly equal online, target and optimizer
arrays**, episode/update counters and RNG states for both ordinary and bootstrap
DQN, including ring reuse; compact resumes were also exercised.

This is intended to make larger own-experience buffers practical on the Mac
Mini, not to change the policy or claim a learning gain from storage alone.
The existing live learners have not been restarted or silently switched.

An [isolated compact-storage check](results/defense/training/compact-replay-smoke-01/config.json)
repeated the original fresh 16,384-action DQN check's learning settings. It
reproduced **every array** in the online model (12), target model (12) and
optimizer (26), the common counters/RNG state, and **all ten complete game
records exactly**. The [parity record](results/defense/training/compact-replay-smoke-01/parity.json)
documents this comparison. Mean/median/best remained **328 / 320 / 360**, all
stage 1; its [replay](results/defense/training/compact-replay-smoke-01/replay/replay.html)
verified **1,655** actions. Full weights, target, optimizer, configuration and
log are preserved. It exited normally and is excluded from the collector.

At its logged **15,416**-action point, the full 8,192-transition buffer held
**3,739** distinct frames: **3,828,736** payload bytes plus **262,144** index
bytes, about **3.90 MiB** versus **64 MiB** for dense screen arrays. Python
metadata and other allocations are excluded from this comparison. This
validates a substantial reduction in duplicated screen data in this check,
not a total-RAM or speedup claim, nor evidence that a larger buffer improves
learning. No recorded evaluation trace was used as training experience.

### Larger-buffer DQN comparison

`defense-dqn-28-large-replay` starts independently from fresh seed-97 weights,
with **200,000** replay transitions instead of run 22's **50,000**, using the
lossless compact storage above. It does not resume the storage check or any
previous model. Its [configuration](results/defense/training/dqn-28-large-replay/config.json)
retains ordinary Double DQN's eight workers, batch 64, learning rate 0.0001,
five-step returns, gamma 0.997, 100,000-T-state actions, stride 1, life
boundaries, 10,000-action warmup, update every 16 aggregate actions, target
copy every 2,000 updates and epsilon decay to 0.05. Own-state resets and
bootstrap heads are disabled. Training and complete-game evaluation remain
uncapped; evaluation runs every 100,000 actions from boot.

More retained own experience is a hypothesis to test, not a diagnosed fix
for run 22's regression. Source hashes differ because intervening optional
features were added; default compatibility and exact dense/compact training
parity were tested as documented above. No policy input, reward bonus or
evaluation demonstration has been added. The sole collector includes run 28
and retains all historical sources; other live learners continue unchanged.

Its [first ten complete games at 100,000](results/defense/training/dqn-28-large-replay/step-000000100000/evaluation.json)
averaged **268**, median **260**, best **280**, all stage-1 losses. This is below
run 22's same-counter mean 280, not an early performance improvement. The full
online/target/optimizer checkpoint and [1,477-action verified replay](results/defense/training/dqn-28-large-replay/first-replay/replay.html)
are preserved.

At **200,000**, it reached
[mean 316, median 320, best 320](results/defense/training/dqn-28-large-replay/step-000000200000/evaluation.json),
with a [1,631-action verified replay](results/defense/training/dqn-28-large-replay/replay-320/replay.html).
The full optimizer/target checkpoint is preserved. This is above run 22's
same-counter mean/best 280, but all ten games still lost in stage 1. One early
batch on reused seeds does not establish improved depth or durable stability.

At **400,000**, run 28 reached
[mean 328, median 320, best 340](results/defense/training/dqn-28-large-replay/step-000000400000/evaluation.json).
The intervening 300,000-action mean was 298, best 300, so the improvement was
not monotonic. The full checkpoint and [1,610-action verified replay](results/defense/training/dqn-28-large-replay/replay-340/replay.html)
are preserved. This is close to run 22's same-counter mean 322 and equal best
340; all games remained stage-1 losses, not evidence of a clear advantage.

Longer training produced substantial gains, but also repeated regressions.
Its strongest single game was **10,370** at **5,700,000** actions:
[mean 9,898, median 10,285](results/defense/training/dqn-28-large-replay/step-000005700000/evaluation.json),
with a [2,515-action verified replay](results/defense/training/dqn-28-large-replay/replay-10370/replay.html).
Its peak mean came at **6,100,000**:
[10,290, median 10,290, best 10,360](results/defense/training/dqn-28-large-replay/step-000006100000/evaluation.json).
Both full online/target/optimizer checkpoints and every intermediate
best-effort replay/checkpoint are preserved. All remained stage-1 losses.
This exceeds run 22's achieved best, but uses substantially more experience
and does not isolate a causal buffer-size advantage at matched compute.

Run 28 stopped cleanly at **7,400,000** actions, **461,874** updates and
**3,783** complete training games. There were **73** complete evaluation
batches; the 7,400,000 checkpoint was not evaluated before shutdown.
The last four completed means were **300, 364, 378, 362**. Its
[final checkpoint](results/defense/training/dqn-28-large-replay/final-checkpoint/state.json),
[last completed evaluation](results/defense/training/dqn-28-large-replay/step-000007300000/evaluation.json)
and [full log](results/defense/training/dqn-28-large-replay/metrics.jsonl)
are preserved. A larger buffer alone did not prevent regression or produce
stage 2. Matched short continuation checks now compare learning rates
0.0001 and 0.000025 from the **6,100,000** peak, with otherwise identical
fresh-buffer restart settings. Neither check is a collector source or a
claimed improvement before its complete-game results are available.

Those short checks have now finished. Both started from the same peak and
added **32,768** actions, ending at **6,132,768**, with eight workers and the
same 200,000-transition compact capacity. Each buffer refilled from new own
experience with the existing 10,000-transition random warmup; no saved
training or evaluation trajectories were loaded. Their configurations differ
only in learning rate and output paths.

| Learning rate | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0.0001 control](results/defense/training/dqn-peak-control-01/checkpoint/evaluation.json) | 9,734 | 10,245 | 10,310 | [2,388](results/defense/training/dqn-peak-control-01/replay/replay.html) |
| [0.000025](results/defense/training/dqn-peak-low-lr-01/checkpoint/evaluation.json) | 5,367 | 5,190 | 7,580 | [2,286](results/defense/training/dqn-peak-low-lr-01/replay/replay.html) |

All twenty complete evaluation games lost in stage 1. Both means are below
the frozen parent's 10,290; the lower rate is substantially worse than the
matched restart control and is **not** being promoted to a full run. This
does not show that slowing updates fixes the long-run regression. Both
checks exited normally; full online/target/optimizer states, configuration,
logs and verified replays are preserved, separate from the collector.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-large-replay-reproduction \
  --artifacts runs/defense-dqn-large-replay-reproduction/artifacts \
  --seed 97 --capacity 200000 --compact-replay
```

### Ordinary DQN with its own reached-state resets

Ordinary DQN can now enable the same opaque, own-experience reset mechanism
used by PPO via `--curriculum-probability`, `--curriculum-share`,
`--curriculum-boot-envs` and `--curriculum-lookback`. This initial integration
uses score cells only: visible score gains within the current life, 20-point
bins, 16 bins per stage and four retained states per bin. Boot-only workers
share newly reached states but never restore one. The archive is initially
empty, stays bounded, and is rebuilt on optimizer resume; native payloads are
neither decoded nor fabricated, serialized into model weights, or policy input.
No PPO trajectory, evaluation replay, demonstration or archived native state
is loaded into DQN training.

Only new visible-score differences enter n-step replay. At terminal/life
boundaries, pending returns end at the actual final screen, not the next
reset screen; restored starting score is not credited as reward. Boot-game
counts and recent scores are separated from restored-segment counts and new
segment reward. Full evaluation and replay verification still use the ordinary
from-boot environment without curriculum options. Bootstrap-head exploration
and these resets cannot yet be combined; the CLI rejects that combination.
The default probability is zero and existing learners continue unchanged.

All **240 regression tests** passed. New tests cover invalid combinations,
terminal-screen/return isolation, separate segment bookkeeping, unchanged
from-boot evaluation arguments, real own-state sharing, reserved boot workers,
and optimizer resume with fresh archives/replay. Truncated unit-test episodes
are bookkeeping checks, not evidence of actual game completion.

A paired short check resumed ordinary DQN's **1,700,000** checkpoint.
Each side trained **32,768** new actions, using four workers, an 8,192-transition
buffer refilled from new experience, 1,024 warmup transitions, and the parent's
batch 64, n-step 5, gamma 0.997, target-copy interval 2,000 and 100,000-T-state
action timing. One side enables 0.5 reset probability, shared score cells,
one boot-only worker and lookback 32; the other keeps resets disabled. Both
are excluded from the collector and evaluate ten complete games from boot.
Both exited normally at **1,732,768**, each with **107,608** cumulative optimizer
updates. Their configurations differ only in output paths and the reset setup
and its recorded provenance. Both refilled their replay buffers with new own
experience; neither loaded evaluation actions or another learner's states.

| Short continuation | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [No resets](results/defense/training/dqn-reset-control-01/checkpoint/evaluation.json) | 284 | 280 | 300 | [1,531](results/defense/training/dqn-reset-control-01/replay/replay.html) |
| [Own-state resets](results/defense/training/dqn-reset-smoke-01/checkpoint/evaluation.json) | 362 | 350 | 540 | [1,919](results/defense/training/dqn-reset-smoke-01/replay/replay.html) |

All twenty complete games were stage-1 losses. Reset scores were higher on nine
paired seeds and equal on one, mean difference **+78**, but **both** checks
regressed substantially from the parent's mean **1,372**. This one short pair
does not establish an overall learning advantage or explain that regression.
Worker count, replay capacity and warmup differ from the full production run.
The control completed 16 boot games; the reset side completed 14 boot games
and five restored segments. All **402** archive events had exact 32-action
lookback; **54** originated in already-restored segments. Worker 0 remained
boot-only, and every segment's logged reward equalled its new visible score.
Both complete checkpoints, configurations, logs and verified replays are
preserved separately; neither is a parent of the longer trial.

`defense-dqn-26-own-resets` instead starts **fresh at counter zero**, with seed
97 and the ordinary baseline's normal eight workers, 50,000-transition buffer,
10,000-transition random warmup, batch 64, n-step 5, gamma 0.997, target copies
every 2,000 updates and updates every 16 aggregate actions. Epsilon decays
from 1 to 0.05 over one million actions after warmup, as in run 22. It changes
the reset setup to probability 0.5, shared score cells, two permanently
boot-only workers and lookback 32. It uses neither short-check weights nor
pretrained features or native snapshots. The
[configuration](results/defense/training/dqn-26-own-resets/config.json)
records the fresh initialization and implementation hashes.

This longer fresh-start comparison tests whether resets help value learning;
it is not presented as a successful fix for the short-check regression.
Training and episodes are uncapped; ten from-boot evaluations occur every
100,000 actions. It uses part of stopped run 17's released compute. Runs 22,
24 and 25 continue unchanged. The sole collector includes run 26 and all
historical sources, excluding both short checks, with the same frozen-policy
verification before any global promotion.

Run 26's [first ten complete evaluations](results/defense/training/dqn-26-own-resets/step-000000100000/evaluation.json),
at **100,000** actions, averaged **204**, median **200**, best **260**, all
stage-1 losses without a mission. This is below ordinary run 22's same-counter
mean/median/best **280**, not an improvement. The full online/target/optimizer
checkpoint and [1,482-action verified replay](results/defense/training/dqn-26-own-resets/first-replay/replay.html)
are preserved. At that checkpoint, training had completed **50** boot games,
**27** restored segments and **5,624** optimizer updates. Segment counts are
not counted as full games. This fresh-start trial continues without promoting
over the stronger shared best.

At **200,000**, run 26 reached
[mean 282, median 280, best 300](results/defense/training/dqn-26-own-resets/step-000000200000/evaluation.json),
with a [1,598-action verified replay](results/defense/training/dqn-26-own-resets/replay-300/replay.html)
and full optimizer/target checkpoint preserved. All ten games remained stage-1
losses. This recovers from its first batch and is only slightly above ordinary
run 22's same-counter mean/best 280; it is not evidence of a robust advantage.

At **400,000**, run 26 reached
[mean 302, median 320, best 340](results/defense/training/dqn-26-own-resets/step-000000400000/evaluation.json),
with a [1,600-action verified replay](results/defense/training/dqn-26-own-resets/replay-340/replay.html).
At **600,000**, it reached
[mean 326, median 320, best 360](results/defense/training/dqn-26-own-resets/step-000000600000/evaluation.json),
with a [1,643-action verified replay](results/defense/training/dqn-26-own-resets/replay-360/replay.html).
Both full optimizer/target checkpoints are preserved; all games remained
stage-1 losses. These small lineage improvements do not replace the shared best.

After further training, run 26's best single game reached **8,360** at
**7,100,000** actions:
[mean 6,480, median 6,395](results/defense/training/dqn-26-own-resets/step-000007100000/evaluation.json),
with a [2,424-action verified replay](results/defense/training/dqn-26-own-resets/replay-8360/replay.html).
Its strongest mean was **6,901** at **7,200,000**, median **6,845**, best
**7,560**. Both complete checkpoints, plus all intermediate best-effort
checkpoints and replays, are preserved. Through 8,100,000, none reached stage
2 or a mission; the latest mean had fallen to 2,462. The learner continues
under review rather than treating its best score as current reliability.

Run 26 subsequently improved to **10,280** at **9,300,000**:
[mean 9,849, median 10,230](results/defense/training/dqn-26-own-resets/step-000009300000/evaluation.json),
with a [2,585-action verified replay](results/defense/training/dqn-26-own-resets/replay-10280/replay.html).
Its peak mean was [**10,130**, median/best **10,160**, at **9,100,000**](results/defense/training/dqn-26-own-resets/step-000009100000/evaluation.json).
All new best checkpoints and replays are preserved, including 9,780, 10,060,
10,180 and 10,220 points. After a sharp regression, its last seven validation
means were **334, 320, 316, 330, 400, 478, 472**. It stopped cleanly at
**10,487,648** actions, with **654,852** updates, **4,672** complete boot games
and **3,072** restored segments. None of its **104** complete validation
batches reached stage 2 or a mission. The
[final optimizer/target checkpoint](results/defense/training/dqn-26-own-resets/final-checkpoint/state.json),
[last validated checkpoint](results/defense/training/dqn-26-own-resets/step-000010400000/evaluation.json)
and [losslessly compressed full log](results/defense/training/dqn-26-own-resets/metrics.jsonl.gz)
are preserved. The original log and all local history remain intact. Its
sustained regression, not a wall-clock limit, prompted freeing its compute.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-own-resets-reproduction \
  --artifacts runs/defense-dqn-own-resets-reproduction/artifacts --seed 97 \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 2 \
  --curriculum-lookback 32
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

Run 24's [first ten complete games](results/defense/training/dqn-24-bootstrap/step-000000200000/evaluation.json),
at **200,000** actions and **11,874** updates, each scored **380**
(mean/median/best **380 / 380 / 380**), all stage 1 losses without a mission.
Its full online/target/prior/optimizer checkpoint and
[1,705-action verified replay](results/defense/training/dqn-24-bootstrap/first-replay/replay.html)
are preserved. Training had completed 128 boot games at that checkpoint.
Ordinary DQN's [same-counter ten games](results/defense/training/dqn-22-fresh/step-000000200000/evaluation.json)
each scored **280** on the same seeds. This is a local early advantage for the
ensemble setup, not evidence of improved depth, broad superiority or lower
compute cost. Architecture, priors and exploration schedule differ; it is not
a single-feature ablation or a comparison against a long zero-prior ensemble.
Both are one training seed with reused validation seeds. Run 24 continues
without replacing the stronger shared 10,480-point best.

At **800,000**, run 24's
[ten complete evaluations](results/defense/training/dqn-24-bootstrap/step-000000800000/evaluation.json)
all scored **520** (mean/median/best 520), still stage-1 losses. Its full
online/target/prior/optimizer checkpoint and
[2,006-action verified replay](results/defense/training/dqn-24-bootstrap/replay-520/replay.html)
are preserved. The intervening 400,000-action mean was 350 (best 380), and
the 600,000-action games all scored 320, so this recovery was not monotonic.
It is a new milestone for this lineage, not a global-best or depth improvement.

The long run eventually moved beyond that early plateau. At **5,200,000**,
it reached [mean 10,186, median 10,200, best 10,220](results/defense/training/dqn-24-bootstrap/step-000005200000/evaluation.json),
with a [2,524-action verified greedy-ensemble replay](results/defense/training/dqn-24-bootstrap/replay-10220/replay.html).
That full online/target/prior/optimizer checkpoint and all intervening
best-effort milestones are preserved. All ten games still lost in stage 1.
The next two means fell to 7,773 and 6,476, so this is a saved capability,
not a claim that the current policy is equally reliable. This run continues;
its earlier 800,000-action diagnostic below is not evidence about these
later learned heads.

At **6,000,000**, the ensemble improved to
[mean 10,258, median 10,260, best 10,280](results/defense/training/dqn-24-bootstrap/step-000006000000/evaluation.json),
with a [2,485-action verified replay](results/defense/training/dqn-24-bootstrap/replay-10280/replay.html).
At **7,000,000**, it reached
[mean **10,334**, median **10,330**, best **10,410**](results/defense/training/dqn-24-bootstrap/step-000007000000/evaluation.json),
with a [2,578-action verified replay](results/defense/training/dqn-24-bootstrap/replay-10410/replay.html).
Both full online/target/prior/optimizer checkpoints are preserved. This is
continued improvement within the bootstrap lineage, but still below the
shared 10,480-point best and entirely stage-1 losses. At 7,200,000 its mean
was 10,170, best 10,200. The learner subsequently paused cleanly for disk
pressure at **7,383,056** actions, **460,815** updates and **3,661** complete
training games. All **36** complete validation batches stayed in stage 1.
Its [pause checkpoint](results/defense/training/dqn-24-bootstrap/pause-checkpoint-000007383056/state.json),
[validation history](results/defense/training/dqn-24-bootstrap/validation-at-000007383056.json)
and [losslessly compressed complete log](results/defense/training/dqn-24-bootstrap/metrics-at-000007383056.jsonl.gz)
are preserved. Resuming restores weights, target, priors, optimizer and RNG,
but refills replay from new own experience; it is not an exact continuation
of the in-memory training buffer. No historical files were removed.

The first post-storage-resume DQN evaluation at
[**7,583,056**](results/defense/training/dqn-24-bootstrap/step-000007583056/evaluation.json)
averaged **10,090**, median **10,100**, best **10,200**, all ten stage-1 losses.
Its full online/target/prior/optimizer checkpoint is preserved. This is slightly
below the final pre-pause mean 10,170 and does not improve the saved 10,410-point
bootstrap best. The replay buffer refilled from new own experience after resume.

Run 24 later retired cleanly at **8,699,808** actions, **542,486** updates and
**4,216** complete boot games. All **42** ten-game validations stayed in stage 1;
its best remained 10,410 and peak mean 10,334 at seven million actions. The
last validation at **8,583,056** had mean **9,970**, median **9,980**, best **10,030**.
The [final online/target/prior/optimizer/RNG checkpoint](results/defense/training/dqn-24-bootstrap/final-checkpoint-000008699808/state.json),
[last validation checkpoint](results/defense/training/dqn-24-bootstrap/step-000008583056/evaluation.json),
[complete validation curve and retirement record](results/defense/training/dqn-24-bootstrap/retirement-000008699808.json)
and [full compressed log](results/defense/training/dqn-24-bootstrap/metrics-at-000008699808.jsonl.gz)
are preserved. The original process was confirmed exited. The prolonged depth
plateau motivated releasing compute for the controlled exploration-rate check;
this was not a wall-clock stop. No prior checkpoint or replay was removed.

A [frozen-head diagnostic at **800,000**](results/defense/diagnostics/bootstrap-800000-heads.json)
compared that checkpoint's greedy ensemble with each individual
learned head plus its saved random prior. Each fixed policy played ten complete,
uncapped games from boot on the reused seeds 10000–10009; a head was never
selected based on the current state or outcome. The ensemble reproduced all
ten saved game records exactly, and model/configuration hashes were unchanged.

| Fixed policy | Mean | Median | Best |
| --- | ---: | ---: | ---: |
| Ensemble | 520 | 520 | 520 |
| Head 0 | 482 | 500 | 520 |
| Head 1 | 508 | 500 | 520 |
| Head 2 | 520 | 520 | 520 |
| Head 3 | 440 | 420 | 520 |
| Head 4 | 368 | 380 | 400 |

All **60** games lost in stage 1. At this checkpoint, ensemble averaging is
not concealing a better single-head score or stage reach on these seeds.
This does not diagnose the training plateau or rule out different behavior at
future checkpoints. The probe makes no parameter updates, is excluded from
promotion and never supplies actions or traces to training. No production
evaluation rule was changed. All **248 regression tests** passed, including
fixed-head selection, shape validation and unchanged evaluation RNG checks.

Repeating the same frozen diagnostic at the much stronger
[5,200,000-action checkpoint](results/defense/diagnostics/bootstrap-5200000-heads.json)
again reproduced all ten saved ensemble game records exactly. Head 0 was
slightly stronger than the ensemble: mean **10,208** versus **10,186**, best
**10,300** versus **10,220**. Heads 1–4 averaged **7,663**, **9,574**, **6,722**
and **7,915**, respectively. All **60** games still lost in stage 1, and even
the best individual head remained below the shared 10,480-point effort.
Thus this later checkpoint contains a modest single-head score advantage,
but no hidden stage reach or mission completion on these seeds. Weights and
configuration were unchanged; no production policy or global replay was
replaced, and no diagnostic actions entered training.

```bash
venv/bin/python -m rl.defense_head_probe \
  results/defense/training/dqn-24-bootstrap/step-000000800000/model.safetensors \
  --envs 4 --output runs/defense-bootstrap-heads-reproduction.json
```

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

Run 23 subsequently stopped cleanly at **14,174,976**, after **2,293,760**
new actions, **693** complete boot games, **449** completed restored segments
and **22** complete validation batches. None reached stage 2 or a mission.
Its strongest mean was
[10,464, median/best 10,480 at 13,486,848](results/defense/training/ppo-23-life-age/step-000013486848/evaluation.json);
the final validation mean was **9,984**, median **10,360**, best **10,460**.
The peak optimizer checkpoint, [full log](results/defense/training/ppo-23-life-age/metrics.jsonl)
and [final optimizer checkpoint](results/defense/training/ppo-23-life-age/final-checkpoint/state.json)
are preserved alongside its earlier verified best replay. Its depth plateau,
not a wall-clock budget, prompted reassigning the compute to a timing/history
experiment. The shared global replay remains unchanged.

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
