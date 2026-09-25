# Fine-cadence DQN age-frontier test (predeclared plan)

Plan amendment on 2026-09-25, while both fresh calibrations were below
100,000 actions and before their first fixed evaluation: the 1,048,576-action
limit is a stability/calibration gate, not an efficacy gate. The earlier
ordinary-DQN learning curve shows that using it to reject stage passage
would underfund the test. The matched efficacy budget and stop criteria below
were fixed before seeing any calibration evaluation result.

Physical-time interpretation added on 2026-09-25 before the efficacy
continuations' first fixed evaluation: 8,388,608 fine actions are an
intermediate matched budget,
not automatically enough to reject a policy still learning basic score play.
This clarification does not change the live trainer settings or any result.

## Why this is a test, not a continuation of the score plateau

The protected learned best scores 10,480 but repeatedly loses around decoded
stage-one course row 33–34 of 126. A separately quarantined forensic route
survived to row 65 at 50,000 T-states per action while scoring only 550.
Neither result is stage passage. The gap means that simply extending the
high-score PPO lineage is a poor test of the completion goal. The earlier
uniform age-reset PPO checks, screen-cell DQN comparison, and age-prioritized
random-hold search did not clear stage one either. This trial changes both
control cadence and where an otherwise ordinary Double DQN practices; its
specific frontier-vs-uniform comparison changes only reset selection.

## Invariants and arms

Both arms start from fresh, matched seed-97 scalar Double DQN weights and use
the original game, all 20 commands, four raw 16x64 video-memory frames,
50,000 T-states per action, observation stride 2, and displayed-score delta
as the only reward. Training may reset to unmodified opaque snapshots reached
by that arm's own play; original-boot evaluation never does. No forensic
route, native course pointer, hidden ship state, demonstrated action, score
bonus, or scripted play is available to the learner. Opaque own-life age is
used solely to retain/select training resets, never as policy input or reward.

Common training settings: eight workers, two reserved boot-only workers,
shared own-state archive, life-loss trigger with a 256-action same-life
lookback, 32 age bins of width 64 actions and four snapshots per bin,
0.75 reset probability, persistent balanced random controls with maximum
128-action hold and exponent 1.5, batch 64, one update per 16 aggregate
actions, 10-step return, and gamma
0.998498873 (approximately the same physical-time discount as 0.997 at
100,000 T-states). Epsilon decays to 0.5 over eight million actions.
The treatment samples uniformly from its four oldest retained own-life bins;
the control samples uniformly from every retained bin. The archive capacity
and admission rule are otherwise identical. Actual occupancies need not be.

## Gates, not an open-ended run

1. Native unit/integration tests and the full regression suite must pass.
2. Run a 65,536-action plumbing smoke **per arm**, using compact replay
   capacity 8192 and warmup 1024 to check the reset/learning/replay path
   quickly. Require finite updates, completed
   original-boot games, actual age-archive events and restored segments,
   original-boot verification of both replay bundles, and healthy disk
   headroom. A smoke score is not evidence of stage progress. If either arm
   fails plumbing, stop and repair; do not launch production.
3. If both pass, run a bounded **fresh** 1,048,576-action calibration per arm with
   compact replay capacity 200,000 and warmup 10,000. The smoke capacity
   holds only about two to three fine-cadence complete games; using it for
   the longer trial would erase nearly all older experience. The larger
   common capacity is predeclared before either pilot arm starts, and
   neither smoke model, optimizer nor experience initializes the pilot.
   Run fixed ten-game original-boot checks every 262,144 actions. This
   phase tests stable learning, real reset exposure, disk behavior and the
   paired configuration; it is **too short to test stage passage**. The
   earlier fresh ordinary DQN was still near a 354-point mean at one million
   100,000-T-state actions and did not pass 1,000 mean until 1.6 million;
   these fine actions simulate only half that physical time. Do not call
   a stage-one calibration outcome a negative efficacy result.
4. If both calibrations are stable and exercise their intended reset modes,
   continue both from their own terminal online/target/optimizer/RNG states
   to **8,388,608 total fine actions** each, retaining the same 200,000-slot
   replay and age-archive settings. A trainer restart necessarily refills
   transient replay and own-state archives; record that discontinuity, do
   not claim exact trajectory continuation. Fixed ten-game original-boot
   checks every 1,048,576 new actions use the same seeds in both arms.
   Stop early for non-finite learning, invalid provenance, repeated disk
   guard signal or another concrete safety failure—not a short-run score.
   Select by visible stage first, then fixed-check mean score. Preserve
   selected/terminal full optimizer states, every fixed result and independently
   verified self-contained replays. Then use two untouched matched 64-game
   seed sets per arm, **618200–618263** and **618400–618463**, and report
   mean, median, best, stage and missions. These ranges are disjoint from
   the smoke and fixed-check seeds and are fixed before efficacy results.
5. The 8,388,608-action fine-cadence budget simulates only as much game time
   as 4,194,304 of the old 100,000-T-state actions; the historical fresh
   DQN did not reach its high-score competence until about 6.1 million old
   actions. Accordingly, a low-score, stage-one result at this budget is
   **inconclusive about passage**, not proof that frontier resets fail.
   Decide a further matched budget from actual learning competence and
   independently replayed failure depth, not a short-run score gap. Do not
   extend merely for score gain after both arms are competent and still
   fail at the same early barrier. A private course-row read may be used only
   for post-hoc diagnostic interpretation, not learner inputs, reward,
   action selection, checkpoint ranking or replay promotion. If both arms
   reach and remain at the same barrier despite a sufficient matched budget,
   stop this mechanism and retain the negative result.

The old 10,480-point learned replay remains protected until a new eligible
policy exceeds it under the repository's stage-first promotion rule. The
gameplay goal still requires the original mission screen and verification;
no score threshold is a substitute.

## Completed plumbing gate (2026-09-25)

The full native regression suite passed **615 tests**. Both 65,536-action
smokes stopped normally with **4,028 finite updates**, unchanged original
game and screen-only/score-only configuration. Removing only run paths and
the frontier-selector fields from their recorded configs yields identical
SHA-256 `1d6db107f4f5fb84bdaa6ca375a005d4b24ac439a4a5d7739ba8404a929ce51d`.

| Arm | Boot games | Restored segments | Own-loss archive events | Fixed ten-game mean / best | Stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oldest four age bins | 14 | 13 | 80 | 332 / 340 | 0 |
| Uniform retained bins | 12 | 13 | 80 | 302 / 320 | 0 |

Both sampled-own-state mechanisms were exercised; the largest saved own-life
age was 792 fine actions. Both fixed checks completed ten original-boot
games. The best policy replay of each arm was independently reexecuted from
its frozen weights at the checkpoint's 50,000-T-state timing and stride 2.
The 30-point smoke-mean difference is **not** a gameplay improvement claim:
both policies are nearly untrained and every game stayed in stage one. The
smoke passes only the plumbing gate. Disk free space remained about 16 GiB.

The compact retained smoke record is the two [complete fixed evaluations](treatment-fixed-evaluation.json)
([control](control-fixed-evaluation.json)), [treatment metrics](treatment-metrics.jsonl)
and [control metrics](control-metrics.jsonl), plus independent frozen-model
[treatment](treatment-recheck/replay.html) and [control](control-recheck/replay.html)
replay bundles. The rechecks verify **3,339** and **3,146** neural actions,
respectively. Their directories and the fixed evaluations were byte-compared
with the stopped local outputs. The smoke optimizer snapshots are not pilot
parents and need not be retained after the archived record is pushed.

## Completed 1,048,576-action calibration (2026-09-25)

Both fresh arms stopped normally at exactly **1,048,576** fine actions and
**64,907** finite optimizer updates. Their exact-process disk guards logged
only `trainer_stopped`, never a low-space signal; free disk remained about
16 GiB. Four fixed ten-game checks per arm finished with no incomplete game.

| Total actions | Oldest-four mean / best | Uniform-bin mean / best | Later-stage games |
| ---: | ---: | ---: | ---: |
| 262,144 | 336 / 360 | 236 / 280 | 0 / 0 |
| 524,288 | 328 / 360 | 378 / 380 | 0 / 0 |
| 786,432 | 348 / 400 | 374 / 400 | 0 / 0 |
| 1,048,576 | 486 / 540 | 388 / 420 | 0 / 0 |

The score lead changed across checks, and **all 80 fixed games remained in
stage one**. This does not test passage of the row-33/34 barrier. The
treatment completed 226 boot games and 393 restored segments; the control
completed 191 and 301. Every completed restored segment's source action
matched an actual own-play archive event. **390/393** treatment sources
versus **161/301** control sources came from own-life age bin 9 or later;
early archive fill accounts for the few treatment lower-bin sources. This
verifies that the reset selectors supplied different practice states, not
that older states led to better from-boot play. No training episode showed
a later-stage event. The maximum saved own-life age was 922/923 fine
actions (treatment/control), an action count rather than a course pointer.

The terminal [treatment](calibration/treatment/terminal/state.json) and
[control](calibration/control/terminal/state.json) checkpoints each retain
online, target, optimizer and policy RNG state. Each `latest` copy was
byte-identical to its terminal step checkpoint for those four files. After
removing paths and the intended frontier-selector fields, the two recorded
configs had the same SHA-256
`ce15d8070aa91a5b2b0764bd0cfc0aa4fd6c48245e59dc7eb125ab8fce5e0445`.
All eight [fixed evaluations](calibration/treatment/fixed-checks/step-000001048576.json)
([control terminal](calibration/control/fixed-checks/step-000001048576.json)),
complete metrics, status and guard logs are retained in the two calibration
directories. All versioned local best-effort replay bundles are retained.

Separate frozen-weight [treatment](calibration/treatment/recheck/replay.html)
and [control](calibration/control/recheck/replay.html) rechecks replayed
**3,824** and **3,477** neural actions from original boot at the checkpoint's
50,000-T-state timing, confirming 540 and 420 points, both stage one.
Terminal checkpoints, replay-version trees, recheck bundles, metrics, guard
logs and each fixed evaluation were byte-compared with their stopped local
sources before archival. The protected 10,480-point best was not replaced.

This calibration passed the predeclared stability and reset-exposure gate.
The longer matched efficacy phase remains necessary before accepting or
rejecting the age-frontier mechanism for stage passage.

## Matched efficacy continuation (started 2026-09-25)

The [treatment terminal state](calibration/treatment/terminal/state.json)
(model SHA-256 `bcc831e166a84aa875eb8878184f38db7f9dcb2d3518ebc1867d79069032b3a4`)
and [control terminal state](calibration/control/terminal/state.json)
(model SHA-256 `e3266f97a302fcefd71de6a680b2e25d3b64e573693428ee2f3ffe4903e4440d`)
now initialize separate live efficacy runs under
`runs/defense-dqn-age-frontier-fine-178-{treatment,control}-efficacy`.
Each restores its own online, target, optimizer and RNG state at
1,048,576 actions / 64,907 updates, then targets 8,388,608 total actions.
The complete recorded configurations, excluding only run/artifact/resume
paths and the intended frontier-selector fields, have identical SHA-256
`9b4c76ceabc814c77ec248a7f06c34cc6a8f45b49c352eeebee84314f4214106`.
The fixed evaluation seeds remain 618100–618109, at every 1,048,576
additional actions. Both exact-process disk guards use an 8 GiB free-space
floor.

On resume, the in-memory prioritized replay and own-state archives start
empty and refill from new experience. This is an optimizer/RNG continuation,
not an exact trajectory continuation; no calibration replay, forensic
route or human action is loaded into training. The calibration archive is
already pushed. After byte comparison and normal process exit, about
164 MB of stopped local calibration copies were removed; selected/terminal
full states, fixed results, logs and verified replay versions are recoverable
from Git. Only unselected local optimizer snapshots were discarded.
