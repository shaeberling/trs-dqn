# Level-10 experiments

## Current status

The user has now chosen **beat all eight original levels**, without changing
the game. At September20 07:31UTC, **the all-eight target and fresh test are complete**.
The final selected checkpoint226,500,608 wins on seed10001, score585, with
one visible reserve and49,159 actions with zero inference mismatches. Watch the
[winning replay](results/level10/best-effort/replay.html). This is a reused
validation milestone, not a fresh success-rate estimate. The declared three
historical-checkpoint comparisons are complete. The third candidate was
disqualified by an incomplete secondary game; no seed was replaced.
The selected policy's **100/100 fresh games** completed naturally: mean
**278.75**, median **277.5**, best **583**, highest **8**, **zero verified wins**,
and **three unverified last-ball Level8 endings**. Winning capability is
proved by the replay; reliable winning is not. See [the final report](ALL_EIGHT.md).

The progress4 trial is now intentionally stopped at116,506,624 (+499,712)
to prioritize winning-policy verification. Its first validation was cancelled;
no complete suite or practice archive was produced. All48 learning suffixes
and nine full-game coverages reconcile, and its weights match the prior
progress16 trial at that pre-archive checkpoint. No improvement is attributed
to four-point archives. See the [completion audit](results/level10/all-eight-progress4-completion-review.json).
No learner, reassessment helper, supervisor or final-test worker remains live.
The final supervisor records `target_verified_and_tested`.

All70 selected116M baseline games reproduced every old per-game field, with
zero verified wins and ten unverified last-ball Level8 endings. Outcome
detection,142 tests and unchanged-learning compatibility checks pass.

At the previous September 19, 08:35 UTC review, training was paused for that decision.
The original game ends after eight levels; displayed Level 9/10 is unreachable
without changing the game. The [author's description](https://pski.net/breakdown-a-new-trs-80-game/)
and [exact local binary audit](results/level10/game-level-cap-audit.json) agree.
The original Level-10 goal was unreachable; its replacement is explicitly authorized.

The final selected reference is
**runs/level10-supervised-refkl01-v2/learner/step-226500608**:
**70/70 complete reused games**, mean **298.94**, median **280**, best **585**,
highest **8**, one verified win and ten unverified last-ball Level8 endings.
Counts for Levels2–8: **65 / 54 / 47 / 39 / 24 / 20 / 11**.
It outranks the first verified122.5M winner (one win, seven reaches8, mean285.96).
The prior116M reference had no verified win, ten reaches8 and mean315.47:
the new win-first selection is a completion milestone, not an improvement in
every score metric or proof of reliable winning. **Correction:** historical
statements of zero Level-8 clears in
11,644 practice segments measured transitions to displayed Level 9, which the
game never shows. They cannot establish the number of final-level wins.
Reaching Level8 is not winning; the newly verified replays provide real evidence.

Stable portable replays: [selected winner](results/level10/best/replay.html)
and [best winning effort](results/level10/best-effort/replay.html), currently
the same585-point game with all49,159 actions verified. The single-game record
does not determine which model is selected. Versioned bundles preserve exact
weights, optimizer, configuration, validation records, replay and inspection.

All **142 tests** pass. The old supervisor stopped intentionally at 236,003,328;
its final checkpoint and byte-identical log archive are retained. The subsequent
**runs/level10-supervised-progress16** experiment stopped cleanly at
**120,102,912** (+4,096,000 actions), after eight complete primary suites and
no model promotion. Its checkpoint and byte-identical log are retained too;
see the [completion audit](results/level10/progress16-completion-review.json).
That supervisor, learner and all 32 workers exited. Its unchanged
`status.json` now records `operator_stopped`, not live training.
No validation or replay trajectories entered training. The supervisor is local
Python, not a Codex/API agent. The subsequent all-eight progress4 run resumed
selected116M and is also now stopped for winning-policy verification, as noted
above. No game extension is authorized.
Final-test seeds **40000-40099 are now used**, exactly once for the frozen test.
They must not be presented as fresh in later experiments. The committed Level-5 package
and original public replay are unchanged.

## Earlier progress summaries (historical, not live status)

The following chronology predates discovery of the eight-level cap. References
to seeking Level 9/10, live processes, or zero final-level clears are historical
and superseded by the current status and measurement correction above.

The initial control, smaller-rate trial, and its extension have ended or been
paused with checkpoints preserved. The smaller-rate direction improved
validation consistency and reproduced level 5. Subsequent controlled trials
compared lower entropy (`runs/level10-entropy001`) and a longer reward horizon
(`runs/level10-gamma999`), from the same verified checkpoint. Lower entropy
completed its tranche and improved the primary suite to mean 132.6 with a
level-5 game. Its secondary audit finished 50/50 complete, mean 122.6, with
two level-5 games, one independently reproduced in a full replay. Its final
validation checkpoint started a longer continuation. That run was recovered
from its 22,253,568 checkpoint after an uncapped validation stalled on a serve;
`runs/level10-entropy001-long-recovered` continued with learning settings and
overall budget unchanged, then was paused cleanly at **27,598,848** after
seven consecutive complete validations missed level 5. Its checkpoints and
complete log are preserved. A separate finer-timing training comparison
(`runs/level10-timing50k`) also completed its budget without a complete level-5
validation, and its log/checkpoints are preserved.
The recovered run's 24,002,560 checkpoint set a new primary mean of 147.4,
with reach counts 2/3/4/5 of 17/7/5/1. Its secondary audit completed 50/50
games, mean 108.4 and one level-5 reach, below the earlier checkpoint's
secondary result. The primary improvement is not a confirmed general improvement.
The reward-horizon trial completed its
budget without improving on the source; its checkpoints and log are retained.
The self-generated start-state curriculum control (`runs/level10-curriculum`)
completed its two-million-action tranche at 22,503,424. None of its seven
complete validation suites reached level 5; one suite was disqualified by a
serve stall. Its best complete mean was 143.95, highest level 4. It produced
only eight completed practice segments starting at level 5, averaging 1.375
new points. Its checkpoints and full log are preserved.
The matched peer-shared curriculum (`runs/level10-curriculum-shared`) finished
cleanly at 22,503,424, with all eight validation suites complete. It distributed newly self-reached training entries, not
validation or replay states. Its **21,000,192** checkpoint completed 20/20
validation games, mean **130.3**, with **two level-5 reaches** (seeds 10005 and
10008). Its separate 50-game validation audit completed: mean **135.12**, median
121, best 304, reach counts 2/3/4/5 **39/15/8/2**. The source's corresponding
mean was 122.6 and counts 37/13/4/2. This improves the reused secondary suite's
mean and intermediate-level counts, not its highest level or level-5 count.
The 21,504,000 checkpoint also reached level 5 in one of 20 complete games,
mean 136.35. Subsequent progress is recorded below; this is not a new depth
milestone or evidence of reliable level-5 play.
The focused comparison (`runs/level10-curriculum-focused`) completed at
23,003,136. Its self-generated level-5 practice improved: first 100 segments
mean 1.04 new points, last 100 mean 10.22; all 2,110 segments still ended at
level 5. Seven validation suites were complete and one serve-stalled suite
was disqualified. The final complete mean was 131.35, highest level 4.
The longer continuation (`runs/level10-curriculum-focused-long`) was paused
cleanly at **28,045,312**, after **5,042,176** new actions. It reached **levels
6, 7, and 8 in restored training segments**, but complete-game validation
regressed; its latest nine suites reached only level 2. These are not from-boot
or frozen-policy depth milestones. The full log and checkpoints are preserved.
The protected-from-boot comparison `runs/level10-curriculum-balanced` was
paused cleanly at **28,815,360**, after **5,812,224** new actions. Reserving
16/32 workers improved some matched validations, but its last 19 complete
suites missed level 5 and its latest three reached only level 2. Logs and
checkpoints are preserved. Restored training reached level 8, not a full-game
depth milestone.
The matched 1e-5-learning-rate trial finished its initial tranche at
**25,006,080**. It completed **from-boot training games at levels 6 and 7**
(scores 334 and 411), but weights changed during those episodes. Its seven
complete primary suites still reached at most level 5; one suite was
disqualified. The final resumable learner now continues in
`runs/level10-curriculum-balanced-lr1e5-long`, with learning settings unchanged
and fresh training archives. Its first scheduled validation is 20/20 complete,
mean **133.3**, best **306**, highest level **5**.
The fixed five-checkpoint average did not improve depth: 20/20 complete,
mean **112.25**, highest level **4**. It is evaluation-only, not a resume source.
The otherwise matched GAE-lambda-0.99 trial completed at **30,007,296** after
**5,001,216** new actions, with all 20 primary suites complete. Four suites
reached level 5, versus eight for the matched original continuation; none
reached frozen level 6. It completed a from-boot training game at
**level 7 / score 381**, with changing weights. A
nearby frozen checkpoint completed all 50 reused secondary validation games:
mean **140.1**, three level-5 reaches. It is a promising candidate, not an
overall reference replacement: its primary suite missed level 5 and its
combined reused sets have fewer level-5 reaches than the retained reference.
The original continuation also completed another from-boot level-6 training
game and rebuilt restored practice through level 8. These are training results,
not new frozen-policy milestones.
All **83 tests** pass, including the optional self-imitation path and its
disabled/enabled/truncation smokes. The initial four-update self-imitation
comparison in **runs/level10-curriculum-balanced-sil** has completed at
**30,007,296**, from the same 25,006,080 source. All 20 primary suites are
complete. The completed audit verifies origins, score rewards, learning
suffixes and bounded buffers, including its own restored later-level starts.
The SIL **28,250,112** checkpoint established **frozen level 6** in both
reused validation sets: primary 20/20 complete, **130.25 / 112.5 / 326**,
one level-6 reach; secondary 50/50 complete, **146.4 / 124.5 / 337**, one
level-6 reach. Its [complete 326-point replay](results/level10/balanced-sil-28m25-primary-level6-replay.html)
matches every one of 15,034 seeded actions and the decoded final screen.
It was the previous **depth reference**, not reliable level-6 mastery: combined
70 reused games contain only two such reaches. The earlier third and tenth
SIL candidates did not improve on the previous reference. Restored training
still reaches 8 and is excluded from frozen performance.
The unchanged five-million-action continuation from that checkpoint,
**runs/level10-sil-level6-continuation**, has completed at **33,251,328**.
All 20 primary suites complete; the selected 31,252,480 checkpoint establishes
the Level-8 reference described below. Its full log and final learner are
preserved. The original long continuation
has now finished at **35,008,512** (+10,002,432 actions), all 40 primary suites
complete, highest frozen level 5. Its log and checkpoints are preserved.
The predeclared one-factor **20-versus-4 replay-update comparison has completed**
in `runs/level10-curriculum-balanced-sil20`, from the same 25,006,080 source,
at **30,007,296**. Nineteen primary suites are complete; the final suite is
disqualified by one incomplete game. Only two eligible checkpoints reach 5,
versus eight for the original SIL4 comparison. Its own restored training
reaches 8 but does not establish frozen performance. Do not adopt the higher
update dose; preserve its final learner, complete log and truncation accounting.
The original continuation's **28,504,064** checkpoint previously improved
both reused validation sets: primary mean **166.7**, two level-5 reaches; secondary mean
**141.18**, four level-5 reaches, all 50 games complete. Its complete **327-point**
replay reproduces the secondary game exactly. It was the previous **higher
combined-mean and level-5-consistency comparison**: 70-game mean 148.47 and
six level-5 reaches, versus the old Level-6 reference's 141.79 and four.

The continuation's **31,252,480** checkpoint is now the **verified Level-8
reference**. Primary 20/20 complete: **169.2 / 122.5 / 328 / level 5**,
six Level-5 reaches. Its supplemental consistency audit completes 50/50:
**182.66 / 148 / 555 / level 8**, counts 2-8 **42/27/18/10/4/2/1**.
The [full 555-point replay](results/level10/sil-continuation-31m25-secondary-level8-replay.html)
reproduces secondary seed 10128 uncapped, with all **22,439** neural actions
verified. Combined reused validation: **178.81 / 140 / 555**, counts
**57/36/26/16/4/2/1**. It improves depth, later-level counts and combined mean
over both earlier references, but not every metric: the older mean reference
has 60 Level-2 games versus 57. All older checkpoints/replays remain retained.
This is one Level-8 reach in 70 reused games, not reliable mastery or fresh-test
evidence. The unchanged **runs/level10-sil-level8-continuation** has completed
cleanly at **36,253,696**, after **5,001,216** new actions. Nineteen primary
suites are complete and one is disqualified; highest eligible primary level
is 7. None of its six secondary audits replaces the Level-8 source. Its final
primary is 20/20 complete, **132.65 / 131.5 / 263 / level 4**. The complete
log, final learner and both correctly discarded training guards are preserved.

The predeclared **runs/level10-sil-level8-cap128k** is now the **sole learner**,
from the same verified 31,252,480 source, with only replay capacity changed
from **32,768 to 131,072** entries. All other learning/validation settings stay
fixed; buffers restart empty. Source, worker, initial-update and suffix checks
pass. Its first update matches the completed control exactly apart from
allocation/timing telemetry, and its live buffer already exceeds the old
capacity. No performance improvement is presumed from startup. The old
f58-source capacity command remains superseded and unlaunched.
Its first primary completes 20/20 at mean 159.25, highest 6, three Level-5
reaches. The secondary audit completes 50/50 at mean 145.78, highest 7.
Combined mean 149.63 and seven Level-5 reaches still trail the retained Level-8
source; do not promote this early checkpoint. Continue the declared trial.
Its second primary is disqualified by one serve-stalled game (19 complete /
one incomplete); no secondary audit follows. Own restored training reaches
7 with source/reward accounting verified, not a new performance milestone.
Level 10 is **not yet reached**;
final-test seeds 40000-40099 remain unused. The sections below are the
chronological experiment record, including earlier in-progress decisions.

[Completed comparison chart](results/level10/training-curve.png) shows all five
archived comparison trials. It does not yet include the longer continuation
or its recovery stage, the finer-timing trial, or either curriculum. Stars highlight verified level-5 reaches; the depth panel's
dotted line marks the level-10 goal. Its JSON/SVG companions and complete logs
are in `results/level10`. Reproduce it with:

```bash
venv/bin/python -m rl.report \
  results/level10/logs/level10-continuation.jsonl \
  results/level10/logs/level10-lr5e5.jsonl \
  results/level10/logs/level10-lr5e5-extended.jsonl \
  results/level10/logs/level10-entropy001.jsonl \
  results/level10/logs/level10-gamma999.jsonl \
  --target-level 10 --highlight-level 5 \
  --baseline results/level5/random.json --output results/level10/training-curve
```

## Target and preserved starting point

Reach displayed level 10 (clear levels 1-9), using only the learned screen-only
policy and score rewards under [GOALS.md](GOALS.md). User requested this follow-up
after the level-5 milestone, committed as `f4c77d6`. No remote push was requested.

Starting checkpoint: `models/breakdown-level5`, 17,502,208 PPO actions plus
500,000 earlier DQN actions. Weights SHA-256:
`f84d6346798330742a1890e9ee6308fbc380c155bac9d24086dde10bf62d3b87`.
Its 20-game validation mean was 88, median 64, best 278, highest level 5
(1/20 reaches). Its now-published 100-game test mean was 83.32, median 72.5,
best 167, highest level 3 (0/100 level-5 reaches). This is not reliable mastery.
Preserve this package, its replay, and the original level-2 public replay.

## Protocol fixed before continuing

- Primary validation: seeds 10000-10019, sampled learned policy, same per-game
  action RNG seed offset of 1,000,000 and 100,000 T-states per action.
- Validation now runs to GAME OVER with `--eval-max-steps 0`; incomplete games
  never qualify for selection or the target. The longer protocol resets
  inherited selection records; re-evaluate the frozen starting checkpoint.
  Later recovery adds a 100,000-action training-validation guard after a
  confirmed serve stall. It still rejects every incomplete suite and resets
  protocol-specific selection records; final test and target replay stay uncapped.
- Secondary validation: seeds 10100-10149, used to compare promising candidates.
- Final test: **reserve seeds 40000-40099**, complete games, used only after
  freezing a validation-selected checkpoint. Do not train/select on these.
  Previous final-test seeds 20000-20049 and 30000-30099 are already published,
  not a new held-out test.
- First-reach success: at least one complete primary validation game reaches
  level 10. Separately report consistency at every reached level; do not call
  a rare validation success robust performance.
- Training safety guard: raise from 30,000 to 100,000 actions per episode to
  accommodate longer play. A guard hit is truncation, not a completed game;
  investigate any such hit. There is no wall-clock time limit.
- Policy input, reward, actions, architecture, and game physics are unchanged.
  Use the existing Apple-Silicon MLX backend and emulator.

## First continuation

Start with the successful level-5 settings: learning rate 0.0001, entropy
coefficient 0.003, 32 environments, rollout 128, batch 256, four PPO epochs,
gamma 0.995, GAE lambda 0.95, life-loss learning boundaries. Initial tranche:
five million additional actions, validation every 250,000 actions. Preserve
numbered checkpoints and separate best-by-score and best-by-level snapshots.
This tranche is a review point, not a stopping condition for the overall goal.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-continuation \
  --resume models/breakdown-level5 \
  --additional-steps 5000000 \
  --learning-rate 0.0001 --entropy 0.003 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000

venv/bin/python -m rl.status runs/level10-continuation

venv/bin/python -m rl.evaluate \
  models/breakdown-level5/model.safetensors \
  --games 20 --seed 10000 --max-steps 0 \
  --output results/level10/baseline-validation.json
```

Changes to later experiments will be recorded with their motivation, source
checkpoint, commands, and measured validation outcomes. If the baseline
continuation plateaus, compare one substantive training change at a time;
do not infer progress from score alone or from changing-policy training games.

## Status

Level-5 commit verified with 25 passing Python tests, original and level-5
replay checks, and all frozen-package checksums. Level 10 is not yet reached.

The continuation is running. Added two regression tests for the level-10
selection/termination rules; all 27 Python tests pass. The uncapped starting
validation completed 20/20 games and exactly reproduced every per-game record
from the frozen level-5 validation (mean 88, median 64, best 278, highest 5).
Record: [baseline validation](results/level10/baseline-validation.json).
The first 249,856 additional training actions completed without errors before
the first checkpoint evaluation at PPO action 17,752,064. Live metrics and
checkpoints are under `runs/level10-continuation`; they are not final results.

First checkpoint at 17,752,064: 20/20 complete validation games, mean 74.05,
median 62.5, best 123, highest level 2 (8/20 reaches). This is below the frozen
starting model, not an improvement. Continue the planned tranche to measure
the trend; keep the starting model as the preserved comparison and fallback.

Secondary baseline completed on seeds 10100-10149: 50/50 complete, mean 89.3,
median 94, best 190, highest level 3; level 2 in 30/50 and level 3 in 2/50,
no level-4 or level-5 reaches. This is a validation comparison, not a fresh
final test. Record: [secondary baseline](results/level10/baseline-secondary-validation.json).

| PPO actions | Complete validation | Mean | Median | Best | Highest level | Level-2 reaches | Level-3 reaches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17,752,064 | 20/20 | 74.05 | 62.5 | 123 | 2 | 8 | 0 |
| 18,001,920 | 20/20 | 76.45 | 60.5 | 177 | 3 | 6 | 1 |
| 18,251,776 | 20/20 | 69.50 | 55 | 245 | 4 | 4 | 1 |
| 18,501,632 | 20/20 | 72.85 | 60.5 | 129 | 2 | 7 | 0 |
| 18,751,488 | 20/20 | 71.60 | 58 | 128 | 2 | 6 | 0 |
| 19,001,344 | 20/20 | 94.65 | 96.5 | 147 | 3 | 12 | 1 |
| 19,251,200 | 20/20 | 65.45 | 60.5 | 109 | 2 | 5 | 0 |
| 19,501,056 | 20/20 | 68.20 | 60.5 | 121 | 2 | 6 | 0 |

None of these continuation checkpoints has surpassed the preserved level-5 model.
A changing-policy training game reached level 4, score 244; this is diagnostic
evidence, not a frozen-model validation result. No training guard hits or
errors have occurred through approximately 700,000 additional actions.

## Smaller-learning-rate comparison

After three continuation evaluations, all primary means remained below the
starting model (74.05, 76.45, 69.5 versus 88), and level-2 reach counts were
8, 6, and 4 versus 10. One checkpoint reached level 4. This small sample does
not establish a permanent plateau, but motivates testing gentler updates to
retain consistent play while the unchanged control continues.

Fork the **same frozen level-5 source**, changing only the learning rate from
0.0001 to 0.00005. Use the same initial policy, optimizer state, random seed,
validation suite, episode guard, and entropy coefficient. Compare checkpoints
at matched action counters. The trial's initial tranche is two million
additional actions; the control retains its five-million-action tranche.
Both learners use separate output directories. This is not a new architecture,
reward, input, or gameplay heuristic.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-lr5e5 \
  --resume models/breakdown-level5 \
  --additional-steps 2000000 \
  --learning-rate 0.00005 --entropy 0.003 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000
```

First matched comparison at 17,752,064 PPO actions (249,856 additional): both
20-game suites completed. The smaller-rate trial had mean 83.3, median 81,
best 120, highest level 2, with 10/20 level-2 reaches. The control had mean
74.05, median 62.5, best 123, highest level 2, with 8/20 reaches. This is an
early consistency improvement over the matched control, not evidence of
level-5 retention or level-10 success. Continue both before choosing a branch.

Second matched comparison at 18,001,920 PPO actions (499,712 additional):
the smaller-rate trial completed 20/20 games with mean **96.25**, median 97.5,
best 239, highest level 4, level 2 in 11/20, and levels 3 and 4 in 1/20.
The matched control had mean 76.45, best 177, highest level 3, level 2 in
6/20, and level 3 in 1/20. The trial improves the mean over the starting
model's primary mean of 88, but has not reproduced its level-5 reach.

This checkpoint is being evaluated on the predeclared secondary validation
suite, not the held-out final test:

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-lr5e5/step-018001920/model.safetensors \
  --games 50 --seed 10100 --max-steps 0 \
  --output results/level10/validation-audit-lr5e5-18m.json
```

The audit completed **50/50 games**: mean **103.98**, median 105, best 221,
highest level 4. Level-2 reaches: **41/50**, level 3: **4/50**, level 4:
**1/50**, level 5: 0/50. Against the frozen starting model on exactly the
same seeds, this improves mean (89.3 to 103.98), level-2 count (30 to 41),
level-3 count (2 to 4), and highest level (3 to 4). This is a secondary
validation improvement, not a final-test result or level-10 success.
Weights SHA-256:
`76ef00e5bd6892e57a81fbab0f7b0f501a19954763cb946ecfb2e91f8aa85456`.

The next smaller-rate checkpoint, 18,251,776, completed its primary suite:
mean **109.75**, median 121.5, best 180, highest level 3; level 2 in **15/20**
and level 3 in **2/20**. Its lower maximum level means it does not replace
the trial's best-by-level checkpoint at 18,001,920.

These results justify prioritizing the smaller-rate direction. Continue the
control through roughly two million additional actions for comparison, then
consider pausing it to give the promising continuation more compute. The
original five-million control tranche may be shortened on measured evidence;
this does not change the level-10 objective.

The control was subsequently paused cleanly after the two-million-action
review. Its last validation (19,501,056) had mean 68.2, median 60.5, best 121,
highest level 2, and 6/20 level-2 reaches. The trainer saved
`runs/level10-continuation/latest` at **19,582,976** actions, then exited 0
with `target_met: false`. Actual additional training: **2,080,768** actions.
No checkpoints were deleted. The complete log is archived at
`results/level10/logs/level10-continuation.jsonl` and was compared byte-for-byte
with the stopped run's log. The smaller-rate trial remains active.

### Level 5 reproduced with stronger consistency

At **18,501,632** PPO actions (999,424 additional), the smaller-rate trial
completed 20/20 validation games: mean **125.35**, median **117**, best **282**,
highest level **5**. Reach counts: level 2 **16/20**, level 3 **4/20**,
level 4 **2/20**, level 5 **1/20**. The level-5 game is seed **10011**,
14,525 actions; seed 10013 reached level 4 with score 258. This checkpoint
improves both the mean and the level-ranking tuple over the frozen starting
model, but has not reached level 10.

Weights SHA-256:
`c656c9e515a007360f9bfa7bb549594a50d7d60dcaf7230b94defef5d240a3d5`.
The checkpoint is undergoing the same secondary validation and an independent
replay recording; neither uses the reserved final-test seeds.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-lr5e5/step-018501632/model.safetensors \
  --games 50 --seed 10100 --max-steps 0 \
  --output results/level10/validation-audit-lr5e5-18m5.json

venv/bin/python -m rl.record \
  runs/level10-lr5e5/step-018501632/model.safetensors \
  --seed 10011 --max-steps 0 \
  --output results/level10/level-5-validation-replay.html
```

The independent recording completed successfully and exactly matched the
selected validation game's score, level, action count, start phase, and model
hash. [Watch the 282-point validation replay](results/level10/level-5-validation-replay.html).
It contains 14,526 frames through GAME OVER. The replay codec/control tests
and metadata-to-evaluation cross-check pass. This is a validation showcase,
not an independent final-test game; the original level-5 replay is unchanged.
The matching [primary validation record](results/level10/validation-lr5e5-18m5.json)
is archived alongside it so the replay check does not require an ignored run
directory. Replay SHA-256:
`323aa07345132f45b158ca27169c29bcf713ff8cb35aaa4479ecfa0d6c764e9b`.

The [secondary audit](results/level10/validation-audit-lr5e5-18m5.json) finished
50/50 complete: mean **105.62**, median **107**, best **252**, highest level
**4**. Reach counts: level 2 **33/50**, level 3 **8/50**, level 4 **1/50**,
level 5 **0/50**. This improves mean and later-level frequency over the
starting model (89.3 mean, 30/50 level 2, 2/50 level 3, highest level 3).
Against the preceding audited smaller-rate checkpoint, level-2 frequency
decreased from 41 to 33 while level-3 frequency increased from 4 to 8;
progress is not uniform at every level. Level 5 remains a rare primary
validation result, not dependable mastery. The final-test seeds remain unused.

At the next primary checkpoint, **18,751,488**, all 20 games completed:
mean **117.8**, median **122.5**, best **233**, highest level **4**;
level 2 in **17/20**, level 3 in **3/20**, level 4 in **1/20**. Keep the
18,501,632 best-by-level checkpoint and continue the trial toward its review
point at two million additional actions.

Later primary checkpoints remain substantially more consistent than the
matched control, without extending the verified maximum beyond level 5:

| PPO actions | Complete | Mean | Median | Best | Highest level | Level-2 reaches | Level-3 reaches | Level-4 reaches | Level-5 reaches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19,001,344 | 20/20 | 127.80 | 120 | 258 | 4 | 16 | 6 | 1 | 0 |
| 19,251,200 | 20/20 | 124.45 | 123.5 | 210 | 3 | 18 | 4 | 0 | 0 |
| 19,501,056 | 20/20 | 127.50 | 124.5 | 252 | 4 | 17 | 5 | 1 | 0 |

A completed changing-policy training game also reached level 5 with score
299. This provides later-state training experience, but is not a frozen-policy
validation result. The target remains displayed level 10.

## Smaller-rate extension

The two-million-action trial ended normally at **19,505,152** PPO actions
(**2,002,944** additional actions after rollout rounding), with exit code 0
and `target_met: false`. Its complete log is archived as
`results/level10/logs/level10-lr5e5.jsonl`, byte-identical to the stopped source.
The final validation did not surpass the best-by-level checkpoint.

Selected source: **18,501,632**, SHA-256
`c656c9e515a007360f9bfa7bb549594a50d7d60dcaf7230b94defef5d240a3d5`.
It has a complete level-5 validation game, stronger primary level counts than
the original milestone, a verified replay, and the 50-game secondary audit
documented above. Retain the newer best-mean checkpoint and every numbered
checkpoint too; do not equate the final saved model with the best model.

Continue from the selected checkpoint for five million additional actions,
retaining learning rate 0.00005, entropy 0.003, and the complete-game validation
protocol. Resume starts fresh training environments, not a bit-for-bit replay
of the previous post-checkpoint training trajectory. The action budget is a
review tranche, not an overall deadline or a replacement for the level-10 goal.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-lr5e5-extended \
  --resume runs/level10-lr5e5/step-018501632 \
  --additional-steps 5000000 \
  --learning-rate 0.00005 --entropy 0.003 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000

venv/bin/python -m rl.status runs/level10-lr5e5-extended
```

The extension started successfully with the expected source hash and Metal
device, at roughly 1,670 training actions per second before validation time.
Its first checkpoint, **18,751,488** (249,856 new actions), completed all 20
validation games: mean **86.55**, median **74**, best **158**, highest level
**3**, level 2 in **10/20** and level 3 in **1/20**. This is below the source
checkpoint, not an improvement. Preserve the source and continue the planned
tranche to measure whether the early dip recovers; no new target is claimed.

At **19,001,344** (499,712 extension actions), validation recovered to mean
**118.25**, median **118.5**, best **254**, highest level **4**; all 20 games
completed, with level 2 in **17/20**, level 3 in **3/20**, and level 4 in
**1/20**. Continue unchanged; this is recovery from the early dip, not a new
best-by-level checkpoint or a level-10 result.

Later extension checkpoints again fell below the source:

| PPO actions | Complete | Mean | Median | Best | Highest level | Level-2 reaches |
|---:|---:|---:|---:|---:|---:|---:|
| 19,251,200 | 20/20 | 84.80 | 74 | 135 | 2 | 11 |
| 19,501,056 | 20/20 | 71.85 | 63.5 | 118 | 2 | 5 |
| 19,750,912 | 20/20 | 86.15 | 97 | 126 | 2 | 12 |
| 20,000,768 | 20/20 | 107.90 | 118 | 199 | 3 | 16 |
| 20,250,624 | 20/20 | 94.60 | 93.5 | 196 | 3 | 12 |
| 20,500,480 | 20/20 | 101.65 | 113.5 | 182 | 3 | 14 |

After eight scheduled validations, none exceeded the source's level-5 rank
or mean 125.35. The extension was paused cleanly at **20,680,704** actions
(**2,179,072** additional), exit code 0 and `target_met: false`. This shortens
its planned five-million tranche on measured evidence, not a wall-clock
limit. Complete log: `results/level10/logs/level10-lr5e5-extended.jsonl`,
byte-identical to the stopped source. No checkpoints were deleted.
The final saved model's training metrics improved shortly before the pause,
so it is also receiving an out-of-band 20-game primary validation check.
Final model SHA-256:
`51aea414401c15de0c36b2860d0365c79251d71d1935a69833197eb233583df6`.

The [paused-checkpoint validation](results/level10/control-paused-validation.json)
completed 20/20 games: mean **108.2**, median **103.5**, best **278**, highest
level **4**, with level 2 in **13/20** and levels 3 and 4 in **2/20** each.
This is better than its last scheduled checkpoint on depth, but does not
replace the preserved 18,501,632 level-5 model. It remains available.

### Restore check and action-probability diagnostic

Investigated the resumed run's regression before attributing it to a new
algorithmic problem. A new test performs several PPO updates, saves/restores
weights and Adam state through the trainer's resume path, then verifies that
further identical updates match uninterrupted training (parameters and
optimizer state within rtol 1e-6, atol 1e-7). It passes. The extension's
learning/validation settings also match its source; differences are run,
source checkpoint, and budget. This rules out the tested restore mismatch,
not all causes of training variance. Resumed training environments are fresh.

Added `rl.inspect_policy` to inspect probabilities on existing recorded screen
frames, without learning or running new games. It decodes the replay without
executing JavaScript, rebuilds the four-frame policy observations, checks the
model hash and visible outcome, and reproduces every seeded recorded action
before reporting metrics. Codec, stack alignment, and probability aggregation
tests were added; **all 33 Python tests pass**.

```bash
venv/bin/python -m rl.inspect_policy \
  runs/level10-lr5e5/step-018501632/model.safetensors \
  results/level10/level-5-validation-replay.html \
  --output results/level10/level5-policy-inspection.json
```

[Inspection result](results/level10/level5-policy-inspection.json): all
**14,525** actions reproduced exactly with batch size 128. Mean joint action
entropy was **0.4132 nats**, direction entropy **0.2670**, and serve entropy
**0.2297**. The mean probability of sampling a direction other than the
most likely direction was **11.09%**; the recorded fraction was **11.13%**.
These are not error rates, and the marginal entropies need not add to the
joint entropy because direction and serve are correlated. This is one
selected successful trajectory, not a representative performance estimate.

## Lower-entropy comparison

The direction diagnostic and the weaker resumed validation motivate testing
less entropy regularization. Fork the **same 18,501,632 checkpoint** as the
ongoing extension, changing only the entropy coefficient **0.003 -> 0.001**.
Keep learning rate 0.00005, screen observations, score rewards, all six
network-selected actions, sampling policy, training seed, and validation
protocol unchanged. No argmax override, scripted serving, demonstrations,
or validation-data training is introduced. The inference-only inspection
does not update any weights.

The hypothesis is that weaker pressure for randomness may improve sustained
play; reduced exploration could also hurt. Run a controlled two-million-action
trial while retaining the current extension as a comparison. Match checkpoints
by additional actions from their identical source. Reserve the final-test
seeds unchanged.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-entropy001 \
  --resume runs/level10-lr5e5/step-018501632 \
  --additional-steps 2000000 \
  --learning-rate 0.00005 --entropy 0.001 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000
```

Initial lower-entropy validations are mixed, not a demonstrated winner:

| PPO actions | Complete | Mean | Median | Best | Highest level | Level-2 reaches | Level-3 reaches | Level-4 reaches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18,751,488 | 20/20 | 83.15 | 65.5 | 135 | 2 | 9 | 0 | 0 |
| 19,001,344 | 20/20 | 109.05 | 95 | 286 | 4 | 11 | 4 | 2 |
| 19,251,200 | 20/20 | 96.65 | 93 | 277 | 4 | 11 | 1 | 1 |
| 19,501,056 | 20/20 | 95.05 | 111 | 134 | 2 | 12 | 0 | 0 |
| 19,750,912 | 20/20 | 113.4 | 114 | 196 | 3 | 15 | 3 | 0 |
| 20,000,768 | 20/20 | 126.75 | 120 | 276 | 4 | 14 | 6 | 3 |
| 20,250,624 | 20/20 | 119.4 | 107 | 256 | 4 | 16 | 6 | 1 |
| 20,500,480 | 20/20 | 132.6 | 124.5 | 274 | 5 | 16 | 5 | 2 |

At the second checkpoint the matched control had mean 118.25, 17 level-2
reaches, three level-3 reaches, and one level-4 reach. Lower entropy therefore
traded lower mean/early-level consistency for more later-level reaches in
this small suite. Both comparisons use complete games, not the final test.

At 20,000,768 the lower-entropy trial exceeds its matched extension control
(mean 107.9, two level-3 reaches, no level-4 reaches) in mean and later-level
counts, but still has no frozen level-5 validation game. Subsequent training
has reached level 5 with 296 points; changing-policy training episodes are
not frozen-policy validation evidence. Continue the planned tranche.

The final scheduled validation at **20,500,480** did reproduce level 5:
**274 points**, seed **10008**, **15,958 actions**, full GAME OVER. It improved
the previous best primary mean (125.35 -> 132.6) and level-3 count (4 -> 5),
with unchanged level-2/4/5 counts (16/2/1). This is a modest improvement on a
reused 20-game validation suite, not proof of generalization or level-10 play.
The run finished its budget cleanly at **20,504,576**, adding **2,002,944**
actions. Its complete log is archived; the final extra rollout is preserved
but has no scheduled validation. Continue from the **20,500,480 validated
checkpoint**, not the unvalidated final rollout or a much earlier snapshot.
Record: [primary validation](results/level10/validation-entropy001-20m5.json).

## Longer lower-entropy continuation

Source: `runs/level10-entropy001/step-020500480`, model SHA-256
`510ce9632f744301f3a0f2b9c100b4c5c070afa88f1dbc466324c7c411d9fb70`.
Keep learning rate 0.00005, entropy 0.001, gamma 0.995, architecture, inputs,
reward, action duration, PPO updates, and life boundaries unchanged. Give the
improving direction **ten million additional actions** as the next review
tranche, preserving every 250,000-action validation snapshot. This is not a
wall-clock limit or completion condition for the overall level-10 goal.
Use the verified 20-worker evaluator below, which resets inherited validation
selection records because the numerical protocol is recorded separately.
Keep the previous level-5 checkpoints as external comparisons and fallbacks.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-entropy001-long \
  --resume runs/level10-entropy001/step-020500480 \
  --additional-steps 10000000 \
  --learning-rate 0.00005 --entropy 0.001 --gamma 0.995 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000

venv/bin/python -m rl.evaluate \
  runs/level10-entropy001/step-020500480/model.safetensors \
  --games 50 --seed 10100 --max-steps 0 --envs 20 \
  --output results/level10/validation-audit-entropy001-20m5.json
```

The secondary audit uses designated validation seeds, not the untouched
40000-40099 final test. The saved candidate is fixed during evaluation;
the continuation updates only its own in-memory copy and new run directory.

Secondary audit completed **50/50 full games**, mean **122.6**, median **115**,
best **283**, highest level **5**. Level-2/3/4/5 reach counts were
**37/13/4/2**. The previous smaller-rate source's corresponding serial audit
was mean 105.62, median 107, best 252, highest 4, with counts 33/8/1/0.
The new audit used 20 evaluation workers; numerical parallelism is recorded
explicitly, and these are reused validation comparisons, not fresh test results.
Both new level-5 games completed: seed 10124 scored 282 in 12,037 actions;
seed 10145 scored 283 in 13,390 actions. The latter independently reproduced
with serial inference: **13,391 replay frames**, identical score, level,
action count, start phase, and terminal fields. This checks that the observed
secondary level-5 success is not dependent on parallel batch size.
Replay: [283-point secondary-validation game](results/level10/entropy001-secondary-replay.html).
Record: [secondary validation](results/level10/validation-audit-entropy001-20m5.json).

Long-run primary validation record: every suite below completed **20/20 full
games**, with 20 workers and no action cap.

| PPO actions | Mean | Median | Best | Highest level | Reach counts (2/3/4/5) |
|---:|---:|---:|---:|---:|---|
| 20,750,336 | 122.65 | 116.5 | 269 | 4 | 16/3/1/0 |
| 21,000,192 | 129.8 | 120 | 289 | 5 | 19/4/1/1 |
| 21,250,048 | 104.7 | 114 | 134 | 2 | 16/0/0/0 |
| 21,504,000 | 111.35 | 118.5 | 256 | 4 | 15/2/1/0 |
| 21,753,856 | 122.85 | 114.5 | 273 | 5 | 16/3/2/1 |
| 22,003,712 | 134.4 | 117 | 284 | 4 | 15/6/4/0 |

At 21,000,192, seed 10015 scored 289 in 15,531 actions. That checkpoint
improved level-2 consistency in this primary suite, but had fewer level-3/4
reaches and a lower mean than the source. The first million additional actions
therefore reproduced level 5 without surpassing the stronger source checkpoint.
Keep that source and continue the planned longer tranche unchanged; no fresh
final-test seeds have been used.

At 21,753,856, seed 10010 completed a game that reached level 5, with 271
points in 12,490 actions. The highest score in that suite was 273, reinforcing
why displayed level is tracked separately from score. This checkpoint repeats level 5 but
still has fewer level-3 reaches than the source; it does not advance the target.

At 22,003,712, mean score rose to 134.4 and level-4 reach count to 4/20,
exceeding the source's mean and level-3/4 counts, but none reached level 5.
This is a useful consistency gain at intermediate levels, not a new depth
record or a reason to discard the stronger level-5-capable checkpoints.

Through the 21,504,000 checkpoint, parallel validation consumed **75.97 of
788.03 logged seconds (9.64%)**. This is an observed run-time share, not a
controlled speed comparison; machine load changed as the other jobs finished.
The long continuation's recovery stage remains active; all five comparison
runs have ended or been paused with their logs archived. The original long
stage's log is also archived, with its external interruption documented below.

## Finer action-timing diagnostic

While the long continuation runs unchanged, evaluate the fixed 20,500,480
checkpoint with **50,000 instead of 100,000 T-states per action**, on primary
validation seeds 10000-10019. The hypothesis is that more frequent neural
decisions could help sustained control. This also changes the spacing of the
four observed frames, so transfer could worsen; it is not a pure latency test
and no improvement is assumed in advance.

No model weights are changed. The same sampled network chooses every action;
there is no steering/serve script, demonstration, new reward, or hidden input.
Play full games without an action cap. Treat this as a separate action-timing
protocol, not a like-for-like continuation checkpoint comparison. The current
learner and frozen packages keep their original timing. Final-test seeds remain
unused.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-entropy001/step-020500480/model.safetensors \
  --games 20 --seed 10000 --max-steps 0 --envs 20 --tstates 50000 \
  --output results/level10/action-duration-50k-validation.json
```

The diagnostic completed **20/20 full games**: mean **101.95**, median
**114.5**, best **202**, highest level **3**, with reach counts 2/3/4/5 of
**14/1/0/0**. This is below the checkpoint's original-timing result (mean
132.6, highest 5). Merely doubling decision frequency did not improve this
fixed policy. That does not establish how a policy trained at the finer timing
would perform, but it gives no reason to change the active learner's timing.
Record: [50,000-T-state diagnostic](results/level10/action-duration-50k-validation.json).

## Argmax transfer diagnostic at the newer checkpoint

The earlier 10,002,432-action model's argmax diagnostic stalled on serves in
two of five games (see LEVEL5.md). The newer 20,500,480-action model has learned
substantially more, so recheck whether deterministic neural action selection
helps control. This is a separate decoding-mode diagnostic, not a change to
the sampled-policy learner, its validation protocol, or the frozen packages.
Every action is still selected by the trained network, with no gameplay rule
or scripted serve intervention.

Use primary seeds 10000-10019, original 100,000-T-state timing, 20 workers,
and a **50,000-action diagnostic cap** because argmax can remain stuck in a
serve state. An incomplete suite cannot qualify for performance selection or
the target, even if a subset of games scores well. A promising complete result
would require further complete-game verification before adopting this mode.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-entropy001/step-020500480/model.safetensors \
  --games 20 --seed 10000 --max-steps 50000 --envs 20 --deterministic \
  --output results/level10/argmax-20m5-diagnostic.json
```

The newer model also failed this mode: **only 8/20 games completed**, while
**12/20 hit the 50,000-action cap**. Of those, **11 were waiting for a serve**;
seed 10007 was still in play at level 3, score 197. The evaluator exited
unsuccessfully as intended. This is not a complete-suite performance estimate
and cannot qualify for selection or the target. Retain sampled decoding.
Record: [argmax diagnostic](results/level10/argmax-20m5-diagnostic.json).

## Investigating a long sampled validation

The long learner remains live but its 22,253,568-action validation is taking
much longer than previous suites. Its saved model is available and the process
is still consuming CPU; a quiet log alone does not establish termination or a
failure. Re-evaluate that fixed checkpoint on the same 20 seeds and 20 workers
with a 100,000-action **diagnostic cap** to identify unfinished states. The
live learner remains unchanged, its original suite is still uncapped, and any
truncated diagnostic suite cannot select a model or satisfy the goal.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-entropy001-long/step-022253568/model.safetensors \
  --games 20 --seed 10000 --max-steps 100000 --envs 20 \
  --output results/level10/long-validation-22m25-diagnostic.json
```

The capped diagnostic finished **19/20 games**; seed **10010** reached displayed
level **5**, score **292**, then remained on **PASS BALL** at 100,000 actions.
It is unfinished and does not count as a level-5 success. A separate 25,000-action
diagnostic recording reproduced the same state. Its final **11,066 frames were
identical**. On the last 10,000 actions, the network chose LEFT **54** times and
LEFT+SPACE **9,946** times, with no other action. Inspection of the frozen
network showed almost all probability in that left-holding pair; the effective
probability of SPACE alone was about **4.66e-10**. No probe actions, serve
intervention, or training on the diagnostic recording were used.
See [incomplete diagnostic](results/level10/long-validation-22m25-diagnostic.json)
and [policy inspection](results/level10/serve-stall-policy-inspection.json).

Added observational evaluation progress every ten seconds (pending seeds,
steps, visible scores/levels, and waiting status), persisted in training JSONL
and exposed by `rl.status`. Added cancellation checks inside evaluation so a
future SIGTERM request saves and exits without selecting partial results.
The first **42 tests passed**; a real four-action smoke process also verified
SIGTERM during validation: checkpoint saved, workers closed, no evaluation
result or target success claimed. The status-display regression adds one test.

The old process had loaded the pre-fix module, so its SIGTERM request could not
interrupt the stuck evaluation. After verifying the already saved model,
optimizer, and RNG-state hashes, terminated only its exact PID **9440**;
the process exited **137**. Only the unfinished evaluation and unsaved emulator
trajectories were discarded. No training update had occurred after the saved
22,253,568 checkpoint. The raw log is archived unchanged, ending at
`validation_start`; the external stop and checksums are recorded separately in
[recovery record](results/level10/long-run-recovery.json).

Recovered training from that checkpoint with the **same learning settings**
and original absolute budget endpoint **30,500,480** (8,246,912 requested
actions remaining at resume). Use a **100,000-action validation guard** so a
failed policy cannot indefinitely prevent further learning. Any truncated suite
is entirely disqualified from checkpoint selection and target success; this is
not a wall-clock limit or a shortcut to declaring a complete game. Changed
validation protocol resets inherited selection records. Final evaluation and
target replay must still run uncapped to actual GAME OVER.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-entropy001-long-recovered \
  --resume runs/level10-entropy001-long/step-022253568 \
  --steps 30500480 --eval-max-steps 100000 --eval-envs 20 \
  --target-level 10 --target-clears 1
```

The recovered process started successfully and advanced beyond the checkpoint.
The original level-2 and level-5 packages remain untouched; final-test seeds
40000-40099 are still unused.

All **43 Python tests pass**, including current-suite progress visibility.
Recovered validation record: each suite below finished **20/20 games**, with
the new 100,000-action guard unused (zero truncations).

| PPO actions | Mean | Median | Best | Highest level | Reach counts (2/3/4/5) |
|---:|---:|---:|---:|---:|---|
| 22,503,424 | 137.95 | 121.5 | 283 | 4 | 15/6/4/0 |
| 22,753,280 | 127.05 | 125.5 | 247 | 4 | 16/7/1/0 |
| 23,003,136 | 118.65 | 120 | 216 | 4 | 16/4/1/0 |
| 23,252,992 | 126.1 | 116.5 | 273 | 4 | 14/4/4/0 |
| 23,502,848 | 114.25 | 120.5 | 249 | 4 | 15/4/1/0 |
| 23,752,704 | 113.25 | 118 | 224 | 4 | 15/3/1/0 |
| 24,002,560 | 147.4 | 120.5 | 292 | 5 | 17/7/5/1 |
| 24,252,416 | 138.9 | 125.5 | 282 | 4 | 16/6/3/0 |
| 24,502,272 | 137.55 | 131.5 | 293 | 4 | 14/9/3/0 |
| 24,752,128 | 145.85 | 126.5 | 296 | 4 | 17/9/4/0 |
| 25,001,984 | 110.45 | 109 | 203 | 3 | 17/3/0/0 |
| 25,251,840 | 127.1 | 123 | 274 | 5 | 18/3/1/1 |
| 25,501,696 | 127.85 | 118 | 276 | 5 | 15/5/3/1 |
| 25,751,552 | 131.7 | 121.5 | 303 | 5 | 16/7/2/1 |
| 26,001,408 | 115.7 | 122.5 | 262 | 4 | 15/2/1/0 |
| 26,251,264 | 113.05 | 106.5 | 291 | 4 | 12/3/3/0 |
| 26,501,120 | 109.8 | 119 | 242 | 4 | 14/2/1/0 |
| 26,750,976 | 133.85 | 126 | 236 | 4 | 17/8/1/0 |
| 27,000,832 | 116.35 | 110.5 | 274 | 4 | 13/5/1/0 |
| 27,250,688 | 95.25 | 64.5 | 201 | 3 | 10/4/0/0 |
| 27,500,544 | 91.95 | 95.5 | 138 | 2 | 12/0/0/0 |

Ten-second active-game snapshots are being written to the training log as
intended. These are intermediate-level results, not a new depth record.
Checkpoint hashes and configuration comparison confirm that recovery changed
no learning hyperparameters; it continues the remaining original budget.

The **24,002,560** checkpoint improved both primary mean and the deeper-level
selection rank over the 20,500,480 source: level-2/3/4 counts rose from 16/5/2
to 17/7/5, with one level-5 game in each. Median score was slightly lower,
120.5 versus 124.5. Seed **10017** scored **292** and reached level **5** in
**14,024 actions**, ending at actual GAME OVER. This is a new primary-validation
best, not yet evidence of improvement on the separate secondary set.
Model SHA-256: `a849d739959e970ff376c15e74670b750df3c0539a064de727e5e40decfb31d9`.
Record: [primary validation](results/level10/validation-recovered-24m.json).

Audit this fixed snapshot on seeds 10100-10149 with the 100,000-action guard;
if any game is incomplete, the suite cannot support selection or a complete-game
performance claim. The two learners continue independently and do not update
the audited snapshot.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-entropy001-long-recovered/step-024002560/model.safetensors \
  --games 50 --seed 10100 --max-steps 100000 --envs 20 \
  --output results/level10/validation-audit-recovered-24m.json
```

The audit completed **50/50 games without truncation**: mean **108.4**,
median **101.5**, best **285**, highest level **5**, reach counts 2/3/4/5 of
**33/8/4/1**. Seed 10100 scored 285 in 16,030 actions. This did **not** confirm
the primary-suite gain: the earlier 20,500,480 source's secondary result was
mean 122.6, median 115, best 283, and counts 37/13/4/2. Retain both snapshots;
the newer model is best on primary depth rank/mean, while the earlier model
remains stronger on this secondary suite. Neither has reached level 6 or 10.
Record: [secondary audit](results/level10/validation-audit-recovered-24m.json).

After the 27,500,544 validation, the last seven complete suites had all missed
level 5; the final two fell to highest levels 3 and 2. This changes the earlier
plan to exhaust ten million additional actions: request a graceful pause and
use that learner's capacity for the now-verified curriculum comparison. This is
an experiment decision from the repeated validation decline, not a wall-clock
limit, goal completion, or an assertion that longer PPO training can never work.

The exact owned learner (PID 22307) accepted SIGTERM, saved its model/optimizer/RNG
state at **27,598,848**, closed its workers, and exited **0**. The recovery stage
ran **5,345,280** additional actions; together with the original long stage this
is **7,098,368**, leaving **2,901,632** of the originally planned tranche unused.
All **21 recovery-stage validation suites** completed 20/20 games with no
truncation; highest verified depth remained 5. No error or target success was
logged. Preserve the 24,002,560 primary-best snapshot, the stronger-secondary
20,500,480 source, and the final paused checkpoint. The full 1,409-line log is
archived byte-identically. See [pause record and checksums](results/level10/long-run-pause.json).

A read-only synthetic HUD sensitivity probe did **not** support the idea that
score/level digits caused this particular stall. Replacing only those digits,
blanking them, or blanking the top row all retained LEFT+SPACE as the dominant
action, with SPACE-alone raw probability still about 5e-10 to 8e-10. These
altered observations were never stepped in the emulator or used for training.
This is evidence about one stalled observation, not all later-level failures.
The [probe record](results/level10/stall-hud-sensitivity.json) distinguishes raw
float32 softmax sums from the implemented sampler's effective probabilities.
No HUD masking, sampler change, or gameplay intervention was adopted.

## Training at finer action timing

The fixed-policy 50,000-T-state probe transferred poorly (mean 101.95, highest
level 3), so changing inference timing alone is not adopted. A separate trial
now lets PPO **learn at that cadence**, testing whether more frequent control
can improve later-level play. The recovered original-timing run continues
unchanged. Source is the verified **20,500,480** checkpoint, SHA-256
`510ce9632f744301f3a0f2b9c100b4c5c070afa88f1dbc466324c7c411d9fb70`.

Scale time-related settings together to preserve nominal action-time horizons
and update cadence as far as practical:

| Setting | Original timing | Finer timing |
|---|---:|---:|
| T-states per action | 100,000 | 50,000 |
| Gamma | 0.995 | sqrt(0.995) = 0.9974968671630001 |
| GAE lambda | 0.95 | sqrt(0.95) = 0.9746794344808963 |
| Rollout actions per environment | 128 | 256 |
| Minibatch size | 256 | 512 |
| Validation interval (actions) | 250,000 | 500,000 |
| Training/validation guard (actions) | 100,000 | 200,000 |
| Comparison tranche (additional actions) | 2,000,000 | 4,000,000 |

Both use 32 environments, learning rate 0.00005, entropy 0.001, four epochs,
and 16 minibatches per epoch. Nominal rollout time is 12,800,000 T-states per
environment in both cases. Squaring each new discount/GAE coefficient recovers
the original coefficient for the same nominal elapsed action time. Four million
new-cadence actions match two million old-cadence actions' nominal 200 billion
T-states, not their action count. HUD settling and resets add variable work.

This is a **finer-cadence protocol comparison**, not a claim that every factor
is held constant: the four-frame observation window spans less time, minibatch
size and sample correlations change, and the model must adapt its motion
features. All inputs remain screen-only and reward remains score difference.
No action override, scripted serve, demonstration, or altered game physics is
introduced. Compare complete primary suites, keep separate protocol records,
and leave final-test seeds 40000-40099 untouched.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-timing50k \
  --resume runs/level10-entropy001/step-020500480 \
  --additional-steps 4000000 --tstates 50000 \
  --gamma 0.9974968671630001 --gae-lambda 0.9746794344808963 \
  --rollout 256 --batch-size 512 --learning-rate 0.00005 --entropy 0.001 \
  --eval-every 500000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 200000 --max-episode-steps 200000 \
  --target-level 10 --target-clears 1
```

An incomplete suite still cannot select a checkpoint or meet the target.
Any target replay and final evaluation must run uncapped at the selected
checkpoint's recorded action duration.

The finer-timing run started successfully and completed PPO updates with finite
losses. Configuration checks verified the source hash, squared discount/GAE
coefficients, nominal rollout time, minibatches per epoch, validation interval,
guard duration, and trial budget. Initial short completed training episodes are
not a frozen-policy validation result; wait for the scheduled complete suite.

The first finer-timing checkpoint at **21,000,192** completed **20/20 games**,
mean **87.75**, median **84**, best **129**, highest level **2**, reach counts
2/3/4/5 of **12/0/0/0**. It is below both the fixed-policy finer-timing probe
and the original-timing continuation at comparable nominal action time. This
early adaptation result is not an improvement; keep the planned comparison
running, with the stronger original-timing checkpoints preserved.

The second finer-timing checkpoint at **21,508,096** completed only **19/20
games**. Seed 10015 was still waiting for a serve at **200,000 actions**, score
187, displayed level 3. The entire suite is disqualified from selection; its
completed-subset mean is not a full-suite performance estimate. Training
continued past this guarded validation without intervention or an action
override. The next checkpoint at **22,007,808** finished **20/20 games**:
mean **122.85**, median **123.5**, best **275**, highest level **4**, reach counts
2/3/4/5 of **15/3/2/0**. This recovers intermediate-level performance but has
not yet reproduced a complete level-5 validation at the finer cadence.
At **22,507,520**, another **20/20** suite completed, mean **105**, median
**102.5**, best **257**, highest **4**, reach counts **11/3/1/0**.
At **23,007,232**, **20/20** completed, mean **119.3**, median **121**, best
**271**, highest **4**, reach counts **13/5/1/0**. The planned four-million-action
comparison remains active; no complete level-5 validation has appeared yet.
At **23,506,944**, **20/20** completed, mean **112.55**, median **109**, best
**247**, highest **4**, reach counts **14/5/1/0**.
At **24,006,656**, **20/20** completed, mean **103.6**, median **102**, best
**267**, highest **4**, reach counts **13/2/2/0**. This trial continues toward
its four-million-action endpoint while the curriculum comparison starts.

The final checkpoint at **24,506,368** completed **20/20 games**, mean **108.3**,
median **81**, best **279**, highest **4**, reach counts **10/3/3/0**. The learner
then exhausted its requested action tranche and exited **0**, with target false,
after **4,005,888 additional actions** (rollout rounding beyond the requested
four million). Seven of eight scheduled suites completed; one was disqualified
by the previously documented serve stall. None of the complete suites reached
level 5. Best complete mean was 122.85, below the original-timing source's 132.6;
highest changing-policy training level was 5, which does not override the frozen
validation result. Retain original timing for the curriculum trial. Archive the
full 733-line log byte-identically and preserve all checkpoints. See
[completion record](results/level10/timing50k-completion.json).

All **44 Python tests pass**, including a synthetic 9-to-10 visible-digit
redraw regression: the transient text `00019` followed by `00010` cannot
invent a level-19 reach. This tests the existing HUD-settling logic; no gameplay
or learning behavior was changed. Frozen level-2/level-5 artifacts remain
byte-identical to the committed milestone.

## Longer reward-horizon comparison

A separate read-only analysis of the recorded level-5 game's visible score
found **1,983 consecutive zero-reward actions** during active level-2 play,
at score 131, followed by a one-point reward. This is roughly 111.8 seconds
of nominal emulated action time, not wall-clock runtime. The recorded frames
showed no serve wait or reserve-icon decrease during this stretch. The
recorded score differences sum to 282, with no negative rewards.
See [reward-gap measurements](results/level10/reward-gap-inspection.json).

For a reward 1,983 actions ahead, discount weight is approximately
**0.0000482** with gamma 0.995, versus **0.1375** with gamma 0.999. This
motivates testing a longer reward horizon for finishing nearly cleared walls.
It is a hypothesis about learning, not proof that discounting caused the
observed regression; the evidence is one selected successful replay.

Use the same **18,501,632** source checkpoint and change only **gamma
0.995 -> 0.999**, retaining learning rate 0.00005 and entropy 0.003.
The reward is still the actual score difference; no new reward, demonstration,
hidden policy input, action override, or validation-trajectory training is
introduced. Compare two million additional actions against the now-archived
matched control. The active lower-entropy trial continues independently.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-gamma999 \
  --resume runs/level10-lr5e5/step-018501632 \
  --additional-steps 2000000 \
  --learning-rate 0.00005 --entropy 0.003 --gamma 0.999 \
  --target-level 10 --target-clears 1 \
  --eval-every 250000 --eval-games 20 --eval-seed 10000 \
  --eval-max-steps 0 --max-episode-steps 100000
```

The run started successfully. Configuration comparison confirms that only
gamma and run/budget fields differ from the control. The source checkpoint
hash matches. Final-test seeds 40000-40099 remain unused.

First longer-horizon checkpoint at **18,751,488**: 20/20 complete, mean
**72.3**, median **61**, best **115**, highest level **2**, level 2 in
**9/20**. This is below the matched control and is not an improvement.
Continue the planned comparison; the value function is now learning returns
under a different discount factor, and one early checkpoint is insufficient
to establish a useful longer-term effect.

Subsequent complete longer-horizon validations remain below the source:

| PPO actions | Complete | Mean | Median | Best | Highest level | Level-2 reaches | Level-3 reaches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19,001,344 | 20/20 | 89.35 | 92 | 129 | 2 | 15 | 0 |
| 19,251,200 | 20/20 | 86.45 | 66.5 | 188 | 3 | 9 | 1 |
| 19,501,056 | 20/20 | 91.6 | 97.5 | 127 | 2 | 13 | 0 |
| 19,750,912 | 20/20 | 74.35 | 62 | 125 | 2 | 7 | 0 |
| 20,000,768 | 20/20 | 104.6 | 121 | 199 | 3 | 16 | 2 |
| 20,250,624 | 20/20 | 106.9 | 117 | 134 | 2 | 17 | 0 |
| 20,500,480 | 20/20 | 93.3 | 105 | 168 | 3 | 12 | 1 |

The longer-horizon trial completed its budget at **20,504,576** after
**2,002,944 additional actions**, with no errors and no target success.
Its eight scheduled validation suites all finished 20/20 games; none reached
level 4 or 5. Highest training level was 4. This two-million-action experiment
does not support adopting gamma 0.999 over the original gamma 0.995 settings.
Preserve the run and archive its complete log; devote the continuing learner
to the stronger lower-entropy direction. This is a measured choice within
these runs, not a universal claim about discount factors.

## Optional parallel validation

The three finished/paused runs spent 32.1%, 42.4%, and 35.8% of their logged
elapsed time between validation-start and validation-result events. Added
optional `rl.evaluate --envs N` and PPO `--eval-envs N` to batch neural
inference while separate worker processes own the global-state C emulators.
Both default to 1, retaining the serial protocol. The active trials loaded
the previous modules at launch and continue unchanged.

Each game retains its own policy RNG seeded with `seed + 1000000`, plus a
separate epsilon RNG with the same seed, matching the serial behavior.
Results retain input seed order even when games finish out of order; repeated
seeds are separate jobs. Random-baseline and watch modes remain serial to
preserve their existing behavior. Truncated games remain excluded from scores
and target selection. No gameplay action is supplied by the scheduler.

Evaluation parallelism is included in the PPO validation-protocol identity:
changing it resets inherited best-selection records. Even if checked games
match exactly, floating-point batched inference is not assumed identical for
every possible trajectory. Before training use, compare all 20 complete game
records at the verified 18,501,632 checkpoint with its original serial suite.

The eight-worker evaluation finished **20/20 complete games**, and every
per-game field and historical summary field matched the archived serial suite
exactly: mean **125.35**, median **117**, best **282**, highest level **5**,
reaches 2/3/4/5 of **16/4/2/1**. The level-5 game still ended after 14,525
actions on seed 10011. Record:
[eight-worker validation](results/level10/parallel-validation-8.json).
Whole-process elapsed time was **96.08 seconds** (`/usr/bin/time -p`). A fresh
[serial control](results/level10/parallel-validation-serial-control.json)
took **102.08 seconds**, again matching every historical game record exactly.
That is only a **1.06x observed speed ratio**, not a demonstrated large speedup.
The historical in-trainer serial validation took 174.41 seconds, but load
varied across all these measurements because other learners were active.
The [20-worker check](results/level10/parallel-validation-20.json) also matched
every field of every historical complete game, taking **80.82 seconds**.
That is a **1.26x observed speed ratio** against the fresh serial control.
Use 20 validation workers in the next continuation, retaining protocol
separation and recording `eval_envs` explicitly. Neither active trial was
restarted for this change. The [comparison manifest](results/level10/parallel-evaluation-check.json)
records measurements, result checksums, and the uncontrolled-load caveat.

All **40 Python tests pass**, including mixed-length scheduling, repeated
seeds, per-game policy/epsilon RNGs, truncation, actual native-worker resets,
invalid worker requests rejected before any send, unchanged PPO sampling,
and checksums/equality of the three archived complete-game verification suites.
The original frozen packages and replays remain byte-identical.

A separate four-action trainer smoke test (`runs/parallel-validation-smoke`)
also exited cleanly: two parallel games were deliberately limited to 12 actions,
both were correctly reported as incomplete, neither updated the best model,
and the final checkpoint preserved `best_mean=-1`, `best_level_rank=null`.
This diagnostic checkpoint is not a training candidate or performance result.

### Possible later method change (not implemented)

If plain PPO loses rare successful behavior again, a candidate is
[Self-Imitation Learning (Oh et al., ICML 2018)](https://proceedings.mlr.press/v80/oh18b.html):
replay the agent's own training state/action/return samples and reinforce
actions only where their realized discounted score return exceeds the learned
value estimate. This could help retain useful rare experiences; that benefit
for Breakdown is a hypothesis, not an observed result. No human/scripted
demonstrations, validation trajectories, new rewards, or policy overrides
would be used. No SIL code or changes to either running learner are present.
Importantly, the paper does **not** show a universal benefit: its
[supplementary Atari table](https://proceedings.mlr.press/v80/oh18b/oh18b-supp.pdf)
reports Breakout scores of 452.0 for A2C+SIL versus 501.6 for A2C. That is a
reason to treat it as an unproven fallback, not to replace the currently
improving smaller-rate PPO run without a controlled experiment.

## Self-generated start-state curriculum

Read-only inspection of the two already recorded full games found that they
spent only **207** and **192 actions** at displayed level 5 before GAME OVER,
adding two points each after entry. Earlier levels occupied thousands of
actions per game. Both visible entry screens show `LOOK OUT` and a different
screen layout. Two selected successful replays cannot establish the typical
failure rate or its cause, but they motivate increasing practice on states
that are rarely reached during ordinary from-boot training.

Implement a **Go-Explore-inspired start-state curriculum**, borrowing the idea
of storing self-visited simulator states and returning to them before further
exploration ([Ecoffet et al., 2021](https://arxiv.org/abs/2004.12919)). This is
not a reproduction of the full Go-Explore algorithm or evidence of a gain for
Breakdown. Unlike the paper's full pipeline, this experiment uses the current
PPO learner directly, without trajectory imitation or a separate return policy.

The new option is **off by default**. In an enabled training worker, reaching
a new visible level (minimum 2) captures an opaque emulator snapshot and the
same four screen frames used by the policy. A bounded reservoir retains eight
entries per level per worker. After an episode ends, half of resets may choose
a stored level uniformly, then one of its retained entries uniformly; the rest
boot normally. Until the worker reaches an eligible entry itself, every reset
boots. No external, human, scripted, validation, test, or recorded-replay states
can seed the archive; it is populated only by that live training worker.

The snapshot restores the exact previous machine state, not a fabricated level,
score, ball inventory, or physics setting. Internal bytes are never passed to
the policy. The network chooses every subsequent action. Restored segments
start reward accounting at zero and receive only **new score differences**;
the starting score is not rewarded again. Every segment is labelled
`full_game=false` and logged as `curriculum_episode`, never included in complete
game statistics, checkpoint selection, or target success. Normal training
games remain separate. All validation and final testing still boot at level 1.

Archives and their sampler RNG are worker-local and in memory; like existing
emulator trajectories, they are not preserved by optimizer/model checkpoints.
Resuming a curriculum run rebuilds its archive through new self-play. Restoring
a snapshot does not rewind the policy or boot RNG. Snapshot compatibility is
limited to the same native build and action duration. Custom native callbacks,
real-time playback, and UI snapshots are unsupported. The native build now
replaces the shared library atomically, preserving mappings used by live jobs.

The first **53 tests passed**, including exact screen/reward/terminal continuation after
a snapshot restore, rejection of invalid snapshots without mutation, unchanged
disabled-curriculum trajectories, reward-baseline accounting, bounded archives,
native worker integration, and exclusion of restored segments from the target.
A full frozen-policy compatibility evaluation reproduced **all 20 historical
per-game records exactly** after the native extension: mean 132.6, median 124.5,
best 274, highest 5, with no truncation. Its artifact regression brings the suite
to **54 passing tests**. Record:
[native compatibility evaluation](results/level10/snapshot-native-compatibility-validation.json).

The separate 65,536-action integration run (`runs/curriculum-smoke`) exited
successfully at 20,566,016. It archived ten self-reached entries across levels
2/3, completed ten normal games and three restored segments. Segment rewards
were exactly 90-58=32, 129-61=68, and 99-65=34. Its deliberately 12-action-capped
validation completed no games and selected no model (`best_mean=-1`,
`best_level_rank=null`, target false). The diagnostic is not a performance trial
or a source checkpoint for future training. A further 4,096-action cache smoke
test also exited successfully. Logs and assertions are recorded in
[curriculum verification](results/level10/curriculum-verification.json).

Host memory was tight while both comparisons and the smoke test were active
(roughly 12 GiB of allocated swap; this alone does not establish current paging
rate). Avoid a third full learner for now. Add optional MLX allocation telemetry
and a free-buffer cache target; the cache smoke used `--mlx-cache-mb 512` and
measured about 758.5 MB peak active allocation. The cache target is not an
instantaneous ceiling: MLX reclaims excess reusable buffers on subsequent
allocations. These counters omit other process/system memory. No running
learner was restarted just to alter its allocator.

Inspection confirmed that importing the PPO entry module also imported MLX,
including when Python's spawn mechanism re-imported it in emulator-only workers.
Deferred backend loading to learner construction/main execution and added an
observational worker-runtime check. A matched 4,096-action smoke run confirmed
all four emulator workers had **no MLX import**, and produced **byte-identical
model and optimizer files** plus the same RNG/step/episode state as the pre-change
cache smoke. No policy or PPO math changed. All **55 tests pass**. Fresh import
memory was lower in a one-process check, but this is not a controlled system-wide
memory or speed estimate. See [worker verification](results/level10/worker-import-verification.json).

**Active comparison:** after gracefully pausing the declining original-timing
long run, start a separate two-million-action trial from the verified 20,500,480 source,
retaining original action timing, learning rate, entropy, discount, rollout,
minibatches, and score reward. Enable only the training-start curriculum plus
the allocator cache target. Evaluation remains the guarded, ordinary from-boot
20-game suite, with any truncation disqualifying the entire result. Compare
against the archived original-timing continuation; the source and all earlier
models remain preserved. Final-test seeds 40000-40099 stay unused.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum \
  --resume runs/level10-entropy001/step-020500480 \
  --additional-steps 2000000 --learning-rate 0.00005 --entropy 0.001 \
  --curriculum-probability 0.5 --curriculum-min-level 2 --curriculum-per-level 8 \
  --mlx-cache-mb 512 --eval-every 250000 --eval-games 20 --eval-envs 20 \
  --eval-seed 10000 --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

The curriculum trial started successfully with **32 workers and zero worker MLX
imports**, completed finite PPO updates, and captured its first level-2 entry
through fresh from-boot play. Configuration/hash checks confirm unchanged seed,
environment count, rollout, batch size, epochs, learning rate, entropy, gamma,
GAE lambda, life boundaries, and action timing. Its first scheduled complete-game
validation was pending at launch; archive entries and restored segments are not performance
results or target successes. No validation replay was loaded into training.

First curriculum checkpoint at **20,750,336**: only **19/20 games completed**.
Seed **10000** remained waiting for a serve at **100,000 actions**, score **242**,
displayed level **4**. The entire suite is disqualified; its completed-subset
mean is not a full-suite performance estimate. Selection stayed at
`best_mean=-1`, `best_level_rank=null`, and no target was claimed. Training
resumed normally after the guard, without a scripted serve or policy override.
The finer-timing trial has now ended, leaving this curriculum learner active.

The next checkpoint, **21,000,192**, completed **20/20 full games** with no
truncation: mean **143.95**, median **128**, best **292**, highest level **4**,
reach counts 2/3/4/5 of **16/6/3/0**. Compared with the source's **16/5/2/1**
and mean 132.6, the mean and intermediate-level counts improve, but the level-5
reach is absent. The matched ordinary continuation at the same counter had mean
129.8 and counts **19/4/1/1**. This is an encouraging complete-game result,
not a new depth record or a confirmed general improvement. Continue the planned
curriculum trial without using the held-out final test. The complete records
and independently checked score arithmetic are in
[curriculum validation](results/level10/validation-curriculum-21m.json).
Checkpoint model SHA-256:
`a4c9831580f290b2a210b8870979d705cb7ec671f0655640edfa4713f0152a53`.

Fresh from-boot training also reached level 5 and stored its entry in the
worker-local archive. This is changing-policy training experience available
for further practice, not frozen-policy validation or target success.

At **21,250,048**, **20/20** games completed, mean **96.6**, median **79.5**,
best **189**, highest **3**, reach counts **10/3/0/0**. This regressed from
the prior checkpoint; keep its stronger snapshot and continue the planned
two-million-action comparison. The curriculum has not yet produced a deeper
complete validation game than the preserved starting models.

Subsequent complete local-curriculum validations:

| PPO actions | Complete | Mean | Median | Best | Highest level | Reach counts (2/3/4/5) |
|---:|---:|---:|---:|---:|---:|---|
| 21,504,000 | 20/20 | 104.95 | 98 | 282 | 4 | 14/2/1/0 |
| 21,753,856 | 20/20 | 99.6 | 102.5 | 218 | 4 | 13/2/1/0 |
| 22,003,712 | 20/20 | 117.2 | 114 | 236 | 4 | 14/4/1/0 |
| 22,253,568 | 20/20 | 117.95 | 121 | 225 | 4 | 14/6/2/0 |
| 22,503,424 | 20/20 | 126.4 | 115.5 | 278 | 4 | 14/5/3/0 |

The local-archive control then exhausted its tranche and exited **0** at
**22,503,424**, after **2,002,944 additional actions**, with target false.
Seven of eight validation suites completed; the first was disqualified by a
serve stall. None of the complete suites reached level 5; best mean remained
143.95 at 21,000,192. Preserve the stronger snapshot and final checkpoint, and
archive the full 964-line log byte-identically.

Training itself completed 244 ordinary games and 202 practice segments with no
guard hits. Only **eight** practice segments started at level 5, averaging
**151.75 actions and 1.375 newly earned points**; none reached level 6. Thus the
local curriculum did not deliver substantial successful late-level practice in
this tranche. Continue the peer-sharing comparison, not this final local model.
These are training diagnostics, not frozen-policy results. See
[local curriculum completion](results/level10/local-curriculum-completion.json).

## Peer-shared curriculum comparison

At approximately **21.35 million** actions in the local-curriculum run, only
**2 of 32 workers** had generated a level-5 entry, and only **one** completed
practice segment started at level 5 (217 actions, three newly earned points).
There were 66 completed practice segments in total, mostly starting at level 2.
This is a concrete limitation of keeping rare states local to their discoverer,
not a claim that more practice will necessarily solve level 5. See
[practice audit](results/level10/local-curriculum-practice-audit.json).

Add optional `--curriculum-share` (off by default) to distribute a newly
self-reached level entry to the other training workers in the **same run**.
Each worker retains the same bounded per-level reservoir and uniform level/entry
reset selection. It considers its own entries once and each peer publication
once. Publication does not interrupt or alter active games; peers can choose
those entries only on a later automatic reset. A publication the discoverer's
reservoir did not retain still reaches peers. The ordinary from-boot reset
fraction remains one half.

No archive files, previous-run archives, validation/test states, demonstrations,
or scripted actions are loaded. The neural network and score-only PPO objective
are unchanged. Opaque snapshots travel only through the environment scheduler;
it strips them before returning results to the learner/logger. Practice records
identify the originating worker/action and remain explicitly `full_game=false`.
Validation still uses ordinary from-boot environments and cannot invoke sharing.

All **59 tests pass**, including exact complete-game continuation after moving a
snapshot to another process, peer-only routing, private-payload removal, no
active-game mutation on receipt, invalid batches rejected before archive
mutation, and proper segment provenance. A matched 65,536-action local smoke
produced **byte-identical model/optimizer files and identical RNG/counters** to
the pre-refactor local smoke. A shared smoke published ten entries to three
peers each and completed five practice segments, three from another worker's
entry; each rewarded only score earned after restoration. Both exited **0**.
Their deliberately 12-action-capped validation was incomplete and selected no
model (`best_mean=-1`, `best_level_rank=null`). Neither smoke checkpoint will
be used for the learning comparison.
Each restored segment's worker/action origin was also matched to an earlier
publication in that same smoke log. Full records:
[shared-curriculum verification](results/level10/shared-curriculum-verification.json).

Launch a separate two-million-action trial from the same verified 20,500,480
source, changing only archive sharing relative to the local curriculum (same
cache target, policy, temporal settings, optimizer, and validation suite). This
new run starts with empty archives and must discover its own entries. The
existing local control continues to its planned endpoint with its already
loaded implementation unchanged. All final-test seeds remain unused.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-shared \
  --resume runs/level10-entropy001/step-020500480 \
  --additional-steps 2000000 --learning-rate 0.00005 --entropy 0.001 \
  --curriculum-probability 0.5 --curriculum-min-level 2 --curriculum-per-level 8 \
  --curriculum-share --mlx-cache-mb 512 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

The shared run started successfully, completed finite PPO updates, and
published its first self-reached level-2 entry to **31 peers**. All 32 workers
reported no MLX import. Source/configuration checks show that, after normalizing
the previously absent sharing flag to false, the only configuration differences
from the local control are **run directory and `curriculum_share`**. Its first
complete-game validation is pending; publications are not performance results.

First shared-curriculum checkpoint, **20,750,336**: **20/20 complete games**,
mean **122.75**, median **110**, best **273**, highest level **4**, reach counts
**15/4/3/0**. The matched local-curriculum suite was incomplete and cannot provide
a full-suite mean comparison; the source's complete mean was 132.6 with one
level-5 reach. This first shared result is valid but not a new depth milestone.
The local control has now completed its tranche, leaving the shared run active.

### Peer-sharing validation follow-up

The shared-curriculum suites below all completed from boot:

| PPO actions | Mean | Median | Best | Highest level | Reaches 2/3/4/5 |
|---|---:|---:|---:|---:|---|
| 20,750,336 | 122.75 | 110 | 273 | 4 | 15/4/3/0 |
| **21,000,192** | **130.3** | **114.5** | **295** | **5** | **17/5/3/2** |
| 21,250,048 | 114.55 | 110.5 | 233 | 4 | 18/5/1/0 |
| 21,504,000 | 136.35 | 122.5 | 302 | 5 | 17/5/3/1 |
| 21,753,856 | 112.2 | 114 | 198 | 3 | 14/4/0/0 |
| 22,003,712 | 87.65 | 90.5 | 137 | 3 | 12/1/0/0 |
| 22,253,568 | 94 | 102 | 192 | 3 | 12/1/0/0 |
| 22,503,424 | 114.45 | 107.5 | 247 | 4 | 16/4/2/0 |

Select the **21,000,192** checkpoint for a secondary-validation audit because
it has the strongest primary depth count, not because of any fresh-test result.
Model SHA-256: `53a59a8777b548898f0260b47aae9b6293b24cefe42dfd43a98f7134191049cc`.
Its [complete primary records](results/level10/validation-curriculum-shared-21m.json)
independently recompute to the summary above: seed 10005 scored 295 in 14,487
actions and seed 10008 scored 282 in 13,495 actions, both GAME OVER at level 5.
Sharing has increased actual later-level practice, but restoration segments do
not enter these validation counts. The 50-game audit uses the existing secondary
seeds 10100-10149, with a 100,000-action guard that disqualifies any incomplete
suite. Final-test seeds remain untouched.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-curriculum-shared/step-021000192/model.safetensors \
  --games 50 --envs 20 --seed 10100 --max-steps 100000 \
  --output results/level10/validation-audit-curriculum-shared-21m.json
```

The [secondary audit](results/level10/validation-audit-curriculum-shared-21m.json)
finished **50/50 complete**, mean **135.12**, median **121**, best **304**,
highest level **5**, counts **39/15/8/2**. Level-5 games: seed 10113 scored 304
in 18,881 actions; seed 10139 scored 288 in 14,168 actions. This is better mean
and intermediate-depth coverage than the source's corresponding audit
(122.6, counts 37/13/4/2), but the same two level-5 reaches and no level 6.
The source and this checkpoint remain preserved. These are repeatedly used
validation seeds, not a fresh test or proof of generalization.

The [304-point replay](results/level10/curriculum-shared-secondary-replay.html)
independently reproduced seed 10113 through GAME OVER, **18,882 frames**.
Every per-game field matches the parallel audit. Decoded screens show entry to
level 5 at frame 18,552 / score 296, followed by **329 actions and eight new
points** before game over. This is a selected success, not typical survival.
The frozen network reproduces all 18,881 seeded actions with zero mismatches;
its input remains the original four screen frames. Replay SHA-256:
`f7615f670c818a43961d8858c663fb36be8c095c1e1856a3e5559b8b45f42237`.
See the [replay verification](results/level10/curriculum-shared-replay-verification.json)
and [policy inspection](results/level10/curriculum-shared-policy-inspection.json).
The recording is local and has not replaced either published milestone.

### Next comparison: concentrate self-generated practice at level 5

At action **21,950,464**, completed practice segments starting at level 5 total
**39 segments / 5,686 actions / 50 new points**. That is more practice starts
than the local control's final eight, but their average length is only about
146 actions. Selecting levels uniformly at resets does not balance the number
of training actions: earlier-level segments last thousands of actions. This
count excludes unfinished segments and level-5 actions reached partway through
other games; it is not an exact count of every level-5 observation.

Use the now-audited **21,000,192** shared checkpoint as the source for a separate
two-million-action focused-curriculum comparison. Set minimum archived level
to **5** and reset probability to **0.95**. Both settings change together as a
single focused-practice intervention, not an isolated effect of either setting.
Once entries exist, each reset retains a 5% chance of a fresh boot; before
discovery, all workers boot normally. Longer full games mean 5% of resets is
not 5% of training actions. New higher-level discoveries remain eligible.

Archives again start **empty** and contain only this run's own learned play;
no old archive, replay, validation state, or test state is imported. Every
restored action is still selected by the current network. Keep all optimizer,
reward, input, action-timing, rollout, sharing, and complete-game validation
settings unchanged. Evaluate from boot every 250,000 actions and watch for
early-level forgetting as well as later-level survival. The original shared
comparison continues to its planned endpoint; its earlier snapshots are safe.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-focused \
  --resume runs/level10-curriculum-shared/step-021000192 \
  --additional-steps 2000000 --learning-rate 0.00005 --entropy 0.001 \
  --curriculum-probability 0.95 --curriculum-min-level 5 --curriculum-per-level 8 \
  --curriculum-share --mlx-cache-mb 512 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

This is a predeclared next experiment, not a result or a claim that level 5
has been learned. It uses the existing tested configuration, without modifying
the trainer or the in-flight shared run.

The focused run started successfully from the verified `53a59a...` model and
optimizer, with finite PPO updates and all 32 workers reporting no MLX import.
Its first validation and newly generated archive entries are pending.

First focused checkpoint, **21,250,048**: **20/20 complete**, mean **129.5**,
median **110**, best **290**, highest level **4**, counts **17/4/3/0**.
No level-5 training entry had been discovered by this validation, so all its
training so far was ordinary from-boot play: this is not yet a test of restored
level-5 practice. Continue the tranche while monitoring actual archive creation
and restored-segment exposure separately from full-game performance.

### Broad shared-curriculum tranche completed

`runs/level10-curriculum-shared` exited **0** after **2,002,944** new actions,
stopping at **22,503,424**, `target_met=false`. All **eight** validation suites
completed their 20 games. The best-by-level checkpoint remains **21,000,192**
(two level-5 reaches), and the best mean is **136.35** at 21,504,000. The final
suite's mean is 114.45, highest level 4; do not replace the earlier audited
checkpoint with the last weights.

The training log contains **241 complete from-boot games**, **242 restored
segments**, **340 archive publications**, and **zero training guard hits**.
All 242 restored segments were checked against their earlier same-run
worker/action publication, starting score/level, `full_game=false`, and reward
equal to final score minus restored initial score. Of those, **58** started at
level 5 (57 peer-sourced), totaling **8,441 actions and 74 new points**:
mean 145.53 actions and 1.276 points. This substantially increased starts
compared with the local control's eight, but has not taught level-5 completion.
The short segments explain the next focused-practice intervention.

The complete **994-line raw log** was copied and compared byte-for-byte with
the stopped source. Checkpoints and full details are preserved in the
[completion record](results/level10/shared-curriculum-completion.json) and
[archived log](results/level10/logs/level10-curriculum-shared.jsonl).
Latest model SHA-256:
`a936750c1c9ccca15dd88adeb1cf3eacc446eadf8492c74c795508f70f4ccfb4`.
The focused run continues; the broad run is stopped and must not be relaunched
merely because its logs no longer change. All **59 regression tests pass**
again, and the original level-2/level-5 packages remain unchanged from HEAD.
No final-test seeds have been used.

Focused checkpoint **21,504,000** completed **19/20** validation games and is
**disqualified**. Seed 10010 reached the 100,000-action guard at score 61,
level 2, `waiting=true`, not GAME OVER. Seed 10000 did complete at level 5,
score 268, but one completed deep game cannot qualify the incomplete suite.
The reported 150 mean is only over the 19 completed games and is **not** a
full-suite performance result. Selection correctly retained the source records
(`best_mean=130.3`, two level-5 reaches), rather than promoting this snapshot.
The [raw disqualified records](results/level10/validation-curriculum-focused-21m5-disqualified.json)
are preserved. No training archive entry existed yet at this checkpoint; no
validation state will be transferred to supply one. The live learner continued
normally after evaluation and remains the only active run.

The focused run's **21,753,856** validation subsequently completed **20/20**:
mean **127.55**, median **119**, best **277**, highest level **5**, counts
**17/4/2/2**. This matches the source's two level-5 reaches but does not beat
its intermediate-depth counts or mean. At this point there were still no
eligible training archive entries, so it remains a from-boot continuation,
not evidence for the focused-reset intervention. Preserve the source and
continue monitoring self-reached training entries rather than importing the
successful validation states.

Focused practice became active at **21,820,492**: worker 11 reached visible
level 5, score 294, from a full game (source action 25,635, episode action
15,148, boot timing 114,575), publishing to 31 peers. A second full-game entry
arrived at **21,946,951**, worker 6, score 270 (source action 29,587, episode
action 12,540, boot timing 56,331). No validation or prior-run archive was used.
By the logged action **21,982,613**, **243** restored level-5 segments had
completed, averaging **130.27 actions / 1.025 new points**, best two new points,
highest level 5. The increased exposure is real, but survival and depth have
not improved yet. Continue this existing trial now that the intended treatment
is actually active; no extra bridging experiment was launched.

### Focused practice shows within-start learning

At **22,535,878**, compare each of the same four self-generated starts' first
50 versus latest 50 completed practice segments. The windows do not overlap;
each origin matches an earlier publication in this run. New-score means:

| Origin worker / action | Starting score | First 50 | Latest 50 | Best new points |
|---|---:|---:|---:|---:|
| 6 / 29,587 | 270 | 1.00 | 3.50 | 26 |
| 9 / 32,864 | 295 | 1.00 | 4.62 | 28 |
| 11 / 25,635 | 294 | 1.06 | 3.42 | 14 |
| 27 / 37,266 | 288 | 1.10 | 5.72 | 23 |

Survival also increased within each start. This is evidence of improving
practice behavior, not merely replacement by easier initial states, but is
still changing-policy training data, not a controlled causal estimate or
complete-game performance. All **1,404** segments still ended at level 5.
Full timing, counts, origins, and non-overlapping windows are in the
[same-start audit](results/level10/focused-same-start-practice-audit.json).

From-boot validation after focused practice began:

| PPO actions | Complete | Mean | Median | Best | Highest level | Reaches 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 22,003,712 | 20/20 | 121.5 | 125 | 269 | 4 | 16/6/1/0 |
| 22,253,568 | 20/20 | 128.1 | 123 | 286 | 4 | 17/5/1/0 |
| 22,503,424 | 20/20 | 120.85 | 109.5 | 282 | 5 | 13/4/3/1 |
| 22,753,280 | 20/20 | 142.6 | 131.5 | 265 | 4 | 16/8/4/0 |
| 23,003,136 | 20/20 | 131.35 | 120 | 263 | 4 | 17/6/1/0 |

No new highest level is verified yet. Keep the stronger audited source as the
public performance reference while the focused trial finishes. The within-start
learning signal supports a longer focused continuation if the final checks
remain sound; any such continuation must preserve earlier checkpoints and
continue complete-game validation, without using the fresh final test.

The focused 22,503,424 checkpoint's level-5 game (seed 10016, score 282)
independently reproduced through GAME OVER in a
[full replay](results/level10/focused-22m5-validation-replay.html), with all
per-game fields matching validation. Decoded screens enter level 5 at frame
10,952 / score 278 and end 186 actions / four new points later. This selected
recording does not show a new depth or survival milestone, despite the stronger
practice scores. Its [verification record](results/level10/focused-replay-verification.json)
includes model/replay hashes. Neither recording nor validation states enter
the training archive.

### Focused trial completed; longer continuation

The focused learner exited **0** at **23,003,136**, after **2,002,944** new
actions. Its **2,543-line raw log** is archived byte-identically. The run
completed **255 from-boot training games** and **2,110 practice segments**,
with no training guard hits. All practice origins and score differences were
checked against the run's four self-reached, from-boot archive publications.
Practice consumed **401,879 actions**; first 100 segments averaged **1.04**
new points, last 100 **10.22**, best **39**, highest level still **5**.
The per-origin audit above supports improvement within the same starts, not
only a changed mixture of entries. These remain training diagnostics.

All seven complete validation suites and the one disqualified suite are
preserved. The final complete 20-game suite has mean **131.35**, median **120**,
best **263**, highest level **4**, counts **17/6/1/0**. No level 6 is proven.
See the [completion record](results/level10/focused-curriculum-completion.json)
and [full log](results/level10/logs/level10-curriculum-focused.jsonl).

Continue the final focused checkpoint for **ten million additional actions**
with learning settings unchanged. This is a further learning experiment,
selected because of the improving frontier practice and intact complete-game
checks, **not** promotion of the final weights as the strongest evaluated model.
Keep the separately audited `53a59a...` checkpoint and its 304-point replay
as the performance reference until a new candidate is properly evaluated.

Source **23,003,136** model SHA-256:
`4983ca8d163732e2d21c4f3357a27746c7fb27bc31e03518339e5a94c3b46929`.
The numbered and latest model/optimizer/state files match. Resume weights,
optimizer, and learner RNG; emulator trajectories and in-memory archives are
not checkpointed. The continuation must discover its own entries again and
cannot import the old archive or any validation/replay state. This warm-up
cost is explicit; before the first entry it will play only from boot.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-focused-long \
  --resume runs/level10-curriculum-focused/step-023003136 \
  --additional-steps 10000000 --learning-rate 0.00005 --entropy 0.001 \
  --curriculum-probability 0.95 --curriculum-min-level 5 --curriculum-per-level 8 \
  --curriculum-share --mlx-cache-mb 512 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

This tranche is a review interval, not a wall-clock limit or a revised goal.
Continue comparing complete-game reach rates, segment survival, and archive
diversity. Final-test seeds 40000-40099 remain unused, and the level-2 and
committed level-5 packages remain unchanged.

The longer continuation started successfully from the verified `4983ca...`
checkpoint, completed finite PPO updates, and reported all **32** emulator
workers without MLX imports. Its configuration comparison changes only the
run/source paths, source hash, and action budget; all learning and validation
settings are unchanged. The initial focused run is stopped and archived;
this continuation is the only active learner. No archive was imported.

Initial long-continuation checks:

| PPO actions | Complete | Mean | Median | Best | Highest level | Reaches 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 23,252,992 | 20/20 | 113 | 119.5 | 234 | 4 | 14/4/1/0 |
| 23,502,848 | 20/20 | 111.05 | 118.5 | 255 | 4 | 13/2/2/0 |
| 23,752,704 | 20/20 | 119.25 | 110 | 298 | 5 | 13/4/2/1 |
| 24,002,560 | 20/20 | 92.75 | 100 | 129 | 2 | 13/0/0/0 |
| 24,252,416 | 20/20 | 101.15 | 110.5 | 217 | 4 | 13/2/1/0 |
| 24,502,272 | 19/20 — disqualified | — | — | — | — | — |
| 24,752,128 | 20/20 | 96.65 | 62 | 203 | 3 | 10/3/0/0 |
| 25,001,984 | 20/20 | 86.8 | 66.5 | 204 | 3 | 10/3/0/0 |
| 25,251,840 | 20/20 | 76.45 | 63.5 | 118 | 2 | 9/0/0/0 |
| 25,501,696 | 20/20 | 72.10 | 60.5 | 116 | 2 | 7/0/0/0 |
| 25,751,552 | 20/20 | 72.80 | 57 | 209 | 4 | 6/1/1/0 |
| 26,001,408 | 20/20 | 69.00 | 56.5 | 121 | 2 | 6/0/0/0 |
| 26,251,264 | 20/20 | 60.15 | 58.5 | 109 | 2 | 3/0/0/0 |
| 26,501,120 | 20/20 | 61.30 | 56.5 | 122 | 2 | 4/0/0/0 |
| 26,750,976 | 20/20 | 59.80 | 55.5 | 119 | 2 | 4/0/0/0 |
| 27,000,832 | 20/20 | 77.05 | 63 | 121 | 2 | 8/0/0/0 |
| 27,250,688 | 20/20 | 59.00 | 53.5 | 127 | 2 | 2/0/0/0 |
| 27,500,544 | 20/20 | 78.05 | 66.5 | 123 | 2 | 10/0/0/0 |
| 27,750,400 | 20/20 | 80.50 | 62.5 | 131 | 2 | 8/0/0/0 |
| 28,000,256 | 20/20 | 77.15 | 63.5 | 125 | 2 | 9/0/0/0 |

No new training archive entry existed through progress action **23,670,784**.
The process is confirmed live and producing finite updates; continue monitoring
the restart phase without presenting these early checks as an improvement.

The long run rebuilt its archive at **23,840,516** (**837,380** new actions):
worker 3 reached level 5 / score 284 in a full training game and published
the entry to 31 peers. Origin: source action 26,169, episode action 13,346,
boot timing 186,945. This was a new same-run discovery, not a restored old
archive or validation state. By progress action **23,937,024**, **81** completed
practice segments from it averaged **10.04** new points, best **35**, highest
level **5**. Practice has resumed around the previous run's late reward range
on this new start, but these are training observations, not a held-out estimate
or a new complete-game depth milestone. The continuation remains active.

### New training frontiers, not full-game milestones

The same run's learned policy subsequently generated entries at levels 6-8.
Every later entry below arose from a restored training segment and was shared
with 31 peers; no external, validation, or replay state was imported.

| Visible level entered | First action | Worker | Start of source episode | Source episode finished at |
|---|---:|---:|---|---|
| 5 | 23,840,516 | 3 | Fresh boot | Level 5, score 310 |
| 6 | 24,209,801 | 8 | Restored level 5 | Level 6, score 338 |
| 7 | 24,310,623 | 30 | Restored level 6 | Level 7, score 386 |
| 8 | 24,455,794 | 17 | Restored level 7 | Level 8, score 526 |

Publication/source-episode action counts, parent archive origins, starting
scores, and newly earned rewards were checked. The full provenance chain is in
the [frontier audit](results/level10/focused-long-frontier-audit.json).
These changing-policy training reaches cannot satisfy the goal or substitute
for a frozen checkpoint playing from boot. The complete validation suites above
show that early-level play has weakened even as training explores deeper states.

### Protect a share of from-boot training actions

Reconstruct exact action origins from contiguous completed episode intervals.
Between actions **23,840,516-24,002,560**, restored starts accounted for
**58,667 / 162,045 actions (36.20%)**. In the next interval,
**24,002,561-24,252,416**, they accounted for **192,340 / 249,856 (76.98%)**.
Both windows are now fully covered by completed episode records; the earlier
provisional second-window estimate lacked 341 actions, now classified as
from-boot. This confirms that a fixed reset probability does not fix the sample
mix as practice episodes grow. See the
[action-balance audit](results/level10/focused-long-action-balance-audit.json).

Add optional `--curriculum-boot-envs`: reserve the first N training workers for
fresh-boot resets while still allowing them to discover and share entries.
Other workers retain the existing reset probability. Reserving 16 of 32 therefore
protects at least half of every rollout's action samples for from-boot play.
The default is zero, preserving the previous behavior. No actions, rewards,
observations, model architecture, or evaluator starts change. New origin
counters measure the actual mixture per rollout and since process start.

All **64 tests pass**, including reserved workers ignoring populated peer
archives, continuing to publish discoveries, correct worker roles, no worker
GPU imports, and invalid settings failing before spawning or creating a run.
A matched 65,536-action smoke with reservation disabled produced **byte-identical
model/optimizer files and identical learner RNG/counters** to the earlier shared
smoke. Its origin counters sum to all 65,536 actions, and incomplete diagnostic
validation selected nothing. A reserved-worker smoke also completed cleanly;
its protected workers played eight games with populated archives and never
restored. It exercised 2,798 restored actions in other workers but ended before
those practice episodes finished. A second reserved-worker check uses reset
probability one to exercise completed practice segments before the comparison
is launched. None of these diagnostic checkpoints is a performance candidate.

The intended matched comparison starts from the same **23,003,136** checkpoint
as the active long control, with the same ten-million-action budget and learning
settings, changing only the reserved worker count from zero to 16. Archives
start empty in both cases. Leave the current long run unchanged while comparing
from-boot consistency and later-level practice. Final-test seeds remain unused.

The probability-one reserved-worker smoke finished **0** with seven complete
practice segments (three sourced from a protected collector), each correctly
labelled `full_game=false` and rewarding only new score. Protected workers
completed six ordinary games with populated archives and never restored.
Every logged rollout retained at least **256/512** from-boot actions; cumulative
counts were **46,359 from boot / 19,177 restored**, exactly 65,536 total.
The capped two-game diagnostic remained incomplete and selected no model.
All three implementation logs are archived byte-identically, with checks in the
[reservation verification](results/level10/reserved-boot-verification.json).

At the long run's disqualified 24,502,272 validation, seed 10010 was still
playing at level 1 / score 56 after 100,000 actions (`waiting=false`), not
waiting for a serve. It is an unfinished game; the endpoint alone does not
establish the cause of its long duration. The
[raw records](results/level10/validation-focused-long-24m5-disqualified.json)
are retained, and the 19-game partial mean is not a valid full-suite result.

### Matched protected-boot comparison launch

The three implementation smoke processes exited successfully and their raw
archives match byte-for-byte. Source model SHA-256 was rechecked as
`4983ca8d163732e2d21c4f3357a27746c7fb27bc31e03518339e5a94c3b46929`.
Use a new run directory; do not resume any diagnostic model or replace the
active control. The only experimental configuration change is 16 reserved
workers instead of zero (the absent flag in the control normalizes to zero).

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced \
  --resume runs/level10-curriculum-focused/step-023003136 \
  --additional-steps 10000000 --learning-rate 0.00005 --entropy 0.001 \
  --curriculum-probability 0.95 --curriculum-min-level 5 --curriculum-per-level 8 \
  --curriculum-share --curriculum-boot-envs 16 --mlx-cache-mb 512 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

Both runs initialize empty in-memory archives from the same checkpoint and
learner RNG. Reserved workers can still discover/share their own entries. The
guarantee is at least 50% from-boot training actions, not exactly 50% restored
actions. Evaluate frozen policies only on ordinary from-boot games; restored
training reaches never qualify the goal. No final-test seeds are used.

Startup is verified: 32 emulator workers report no MLX import; the first 16
disable restored resets and the other 16 enable them. Normalizing the absent
control flag to zero, the saved configurations differ only in the run path
and reserved-worker count. At action **23,064,576**, updates are finite, all
61,440 new actions are from boot (no new archive yet), and every logged
rollout/cumulative origin counter is consistent. Both learner processes are
confirmed active; neither diagnostic process remains running. The frozen
level-2/level-5 models and replay files still match the committed versions.

The first balanced validation, at **23,252,992**, completes all 20 games:
mean **113**, median **119.5**, best **234**, highest level **4**, reach counts
2/3/4/5 = **14/4/1/0**. Before any new archive exists, the balanced and control
checkpoints have byte-identical model, optimizer, and evaluation JSON files;
learner RNG, action/episode counters, and best-selection state also match.
Model SHA-256: `78a6be23b929076fecf36a97823a24771c65d541626abfdd92f0e54fdf61dc21`.
This is a useful deterministic compatibility check, not evidence that protected
workers have improved performance: their reset behavior has not diverged yet.

Balanced-run validation (the first three checkpoints precede its archive):

| PPO actions | Complete games | Mean | Median | Best | Highest level | Reach counts 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 23,252,992 | 20/20 | 113.00 | 119.5 | 234 | 4 | 14/4/1/0 |
| 23,502,848 | 20/20 | 111.05 | 118.5 | 255 | 4 | 13/2/2/0 |
| 23,752,704 | 20/20 | 119.25 | 110 | 298 | 5 | 13/4/2/1 |
| 24,002,560 | 20/20 | 85.20 | 62 | 202 | 3 | 9/1/0/0 |
| 24,252,416 | 20/20 | 95.20 | 85 | 205 | 3 | 10/2/0/0 |
| 24,502,272 | 19/20 — disqualified | — | — | — | — | — |
| 24,752,128 | 20/20 | 111.90 | 118 | 248 | 4 | 14/3/1/0 |
| 25,001,984 | 20/20 | 83.25 | 59.5 | 241 | 4 | 7/2/1/0 |
| 25,251,840 | 20/20 | 103.70 | 67 | 211 | 4 | 10/5/2/0 |
| 25,501,696 | 20/20 | 96.05 | 62.5 | 230 | 4 | 9/2/1/0 |
| 25,751,552 | 20/20 | 96.55 | 98.5 | 198 | 3 | 13/2/0/0 |
| 26,001,408 | 20/20 | 121.25 | 105 | 280 | 4 | 13/4/3/0 |
| 26,251,264 | 20/20 | 100.55 | 94.5 | 255 | 4 | 16/2/2/0 |
| 26,501,120 | 20/20 | 83.55 | 78 | 134 | 2 | 10/0/0/0 |
| 26,750,976 | 20/20 | 85.00 | 71 | 150 | 3 | 13/1/0/0 |
| 27,000,832 | 20/20 | 83.20 | 85 | 125 | 2 | 12/0/0/0 |
| 27,250,688 | 20/20 | 97.90 | 77.5 | 268 | 4 | 10/2/2/0 |
| 27,500,544 | 20/20 | 78.85 | 62.5 | 132 | 3 | 8/1/0/0 |
| 27,750,400 | 20/20 | 77.35 | 61.5 | 181 | 3 | 7/1/0/0 |
| 28,000,256 | 20/20 | 84.70 | 63 | 180 | 3 | 9/1/0/0 |
| 28,250,112 | 20/20 | 73.35 | 61 | 135 | 2 | 7/0/0/0 |
| 28,504,064 | 20/20 | 79.75 | 66.5 | 128 | 2 | 9/0/0/0 |
| 28,753,920 | 20/20 | 71.15 | 66.5 | 114 | 2 | 10/0/0/0 |

All three pre-archive numbered checkpoints have byte-identical model, optimizer, and
evaluation JSON files to the control at the same action counts. The level-5
validation reach is therefore a reproduced prior result, not a treatment gain.
No evaluation state is added to the training archive.

The first post-divergence result at 24,002,560 is mixed: highest level 3 versus
the matched control's 2, but mean 85.2 versus 92.75. It does not establish a
performance improvement; continue to compare the subsequent complete suites.

The balanced run rediscovered its first training entry at **23,840,516**:
protected worker **3**, level **5**, score **284**, source action **26,169**,
episode action **13,346**, boot timing **186,945**, shared with 31 peers.
This exactly matches the control's first discovery. Subsequent reset choices
now differ. Through **23,969,792**, 31 complete restored segments all came
from that entry, ran only in workers 16-31, and credited only new score.
Protected workers completed 16 ordinary games with populated archives without
restoring. All logged rollout and cumulative origin checks pass; the latest
rollout contained **3,328 from-boot / 768 restored** actions. See the
[first-practice audit](results/level10/balanced-first-practice-audit.json).

### Reallocate compute after the long control's plateau

At the review through **27,889,956**, the control has 15 complete validation
suites since its last level-5 reach; the latest eight all reach only level 2.
It has completed **847** level-8 practice segments, best **52** new points,
none reaching level 9. On seven identical starts with at least 50 segments,
disjoint first/last-25 reward means decline on five, tie on one, and improve
on one. These are training diagnostics, not independent performance estimates;
they do not show consistent further frontier learning. Detailed observations
are saved in the [plateau review](results/level10/focused-long-plateau-review.json).

If no new depth arrives, gracefully pause the control after its next scheduled
validation at **28,000,256**, about five million new actions. This shortens its
initial ten-million-action ceiling based on observed regression/plateau, not a
wall-clock limit. Preserve all checkpoint/log evidence and let the protected-
boot run continue its planned budget with more available compute. Comparisons
beyond the control's stopping point will not be called matched-budget results.

The condition remained true: the 28,000,256 validation completed all 20 games,
mean **77.15**, median **63.5**, best **125**, highest level **2**. SIGTERM was
handled cleanly; the process exited **0** at **28,045,312**, target false.
Actual added actions: **5,042,176**; unused original ceiling: **4,957,824**.
There were 20 validation suites (19 complete, one disqualified), 406 complete
from-boot training games, and **4,454** restored segments, including **4,323**
peer-sourced. All restored origins and newly earned rewards were verified;
there were no training truncations. Final practice counts by starting level
5/6/7/8 were **1,489 / 1,072 / 1,007 / 886**. No full-game depth beyond 5 and
no restored depth beyond 8 was reached.

The **5,435-line** raw log is archived byte-identically, SHA-256
`37b12757f56033ef0fb9768ee40445260e249bf3fd340cbfde14d5101f70bd5d`.
Latest model/optimizer/state hashes and the complete pause audit are in
[focused-long-pause.json](results/level10/focused-long-pause.json).
The learner, optimizer, and RNG are saved. In-memory archive banks and
unfinished emulator trajectories are not checkpointed and ended with the
process; a future restart would rebuild them from its own new training games.
Only the balanced learner remains active. At its 24,252,416 check, all 20 games
finish, mean **95.2**, highest level **3**, versus matched control **101.15 / 4**;
there is still no demonstrated treatment gain. Continue gathering evidence.

The balanced 24,502,272 suite is disqualified: seed **10010** remained at
level **5** / score **288**, waiting to serve after 100,000 actions. Its
displayed level 5 is not a completed-game result, and the 19-game partial mean
is not comparable to full-suite means. The
[raw evaluation](results/level10/validation-balanced-24m5-disqualified.json)
is archived byte-identically; no model is selected from this suite.

The protected-boot run has now rebuilt a deeper training archive:

| First entered level | Global action | Worker/source action | Source episode start | Source episode end |
|---|---:|---|---|---|
| 5 | 23,840,516 | 3 / 26,169 | From boot | Level 5, score 310 |
| 6 | 24,362,781 | 28 / 42,489 | Restored level 5 | Level 6, score 329 |
| 7 | 24,654,112 | 31 / 51,593 | Restored level 6 | Level 7, score 412 |
| 8 | 24,844,436 | 19 / 57,541 | Restored level 7 | Level 8, score 525 |

All first entries trace to this run's own level-5 discovery, boot timing
186,945; no validation/replay state was imported. Source episode alignment,
parent origins, new reward, protected-worker roles, and action counters pass
the [frontier audit](results/level10/balanced-frontier-audit.json).
These are restored training reaches, not new frozen-policy/full-game milestones.

### Smaller-update protected-boot comparison

The 24,752,128 balanced validation improves on its matched control (mean
111.9 / level 4 versus 96.65 / level 3), but the next suite at 25,001,984 is
mixed again (83.25 / level 4 versus 86.8 / level 3). There is no consistent
full-game improvement yet. Recent PPO update logs include mean approximate
KL around 0.014-0.016; that is observational evidence of policy movement, not
proof that update size causes the regression.

Test a five-times-smaller learning rate, **1e-5** instead of **5e-5**, from the
same **23,003,136 / 4983ca...** source. Keep 16/32 protected workers, all other
learning/evaluation settings, and fresh empty archives unchanged. Start with
two million additional actions for a matched early comparison; extend if the
results warrant it. The existing balanced run continues unchanged. This does
not change observations, rewards, policy sampling, or actions, and imports no
training states from another run. Final-test seeds remain untouched.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced-lr1e5 \
  --resume runs/level10-curriculum-focused/step-023003136 \
  --additional-steps 2000000 --learning-rate 0.00001 --entropy 0.001 \
  --curriculum-probability 0.95 --curriculum-min-level 5 --curriculum-per-level 8 \
  --curriculum-share --curriculum-boot-envs 16 --mlx-cache-mb 512 \
  --eval-every 250000 --eval-games 20 --eval-envs 20 --eval-seed 10000 \
  --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

The smaller-rate process is active. Saved configurations differ only in the
learning rate, run path, and explicit initial action budget. All 32 workers
omit MLX and report the expected 16 protected / 16 practice-enabled roles.
Initial updates are finite; at **23,056,384**, its **53,248** new actions are
all from boot, with consistent origin counters. The source model hash remains
`4983ca8d163732e2d21c4f3357a27746c7fb27bc31e03518339e5a94c3b46929`.

Meanwhile, the original balanced run's 25,251,840 validation completes all
20 games, mean **103.7**, median **67**, best **211**, highest level **4**,
reach counts **10/5/2/0**. This is stronger than the matched unprotected
control's **76.45 / level 2**, but still not a new full-game depth milestone
or a replacement for the stronger audited reference. Both balanced trials
continue; no fresh final-test games have been used.

The smaller-rate run's first validation, **23,252,992**, completes all 20
games: mean **121.65**, median **124**, best **313**, highest level **5**,
reach counts **16/4/1/1**. The matched 5e-5 run scored 113 and reached level 4.
The 313-point game is seed **10005**, 12,206 actions, normal GAME OVER.
This is a promising early comparison but not a new depth milestone or a
stronger primary level-reach count than the audited reference.

Model SHA-256: `ec84a0b18295b9002529b66976860af46ba3664f157c29b65570206f62d2b596`.
The [raw primary suite](results/level10/validation-balanced-lr1e5-23m25.json)
is archived byte-identically. Audit this frozen checkpoint on the separate,
already-reused validation seeds **10100-10149**, 50 ordinary complete games
requested with the 100,000-action diagnostic guard. Do not select from an
incomplete suite or use the untouched final-test seeds for this decision.

### Replay-based learning option reviewed, not implemented

[Self-Imitation Learning (Oh et al., 2018)](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)
uses the agent's own recorded state/action/return tuples and reinforces them
when their return exceeds the current value estimate. It is a possible way
to retain rare successful experience without external demonstrations, reward
shaping, or scripted actions. Its benefits are not universal: the
[supplement's Atari table](https://proceedings.mlr.press/v80/oh18b/oh18b-supp.pdf)
reports lower Breakout performance for A2C+SIL than A2C (452.0 versus 501.6),
and the paper's PPO experiments are on MuJoCo, not Breakdown.

Treat SIL as a conditional future experiment, not an established fix here.
No replay-learning code or new training objective has been added. Prioritize
the smaller-rate checkpoint audit and the active matched trials. Any future
implementation must use only actual training experience, bound its memory,
handle incomplete returns honestly, preserve disabled-mode compatibility,
and keep validation/replay recordings/final-test data out of training.

### Smaller-rate audit and training progress

The first checkpoint's separate 50-game audit exited **0**, all games complete:
mean **111**, median **109**, best **287**, highest level **5**, reach counts
**34/8/4/1**. The retained shared reference has **135.12 / 121 / 304 / level 5**
and **39/15/8/2** on that same reused set. Raw game order, completion, scores,
medians, and reach counts were independently checked in the
[validation audit](results/level10/balanced-lr1e5-first-validation-audit.json).
Do not promote this checkpoint as an overall improvement.

Its selected [313-point full replay](results/level10/balanced-lr1e5-23m25-validation-replay.html)
exactly reproduces primary seed 10005: 12,207 frames / 12,206 actions,
normal GAME OVER, no truncation. Level 5 begins at frame **11,712**, score
**279**; the game earns **34** more points over **494** actions there. This
describes one selected game, not a level-5 performance estimate. Replay SHA-256:
`ba3c38ab9f47799c2f68ae2b9c1f6a7634da3ed30db66ed9133e327ad8bb8123`.
The neural checkpoint reproduces all recorded actions with **zero mismatches**;
see the [replay verification](results/level10/balanced-lr1e5-first-replay-verification.json)
and [policy inspection](results/level10/balanced-lr1e5-23m25-policy-inspection.json).

| Smaller-rate PPO actions | Complete games | Mean | Median | Best | Highest level | Reach counts 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 23,252,992 | 20/20 | 121.65 | 124 | 313 | 5 | 16/4/1/1 |
| 23,502,848 | 19/20 — disqualified | — | — | — | — | — |
| 23,752,704 | 20/20 | 96.20 | 65.5 | 257 | 4 | 10/2/2/0 |
| 24,002,560 | 20/20 | 141.75 | 127.5 | 282 | 4 | 14/7/6/0 |
| 24,252,416 | 20/20 | 133.25 | 118 | 292 | 5 | 15/5/3/1 |
| 24,502,272 | 20/20 | 111.35 | 106 | 208 | 3 | 15/4/0/0 |
| 24,752,128 | 20/20 | 103.30 | 116 | 133 | 2 | 17/0/0/0 |
| 25,001,984 | 20/20 | 135.80 | 123 | 272 | 4 | 15/6/4/0 |

At 23,502,848, seed **10001** was waiting to serve at level **1** / score
**59** after 100,000 actions. The
[raw incomplete suite](results/level10/validation-balanced-lr1e5-23m5-disqualified.json)
is archived exactly and cannot select a model; its partial mean is not a
full-suite result.

During actual training, worker **29** completed its initial from-boot game at
**23,549,758**, score **334**, displayed level **6**, after **17,082** actions.
It entered level 5 at 23,503,102 and level 6 at 23,541,086. The game starts at
level 1 / score 0 / boot timing 4,347; it has no restored source, no prior reset,
and normal termination. Its exact per-worker action alignment is verified in
[the training-game audit](results/level10/balanced-lr1e5-full-training-level6.json).
This is genuine from-boot training progress, **not a frozen-policy milestone**:
the learner updated its weights throughout the episode.

To investigate transfer, the later complete-primary checkpoint at **23,752,704**
was audited on the same 50 secondary seeds. Model SHA-256:
`960b4ca44e8aad1be53e01afefdb6453b048ea93a0d82f06b9d71212ca3927db`.
The audit exited **0**, all 50 complete, mean **104.32**, median **81.5**, best
**279**, highest level **4**, counts **27/9/4/0**. It does not reproduce the
training level 6; [raw results](results/level10/validation-audit-balanced-lr1e5-23m75.json)
remain validation-only. The one-million-action primary check later improves
average score to 141.75 but still reaches only level 4. Continue learning and
keep the frozen performance reference unchanged. Final-test seeds are unused.

### Fixed checkpoint-averaging diagnostic

The gap between changing-policy training depth and frozen validation motivates
one inexpensive, fixed averaging check, inspired by
[stochastic weight averaging](https://arxiv.org/abs/1803.05407). This is an
exploratory PPO adaptation, not a demonstrated Breakdown improvement.
Uniformly average the smaller-rate run's last five scheduled checkpoints:
**24,002,560; 24,252,416; 24,502,272; 24,752,128; 25,001,984**. The window and
equal weights are fixed before inspecting the final tranche validation; do
not filter inputs by their validation scores. Validate the resulting single
neural policy on the usual 20 primary games. It adds no training actions and
does not mix or override actions at play time.

The standalone `rl.average` utility requires matching PPO run configurations,
compatible finite FP32 parameters, distinct inputs, and a new output path.
It records source hashes and marks its result **evaluation-only**: there is no
optimizer/RNG state for training resume and no inherited performance claim.
The active learners and all existing model/evaluation behavior are unchanged.

The diagnostic finished successfully: **20/20 complete**, mean **112.25**,
median **108**, best **262**, highest level **4**, counts **14/3/1/0**.
The [raw suite](results/level10/validation-balanced-lr1e5-average.json) identifies
model SHA-256 `5e743982b17d4d61fe91ec836bdbeb9a2aeeedc9be8a5c041d6dc745bb3c02cc`.
The exact fixed input window was used, with no validation-score filtering.
Do not promote it or resume from it. The CLI now explicitly rejects
evaluation-only resume before loading the GPU backend or creating a run.
All **71 tests** pass, including averaging compatibility/immutability and
the early resume rejection.

### Smaller-rate tranche completion and full training level 7

The learner finished at **25,006,080**, exactly **2,002,944** new actions.
Seven of eight primary suites completed; their highest level was 5, and the
last result was **135.8 / 123 / 272 / level 4**. It completed **219 from-boot
training games**, **494 restored segments** (450 peer-sourced), and published
63 entries. All completed origins, new-score rewards, protected-worker roles,
and logged action totals were independently verified; there were no training
guard hits. Cumulative actions: **1,459,894 from boot / 543,050 restored**.

Protected worker **4** completed a from-boot game at **24,690,917**:
**score 411 / level 7**, 17,018 actions, normal GAME OVER, initial level 1 /
score 0 / boot timing 37,452. It entered level 5 at 24,558,437, level 6 at
24,590,181, and level 7 at 24,665,605. Its previous game ended at 24,146,341;
all episode and publication action offsets match exactly at 32 workers.
This worker never restores training snapshots. It is real **changing-policy
training progress**, not a frozen-policy level-7 result.

The [frontier audit](results/level10/balanced-lr1e5-frontier-audit.json) traces
the first restored entries through level 8 to their actual training parents.
No level 9 was reached. The
[completion audit](results/level10/balanced-lr1e5-completion.json) contains final
counts, checkpoint hashes, and the level-7 game's full alignment check.
The **1,001-line / 549,496-byte** raw log is archived byte-identically, SHA-256
`a30f6472d7323653ecd5d88f33034e2a31ea91a73904f02a023b71942b8e8ec6`.
The terminal stopped record and saved state agree; the original process was
absent when checked, but its closed session's exit code was not re-read.

### Pause the higher-rate comparison; extend the smaller-rate learner

The 5e-5 balanced run's last level-5 validation remains 23,752,704. Its next
**19 complete suites** miss level 5; its latest three reach only level 2.
It also remains at full-training level 5 and restored level 8. This supports
an adaptive pause, not a claim that more training can never work.
SIGTERM was handled cleanly: **exit 0**, stop **28,815,360**, target false,
**5,812,224** added actions. Its 23 validations comprise 22 complete suites
and one disqualified suite. All **943** full training games and **2,827**
practice segments were checked for origin/reward correctness; no training
guard fired and protected workers never restored. The
[pause audit](results/level10/balanced-pause.json) preserves final hashes and
counts; the **4,362-line / 2,330,130-byte** log is copied exactly.
The in-memory archives and unfinished trajectories are not saved by resume.

Continue the smaller-rate **final learner**, not the average, for an initial
ten-million-action extension subject to validation-based review:

```sh
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced-lr1e5-long \
  --resume runs/level10-curriculum-balanced-lr1e5/latest \
  --additional-steps 10000000 --learning-rate 0.00001 --entropy 0.001 \
  --curriculum-probability 0.95 --curriculum-min-level 5 \
  --curriculum-per-level 8 --curriculum-share --curriculum-boot-envs 16 \
  --mlx-cache-mb 512 --eval-every 250000 --eval-games 20 --eval-envs 20 \
  --eval-seed 10000 --eval-max-steps 100000 --max-episode-steps 100000 \
  --target-level 10 --target-clears 1
```

Start **25,006,080**; source model SHA-256
`a68d536f2a5492e2a707e12ac8ccb0518de1b7b0a820e8ec879292c9291f2458`.
Model, optimizer, and learner RNG resume; all learning settings are unchanged.
The only config differences are source/run paths, source hash, and budget.
Archives and emulator games start fresh. All 32 workers omit MLX, the first
16 are protected, and initial action counters and finite updates are checked.
The first validation at **25,251,840** completes all 20 games: mean **133.3**,
median **116**, best **306**, level **5**, counts **15/7/3/1**. No new frozen
depth or general improvement is established. The stronger audited reference
and all frozen Level 5 artifacts remain unchanged; fresh final seeds are unused.

### Matched longer advantage-trace comparison

Keep the smaller-rate continuation running and compare **GAE lambda 0.99**
against its 0.95, from the same final 25,006,080 source and fresh archives.
This tests delayed credit assignment, not a different game reward or model.
[GAE equation 16](https://arxiv.org/html/1506.02438v6#S3) weights future TD
residuals by `(gamma * lambda)^delay`; increasing lambda trades lower
value-estimation bias for higher variance. It does not change the discounted
value-function objective when gamma is held fixed. This motivates an experiment,
not a claim that credit assignment is the proven cause of the plateau.

At gamma 0.995, the trace factor changes from 0.94525 to 0.98505. Actual traces
still stop at episode/life boundaries and the 128-action rollout boundary.
The earlier gamma-0.999 comparison changed the reward horizon instead; the
50k-action-timing comparison adjusted both factors to preserve temporal scales.
Neither isolated this lambda change at the original timing.

Use an initial **5,000,000** additional actions and review matched complete
primary suites. Keep learning rate 1e-5, entropy 0.001, gamma 0.995, 32 workers,
16 protected from-boot workers, all curriculum settings, and every evaluation
setting unchanged. No new replay loss, environment change, or action override.

```sh
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced-gae099 \
  --resume runs/level10-curriculum-balanced-lr1e5/latest \
  --additional-steps 5000000 --gae-lambda 0.99
```

The resume inherits all other saved settings. Confirm exact configuration
differences, worker roles, finite updates, and action origins after launch.
The existing GAE code requires no change; a delayed-reward analytic test checks
both lambdas and confirms that the longer trace cannot cross a terminal boundary.

The trial is active; the [launch verification](results/level10/balanced-gae099-launch-verification.json)
confirms the same model/optimizer/state source, exactly four configuration
differences versus the control (lambda, run path, and the two budget fields),
32 GPU-free emulator workers per run, 16 protected workers each, finite updates,
and exact logged action totals. All **72 tests** pass. The original learner's
four primary suites through 26,001,408 are complete, with means
**133.3 / 100.35 / 121.4 / 132.2**, highest levels **5 / 4 / 4 / 4**. It has
rediscovered restored training levels 6 and 7, not a new frozen-policy milestone.

The lambda-0.99 trial's first validation, **25,251,840**, completed all 20
games: mean **128.55**, median **118.5**, best **278**, highest level **4**,
counts **15/6/2/0**. The matched lambda-0.95 result was **133.3 / level 5**,
counts **15/7/3/1**. This first comparison does not favor the change; continue
both trials and review subsequent complete suites. The original continuation's
fifth suite at **26,251,264** is also 20/20 complete, mean **117.8**, median
**121**, best **200**, level **3**, counts **17/4/0/0**. Neither trial has
established a new frozen-policy depth milestone. Frozen model/replay integrity
and `git diff --check` still pass; the last commit remains `f4c77d6`, not pushed.

### Further training depth and a frozen longer-trace audit

The unchanged continuation completed a new **from-boot level-6 game**, score
**348**, at **26,363,653**: protected worker **4**, 18,240 actions, boot timing
193,963, normal GAME OVER. It entered level 5 at 26,295,941 (score 286) and
level 6 at 26,344,165 (score 326). The
[game audit](results/level10/balanced-lr1e5-long-full-training-level6.json)
checks zero-score boot origin and exact offsets from the worker's previous
terminal episode. Its policy changed during the game.

That run also rebuilt restored level 8 at **26,564,190**. Worker 29 started
from the level-5 snapshot discovered in the above full game, then traversed
levels 5-8 in one restored segment, ending at **530** after 6,753 actions:
**244 newly earned points**, not 530 points of new reward. The
[frontier audit](results/level10/balanced-lr1e5-long-frontier-audit.json) checks
source episode alignment and parent origins for first entries 5-8. This is
multi-level practice progress, not a from-boot or frozen-policy level-8 result.

| Continuation PPO actions | Complete | Mean | Median | Best | Highest | Counts 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 26,501,120 | 20/20 | 142.25 | 127.5 | 257 | 4 | 19/5/2/0 |
| 26,750,976 | 20/20 | 135.45 | 117.5 | 274 | 5 | 17/7/3/1 |

The lambda-0.99 comparison's next validations are also complete:

| Longer-trace PPO actions | Complete | Mean | Median | Best | Highest | Counts 2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| 25,501,696 | 20/20 | 103.65 | 117.5 | 219 | 3 | 12/3/0/0 |
| 25,751,552 | 20/20 | 126.50 | 123 | 276 | 4 | 14/4/2/0 |

During training, protected worker **2** completed a from-boot **level-7 /
381-point** game at **25,790,627**, after 14,926 actions, boot timing 79,372,
normal GAME OVER. It entered levels 5/6/7 at **25,683,747 / 25,713,955 /
25,774,403**. The previous worker game ended at 25,312,995; all action offsets,
score-only reward, and full-game provenance are verified in the
[training-game audit](results/level10/balanced-gae099-full-training-level7.json).
This is changing-policy training, not proof that any single checkpoint reaches 7.

Audit the nearby frozen **25,751,552** checkpoint on the already-reused
secondary seeds **10100-10149**, 50 games with the 100,000-action diagnostic
guard and 20 workers. This is a training-motivated transfer diagnostic, not an
assertion that its primary result beats the reference. Source model SHA-256:
`02ca26d14eca7afaf294abdedcf1a519fcf3df1065c6c9a56c7860dd0f634cc3`.
The [primary suite](results/level10/validation-balanced-gae099-25m75.json) is
archived byte-identically. The secondary audit is in progress; leave both
learners running. Final seeds 40000-40099 remain unused.

The secondary audit subsequently exited **0**, **50/50 complete**: mean
**140.1**, median **121.5**, best **286**, highest level **5**, reach counts
**44/13/9/3**. Level-5 seeds are **10100 / 10118 / 10143**, scores
**269 / 286 / 281**. The retained reference's same-set result is
**135.12 / 121 / 304 / level 5**, counts **39/15/8/2**. This is an improvement
in this reused set's mean and level-5 count, not demonstrated fresh-set
generalization. No frozen level 6 was reached.

All primary/secondary seed orders, termination flags, means, medians, best
scores, and reach counts for candidate and reference were independently
recomputed in the [comparison audit](results/level10/balanced-gae099-first-validation-audit.json).
Across the combined 70 reused games, the candidate has mean **136.21** and
counts **58/17/11/3**, versus reference mean **133.74** and **56/20/11/4**.
That mixed result does not justify an overall promotion. Preserve both models
and continue the planned matched training; these are descriptive reused-set
comparisons, not a new independent test.

The [286-point full replay](results/level10/balanced-gae099-25m75-secondary-replay.html)
reproduces seed **10118** exactly: 13,352 frames / 13,351 actions, normal
GAME OVER, no action cap. It enters level 5 at frame **13,170**, score **282**,
and earns **4** more points over **181** actions. All seeded neural actions
match with **zero mismatches**. See the
[replay verification](results/level10/balanced-gae099-first-replay-verification.json)
and [policy inspection](results/level10/balanced-gae099-25m75-policy-inspection.json).
Replay SHA-256:
`d9a2a1304dde5177a3e960ea6025f3b219f8cb9e01b4993f77c41559b1f80b03`.
It is a selected validation game, not training input or a performance estimate.
The original hosted replay and both frozen milestone packages are unchanged.

Both learners remain active. The original continuation's **27,000,832**
primary is 20/20 complete, **138.5 / 127.5 / 273 / level 4**, counts
**18/7/2/0**. The longer-trace trial's **26,001,408** primary is also 20/20,
**132.3 / 116 / 244 / level 4**, counts **18/5/2/0**. Continue the existing
budgets and compare subsequent complete games; no final model is selected and
the fresh 40000-40099 test set remains untouched.

### Optional self-imitation preparation

The longer-credit and original continuation trials are still running; do not
replace them mid-comparison. Their gap between changing-policy training depth
and frozen results motivates preparing the previously deferred SIL option.
It remains an experiment, not a diagnosis or claimed fix. The earlier caveat
about the paper's weaker A2C+SIL Breakout result still applies.

The new `rl.sil` path is disabled by default and can only collect directly from
the training step loop. It uses the agent's own raw screen/action/score data,
with no replay-file loading and no evaluation/recording hooks. A bounded suffix
is committed only at a real learning terminal; all truncations are discarded.
Returns use only newly earned score, not a restored snapshot's starting score.
No network architecture, emulator behavior, or inference action rule changes.
See [implementation controls](TRAINING.md#optional-own-experience-self-imitation).

The first GPU test caught uncaptured state when an auxiliary gradient and update
were split across compiled functions. The auxiliary path now uses ordinary
lazy MLX operations; the original compiled PPO path is unchanged. Tests cover
exact returns, suffix bounds, truncations, worker/life isolation, input ownership,
prioritized replay, detached actor advantages, one-sided critic gradients,
all-zero-batch Adam skipping, and interleaved PPO/SIL optimizer resume.

All **83 tests** pass. Three real emulator smokes also exited 0:

- Disabled, 65,536 actions: model/optimizer byte-identical to the archived
  reserved-boot probability-one smoke, with identical learner RNG/counters and
  selection state. No SIL RNG, buffers, or logs were introduced.
- Enabled, 65,536 actions: 240 auxiliary updates from 39 real learning-terminal
  suffixes; 2,706 from-boot and 2,176 restored transitions committed. All 39
  visible-score differences match newly earned reward. Replay/pending bounds,
  finite logged losses, worker origins, and exact action accounting pass.
- Truncation, 4,096 actions with a 16-action episode guard: all 256 cut-off
  suffixes discarded, zero replay entries and zero SIL updates. No unfinished
  suffix is turned into a false terminal.

All use deliberately incomplete 12-action validation, select no model, and
are implementation diagnostics only. Their checkpoints will not seed a real
comparison. Full raw logs are archived byte-identically; see the
[smoke audit](results/level10/sil-smoke-verification.json) and
[disabled compatibility](results/level10/sil-disabled-compatibility.json).
Neither active learner has SIL enabled, and no real SIL performance trial has
started. These checks establish implementation behavior, not improved play.

### Stronger frozen candidate at 28.5M

The unchanged smaller-rate continuation produced a stronger primary result at
**28,504,064**: all 20 games complete, mean **166.7**, median **145.5**, best
**294**, highest level **5**, reach counts 2/3/4/5 **18/10/5/2**.
Frozen model SHA-256:
`285278f47ad46d19c44088abc070673d38e2735a5978ce8e32d7d7ac055b5c1f`.

Its secondary audit exited 0, all **50/50 complete**: mean **141.18**, median
**124**, best **327**, highest level **5**, counts **42/16/8/4**. Level-5 seeds
are **10109 / 10125 / 10130 / 10131**, scores **294 / 327 / 294 / 267**.
The previous shared reference scores **135.12 / 121 / 304**, counts
**39/15/8/2** on this set. Across the combined 70 reused games, the candidate
has mean **148.47** and counts **60/26/13/6**, versus **133.74** and
**56/20/11/4**. All individual seed orders, termination fields, summary
statistics, source hashes, and primary file identity are independently verified
in the [comparison audit](results/level10/balanced-lr1e5-long-28m5-validation-audit.json).

Use this checkpoint as the new **validation comparison reference**, retaining
all earlier candidates. Both sets have been reused for selection; the result
does not establish independent generalization. It is still level 5, not 6 or
10, and the untouched final test must wait for the planned depth gate.

The [327-point full replay](results/level10/balanced-lr1e5-long-28m5-secondary-replay.html)
matches seed **10125** exactly: 16,789 frames / 16,788 actions, normal GAME OVER,
no action cap. It enters level 5 at frame **16,159**, score **292**, then earns
**35** points over **629** actions. See the
[decoded-screen verification](results/level10/balanced-lr1e5-long-28m5-replay-verification.json).
All **16,788** seeded neural actions reproduce with **zero mismatches** in the
[policy inspection](results/level10/balanced-lr1e5-long-28m5-policy-inspection.json),
which also exited 0. No action was overridden to produce the replay.
Replay SHA-256:
`ed9491bd8aa95ebac8d37102b4c75165c47d1570938adb5d7b76beacf58ad7a2`.
This selected game is neither training input nor a performance estimate.

Subsequent primary results remain variable. All rows below are 20/20 complete;
they do not replace the frozen candidate merely because they are newer.

| PPO actions | Original continuation mean / median / best / level; reach counts 2/3/4/5 | Longer GAE mean / median / best / level; reach counts 2/3/4/5 |
|---|---|---|
| 26,251,264 | 117.8 / 121 / 200 / 3; 17/4/0/0 | 146.7 / 120.5 / 281 / 4; 15/8/5/0 |
| 26,501,120 | 142.25 / 127.5 / 257 / 4; 19/5/2/0 | 131.4 / 116 / 320 / 5; 16/5/3/1 |
| 26,750,976 | 135.45 / 117.5 / 274 / 5; 17/7/3/1 | 101 / 96 / 252 / 4; 13/2/1/0 |
| 27,000,832 | 138.5 / 127.5 / 273 / 4; 18/7/2/0 | 134.1 / 120.5 / 268 / 5; 15/8/3/1 |
| 27,250,688 | 131.95 / 125 / 279 / 4; 16/6/2/0 | 111.35 / 104.5 / 244 / 4; 17/3/1/0 |
| 27,500,544 | 148.55 / 125 / 286 / 4; 17/6/5/0 | 106.6 / 85.5 / 248 / 4; 10/4/1/0 |
| 27,750,400 | 142.05 / 124.5 / 294 / 5; 16/6/4/2 | 102.25 / 108.5 / 272 / 4; 12/2/1/0 |
| 28,000,256 | 112.3 / 113.5 / 242 / 4; 15/4/1/0 | 92.45 / 86 / 195 / 3; 12/1/0/0 |
| 28,250,112 | 109.3 / 119 / 264 / 4; 14/2/1/0 | 98.85 / 93 / 241 / 4; 12/2/1/0 |
| 28,504,064 | 166.7 / 145.5 / 294 / 5; 18/10/5/2 | 96.7 / 83.5 / 239 / 4; 10/2/1/0 |
| 28,753,920 | 121.65 / 111.5 / 250 / 4; 12/6/4/0 | 130.75 / 120.5 / 291 / 5; 15/5/2/1 |
| 29,003,776 | 116.65 / 98 / 278 / 5; 12/5/2/1 | 101 / 85.5 / 266 / 4; 10/3/1/0 |
| 29,253,632 | 130.25 / 120.5 / 285 / 5; 17/5/2/1 | 98.45 / 87.5 / 330 / 5; 11/2/1/1 |
| 29,503,488 | 139.1 / 123 / 293 / 5; 14/5/5/2 | 107.25 / 113.5 / 195 / 3; 15/4/0/0 |
| 29,753,344 | 127.55 / 108 / 282 / 5; 13/5/4/2 | 97.85 / 103.5 / 187 / 3; 12/2/0/0 |
| 30,003,200 | 132.9 / 123 / 296 / 4; 13/5/4/0 | 80.45 / 63.5 / 179 / 3; 8/1/0/0 |

Continue the existing budgets, monitor genuine full-game depth, and keep
self-imitation disabled until starting a separately identified comparison.
Fresh final-test seeds **40000-40099** and the frozen milestone files remain
untouched; the public replay has not been changed.

The longer-trace trial also rebuilt restored level-8 practice. Its first entry
is worker 28 at **26,383,005**, score **512**, from a same-run level-7 start
(worker 21/action 33,217, score 369, boot offset 79,372). It ends at
**26,392,189**, score **513 / level 8**, with **144 newly earned points** over
4,206 actions. Source fields and exact publication-to-end step alignment pass
in the [practice audit](results/level10/balanced-gae099-level8-practice-audit.json).
This is restored training, not a full-game level-8 milestone.

### Predeclared self-imitation comparison

A [current two-run comparison chart](results/level10/balanced-trace-comparison.png)
and its [source records](results/level10/balanced-trace-comparison.json) show
the original continuation through primary **30,752,768** and the finished
longer-GAE tranche through **30,003,200**. Lines are from-boot changing-policy training only;
dots are complete frozen primary suites. Restored practice is excluded.
The x-axis includes the shared 500,000-action DQN initialization, so it is
500,000 above the PPO-only counters used in these notes. Stars mark level 5,
not the level-10 goal. This is a timestamped chart snapshot, not live telemetry.
Its random-baseline line is the historical 50-game set (seeds 20000-20049,
mean 1.88), not a newly evaluated or seed-matched comparison.

When the active five-million-action longer-trace tranche is reviewed, use
that freed capacity for SIL. Predeclare the source as the same original
**25,006,080** resumable learner (`level10-curriculum-balanced-lr1e5/latest`,
model `a68d536f2a5492e2a707e12ac8ccb0518de1b7b0a820e8ec879292c9291f2458`)
so it can be compared at matched action counts against the unchanged
continuation. Use fresh own-training curriculum/replay buffers, lambda 0.95,
all other control settings unchanged, plus four SIL updates per rollout,
batch 512, capacity 32,768, suffix limit 2,048, loss weight 0.1, value weight
0.01, priority alpha 0.6/beta 0.1. Initial budget: five million new actions.
This is a method comparison, not a continuation of a diagnostic smoke or an
immediate replacement of the stronger frozen 28.5M reference. Check origins,
finite updates, memory, and first complete validations before extending it.

### Longer-trace best-primary audit

After 15 complete primary suites through **28,753,920**, rank this trial's
checkpoints by the existing complete-game level-reach counts (levels 10 down
to 2, then mean). The selected checkpoint is **27,000,832**: mean **134.1**,
counts **15/8/3/1**, model SHA-256
`74444a0ed2f13a1f780cf9eff6645d3f3e5d57562f985a631869d29ecfb49cc7`.
Its [selection record](results/level10/balanced-gae099-27m-audit-selection.json)
was saved before the new secondary audit. The primary raw file is copied
byte-identically. Evaluate the existing 50 secondary seeds **10100-10149**,
not the fresh final set. No outcome is presumed and the stronger 28.5M original
continuation remains the reference. The earlier 25.75M audit was selected
near the training level-7 event; this candidate instead tests this trial's
currently strongest primary depth-consistency ranking.

Meanwhile, both learners remain active. Original-continuation **29,503,488**
is 20/20 complete, **139.1 / 123 / 293 / level 5**, counts **14/5/5/2**.
Longer-trace **28,753,920** is 20/20 complete, **130.75 / 120.5 / 291 / level 5**,
counts **15/5/2/1**. Subsequent checkpoints may change the within-trial ranking;
do not treat this ongoing review as the tranche's final selection.

The secondary audit subsequently exited **0**, all **50/50 complete**: mean
**119.38**, median **116**, best **290**, highest level **5**, counts
**38/8/4/3**. Level-5 seeds are **10107 / 10118 / 10138**, scores
**290 / 279 / 285**. Independently recomputed seeds, termination flags,
statistics, source hashes, and primary file identity are in the
[audit](results/level10/balanced-gae099-27m-validation-audit.json).
Across both reused sets, mean is **123.59**, counts **53/16/7/4**;
the original continuation's 28.5M reference has **148.47**, **60/26/13/6**.
Do not promote this candidate or produce another selected replay merely for
an unchanged depth. Preserve the checkpoint and raw evaluation; the final
test set remains unused. The shorter training tranche still has roughly
one million actions remaining and continues unchanged.

### Longer-trace tranche completed; self-imitation launched

The longer-trace learner subsequently exited **0** at **30,007,296**, after
**5,001,216** new actions (the requested budget rounded to a whole rollout),
wall time **6,196.17 seconds**, target false. All **20/20 primary suites**
finished all 20 games. Four checkpoints reached level 5, totaling four such
games across those reused suites, versus **eight checkpoints / twelve games**
for the original continuation at the same 20 action counts. These repeated
suites describe the training comparison, not independent final-test trials.
The last suite is **80.45 / 63.5 / 179 / level 3**, counts **8/1/0/0**.
The best primary-ranked checkpoint remains **27,000,832**, already audited
and not promoted. No new frozen-policy depth was established.

The finished run contains **666 from-boot training games** (highest 7) and
**1,937 restored practice segments** (highest 8), with **zero truncations**.
All origins, own-score rewards, practice source references, protected-worker
roles, logged action totals, and primary seed/termination/mean/reach counts
pass the [completion audit](results/level10/balanced-gae099-completion.json).
The complete raw log is archived byte-identically: **3,241 lines / 1,761,767
bytes**, SHA-256
`4d34b946f9c91c29e4fbeed1cb3e2d87b2ff1e48f7fab32ef6564915f844eb9f`.
Final resumable model SHA-256:
`620b7cb0bb00e345977acc9b588f3bf9a09afd2c34535e2863adc383acd673e5`.
Optimizer and state hashes are recorded in the audit. Preserve all candidates;
do not extend this setting now or substitute its final weights for the reference.

The freed compute slot now runs the predeclared own-experience SIL comparison:

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced-sil \
  --resume runs/level10-curriculum-balanced-lr1e5/latest \
  --additional-steps 5000000 --gae-lambda 0.95 \
  --sil-updates 4 --sil-batch-size 512 --sil-capacity 32768 \
  --sil-suffix-steps 2048 --sil-loss-weight 0.1 --sil-value-weight 0.01 \
  --sil-priority-alpha 0.6 --sil-priority-beta 0.1
```

Source model/optimizer/RNG is the same **25,006,080** learner used by both
earlier comparisons, not a diagnostic checkpoint. Other control settings are
inherited unchanged: learning rate 1e-5, entropy 0.001, gamma 0.995, lambda
0.95, 32 workers/16 protected from boot, 128-step rollout, batch 256/four PPO
epochs, own shared level-5-plus curriculum. Training archives and SIL buffers
start empty. Primary validation stays 20 games on seeds 10000-10019 every
250,000 actions, with the existing 100,000-action guard and level-10 gate.

Initial [launch verification](results/level10/balanced-sil-launch-verification.json)
through **25,092,640** confirms exact source hashes and only the declared
configuration differences, all 32 CPU-only workers and their intended reset
roles, 56 checked learning-suffix reward sums, finite logged PPO/SIL losses,
bounded replay/pending storage, and exact logged action accounting. At
25,088,000 there are **76 applied auxiliary updates**, replay size 32,768,
pending screens 135,790,592 bytes, and MLX peak **1,577,597,648 bytes**.
All experience so far is newly generated from boot; restored SIL collection
must be checked once this run discovers and reuses later-level entries.
No complete SIL validation is available at this launch audit, and no improvement
is claimed. The original continuation remains active and unchanged.

Before launch, two memory samples showed no new swap-outs and the system's
memory-pressure query reported 37% free; the run replaces the finished learner,
not a third simultaneous training job. Keep monitoring memory and throughput
as replay fills. No unrelated applications were stopped.

The original continuation's **30,253,056 / 30,502,912 / 30,752,768 / 31,002,624**
primaries are all 20/20 complete, means **138.75 / 117.35 / 99.15 / 126.9**,
highest levels **5 / 4 / 4 / 5**, with level-5 counts **2 / 0 / 0 / 1**.
The 28.5M frozen checkpoint remains the reference. Level 10 is not reached,
and the fresh final set plus all committed milestone files remain untouched.

The first SIL primary at **25,251,840** subsequently completed **20/20**:
**98.6 / 89.5 / 264 / level 4**, counts **11/2/1/0**. The matched original
checkpoint is **133.3 / 116 / 306 / level 5**, counts **15/7/3/1**. Raw primary
JSON is copied exactly and seed/termination/statistics plus saved SIL RNG and
non-persisted replay flags are verified in the
[first-validation audit](results/level10/balanced-sil-first-validation-audit.json).
This first result is weaker, not a positive performance claim; continue the
declared trial and assess subsequent matched suites. No secondary audit or
new selected replay is justified by this result.

The original continuation's first training guard occurs at **30,961,224**,
worker 7, boot offset 75,592: score **189 / level 3**, waiting to serve after
100,000 actions, `terminated=false`, `truncated=true`. It is excluded from
complete-game performance. The learner resumes ordinary training; no scripted
serve is introduced, and this guard does not satisfy any depth gate.

### First own restored SIL experience verified

The SIL run's first level-5 entry is worker **29**, action **11,824**, global
step **25,384,446**, score **287**, boot offset **114,870**. It was generated
from boot by this run's own learned policy and shared with its 31 peers. That
full training game ends at **25,390,302**, score **291 / level 5**, after
12,007 actions. Its weights changed during play, so this is not a frozen
level-5 validation result.

The first restored episode is worker **20**, ending at **25,389,557**, score
**288**, 137 actions from that score-287 snapshot. SIL splits it at the real
learning boundaries: an **118-action** suffix earning **one** point and a
**19-action** suffix earning **zero**. The first discounted return is
**0.6149486215**, not 287 or 288; the stored starting score is not reward.

The [restored-collection audit](results/level10/balanced-sil-first-restored-audit.json)
through **25,467,278** verifies all **38 completed practice episodes** against
their same-run archive sources, exact per-life step coverage and origin flags,
and all **330 SIL suffix score sums**. All logged action totals, replay/pending
bounds, and losses pass. The latest included progress has **440 auxiliary
updates**, **10,494 restored transitions committed**, no discarded truncations,
and replay size 32,768. These are collection/learning checks, not proof of
better complete-game play. Continue the planned matched validation sequence.

### SIL reaches frozen level 5 and restored level 8

The next two primary suites both finish 20/20 games:

| PPO actions | Mean / median / best | Highest level | Reach counts 2/3/4/5 | Matched control mean / level |
|---|---|---|---|---|
| 25,251,840 | 98.6 / 89.5 / 264 | 4 | 11/2/1/0 | 133.3 / 5 |
| 25,501,696 | 127.2 / 117 / 285 | 5 | 14/5/3/1 | 100.35 / 4 |
| 25,751,552 | 150.5 / 125.5 / 303 | 5 | 16/7/6/2 | 121.4 / 4 |
| 26,001,408 | 113.75 / 114 / 242 | 4 | 14/3/2/0 | 132.2 / 4 |

The second checkpoint's level-5 game is seed **10009**, score **280**. Its
[raw primary](results/level10/validation-balanced-sil-25m5.json) and
[audit](results/level10/balanced-sil-first-level5-validation-audit.json) verify
all seeds, termination fields, statistics, and checkpoint hashes. The first
three results are mixed, not evidence that every SIL checkpoint is stronger.

The third frozen model is
`d670b488dabbb34645eac43a8c460eac72f98b233e1709f7bce4042c7073cbc0`.
Its level-5 games are seeds **10007 / 10017**, scores **254 / 303**. It ties
the retained reference's two primary level-5 reaches and adds one level-4
reach (six versus five), so it ranks higher by the existing depth-consistency
rule despite lower mean and level-2/3 counts. The
[selection record](results/level10/balanced-sil-25m75-audit-selection.json)
and exact [raw primary](results/level10/validation-balanced-sil-25m75.json)
are saved before evaluating the 50 reused secondary seeds **10100-10149**.
No secondary outcome is presumed and the stronger audited original model
remains the comparison reference until that audit is reviewed.

Meanwhile, the same run's own restored practice advances further:

- Worker **16** first publishes level 6 at **25,448,241**, score **327**, from
  the original level-5 archive. Its 1,473-action practice episode ends at
  **25,463,633**, score **339**, **52 newly earned points**.
- Worker **31** later starts at level 6, score **327**, from worker **26**,
  action **14,496**. One **5,308-action** practice episode publishes level 7
  at **25,562,560** and level 8 at **25,641,568**, then ends at **25,685,664**,
  score **528**, **201 newly earned points**. These are two entries in the
  same episode, not two separate 201-point successes.

The [frontier audit](results/level10/balanced-sil-first-frontier-audit.json)
checks exact publication-to-end offsets, same-run source references, starting
scores/levels and boot offsets, and newly earned reward for these first
reaches. All are restored training, not frozen or from-boot depth milestones.
No level 9/10 has been verified. Continue the unchanged training budgets and
review complete-game validation independently of this faster practice progress.

The third checkpoint's secondary audit subsequently exited **0**, all
**50/50 complete**: **129.32 / 119 / 284 / level 5**, reach counts
**40/10/7/1**. Its only level-5 game is seed **10126**, score **284**. The
retained reference has **141.18 / 124 / 327 / level 5**, counts **42/16/8/4**
on the same reused set. The
[comparison audit](results/level10/balanced-sil-25m75-validation-audit.json)
recomputes all seeds, completion flags, summary statistics and combined
results, and checks checkpoint hashes and exact primary-file identity.
Combined 70-game reused-set mean is **135.37**, counts **56/17/13/3**,
versus reference **148.47**, **60/26/13/6**. Do **not** promote this checkpoint
or produce another selected replay for the unchanged depth. Continue the
declared SIL trial: its primary comparisons improved after the first suite,
but it has not produced a stronger audited model. This comparison is against
the retained reference, not a same-budget secondary ablation.

The original continuation's **31,252,480 / 31,502,336 / 31,752,192** primary
suites are all 20/20 complete, means **115.7 / 137.25 / 109.9**, all highest
level 4. No final model is selected; both live learners continue, and fresh
final-test seeds **40000-40099** remain unused.

The fourth SIL primary at **26,001,408** is also 20/20 complete, mean
**113.75**, highest level **4**, below the matched control mean **132.2**.
At **26,030,080**, SIL has applied **996 auxiliary updates** and committed
**178,037 restored transitions**, with no truncated suffixes collected and
MLX peak still **1,577,597,728 bytes**. Performance remains mixed; continue
the planned comparison without promoting the earlier primary peak.
The original continuation's **32,002,048** primary is 20/20 complete,
**101.55 / 95 / 257 / level 4**, counts **11/3/2/0**. The reference is unchanged.

### SIL from-boot level 6 and replay-dose follow-up

The current SIL run's first complete from-boot level-6 training game ends at
**26,559,456**, worker **31**, score **317**, after **12,861 actions**, boot
offset **161,644**. It publishes level 5 at **26,518,400 / score 269** and
level 6 at **26,550,112 / score 309**. The
[episode audit](results/level10/balanced-sil-first-full-training-level6.json)
verifies normal boot origins, game-over completion, exact worker/publication
and learning-boundary offsets, and raw score reward. Weights changed during
play, so this is not frozen-policy level 6.

Its four learning suffixes retain rewards **41 / 8 / 25 / 44**. A dropped
8,564-action prefix accounts for the other **199 points**: only **118 of 317
points** are represented in the retained suffix transitions. The bounded
collector intentionally discards old prefix experience; it does not falsely
credit the inherited visible score or shorten retained states' true future
returns. Do not equate episode reward with replayed reward when a prefix
was dropped.

The next four primary suites are all 20/20 complete:

| PPO actions | Mean / median / best | Highest level | Reach counts 2/3/4/5 | Matched control mean / level |
|---|---|---|---|---|
| 26,251,264 | 102.95 / 109 / 196 | 3 | 14/2/0/0 | 117.8 / 3 |
| 26,501,120 | 140.35 / 115 / 312 | 5 | 15/6/4/2 | 142.25 / 4 |
| 26,750,976 | 108.05 / 105 / 193 | 3 | 15/4/0/0 | 135.45 / 5 |
| 27,000,832 | 106.45 / 95.5 / 246 | 4 | 13/4/2/0 | 138.5 / 4 |

The [progress audit](results/level10/balanced-sil-27m-progress-audit.json)
independently verifies all first eight matched suites' seed lists, completion
flags and summary statistics, with raw-file hashes. Performance remains
mixed and the best primary-ranked SIL checkpoint remains the already-audited
25,751,552 candidate. The reference is unchanged. The original continuation's
32,251,904 / 32,501,760 / 32,751,616 / 33,001,472 primaries also finish all
20 games: means **117.3 / 111.95 / 123.35 / 82.5**, highest levels
**5 / 4 / 5 / 2**, level-5 counts **1 / 0 / 1 / 0**. Continue the existing
action tranches; no final-test games are used.

A read-only [replay-retention audit](results/level10/balanced-sil-replay-retention-audit.json)
reconstructs FIFO insertion/eviction from committed suffix lengths through
26,631,851. Of 1,377 suffixes, 1,346 have already been fully evicted; their
median residence is **44,686.5 newly generated actions / 57.72 seconds**.
The successful worker-31 level-6-to-8 episode's two suffixes leave completely
after **42,502 / 54,611** new actions. Priorities influence sampling, not
retention. These are FIFO/nominal-batch measurements, not exact sample identities
or proof that useful experience was insufficiently learned.

With 32 workers times 128 actions, four replay batches of 512 provide **0.5
nominal replay draws per new action**. The original
[SIL supplement](https://proceedings.mlr.press/v80/oh18b/oh18b-supp.pdf) used
four 512-state batches per 80 new actions for Atari A2C+SIL (derived ratio
25.6), and ten per 2,048-step batch for PPO+SIL on MuJoCo (derived ratio 2.5).
These are different algorithms/tasks and episode conventions, not a directly
matched recommendation. The [paper](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)
used game-over, not life-loss, terminals for Atari; its Breakout result did
not improve. Our bounded life-terminal adaptation may behave differently.

Predeclare **20 instead of 4 SIL updates per rollout** (2.5 nominal draws/new
action), leaving capacity, suffix history, priority settings, loss weights,
PPO and curriculum unchanged. Use the same **25,006,080** source with exact
model/optimizer/state hashes rechecked, fresh buffers, and a five-million-new-
action tranche. This isolates replay optimization dose, not a capacity or
terminal-boundary change. It may reinforce suboptimal actions; success is
not presumed.

The [prelaunch protocol](results/level10/balanced-sil20-protocol.json) records
the exact command, source hashes, validation rules and checks. Start
`runs/level10-curriculum-balanced-sil20` only after the original long
continuation finishes and its results are reviewed, using that freed slot.
Do not add a third learner or change either live process. No new experiment
is running yet, and no training or validation data is imported into replay.

The separate [SIL comparison chart](results/level10/balanced-sil-comparison.png)
shows the original continuation through primary **33,251,328** and SIL
through primary **27,000,832**, alongside their from-boot rolling training
metrics. Restored practice is excluded. Stars mark complete frozen level-5
reaches; the dotted depth line is the level-10 target. The horizontal random
baseline is the historical 100-game Level-5-package baseline (mean 1.54),
not a newly sampled matched validation baseline. The x-axis includes the
earlier 500,000 DQN actions in each checkpoint lineage. Reproduce with:

```bash
venv/bin/python -m rl.report \
  runs/level10-curriculum-balanced-lr1e5-long \
  runs/level10-curriculum-balanced-sil \
  --target-level 10 --highlight-level 5 \
  --baseline results/level5/random.json \
  --output results/level10/balanced-sil-comparison
```

The chart's JSON/SVG companions retain the plotted validation records and
latest progress. Both training handles were verified live after this review;
memory pressure reported 43% free, and neither learner was restarted.

Eight subsequent bounded observer checks verified both real training handles
live and advancing, through approximately **33.57M / 27.46M** actions. The
observer then ended normally; neither learner was stopped. New complete
primaries are original **33,251,328: 119.85 / 116 / 273 / level 5**,
counts **15/3/2/1**, original **33,501,184: 121 / 115 / 278 / level 5**,
counts **12/5/3/1**, and SIL **27,250,688: 121.55 / 112 / 268 / level 5**,
counts **13/5/2/1**. All seed lists, termination flags and summary statistics
check out; these results do not improve the retained reference. A later
memory-pressure sample reports 37% free. The queued SIL20 trial is still
not running. Milestone-file integrity against `f4c77d6` remains unchanged.

### SIL's first three-reach primary selected for secondary audit

The **27,500,544** SIL checkpoint completes all 20 primary games:
**134.5 / 117 / 315 / level 5**, reach counts **14/6/3/3**. Its level-5
games are seeds **10000 / 10006 / 10009**, scores **315 / 281 / 306**.
Three level-5 reaches outrank the retained reference's two under the existing
depth-count rule, despite lower mean and intermediate-level reach counts.
It is the best primary-ranked checkpoint among this trial's first ten suites.

The [selection record](results/level10/balanced-sil-27m5-audit-selection.json)
verifies all ten primary seed lists, complete-game flags and statistics and
records exact checkpoint hashes before the secondary evaluation. The
[raw primary](results/level10/validation-balanced-sil-27m5.json) is preserved
byte-identically. Evaluate these frozen weights on the existing **10100-10149**
secondary seeds before considering any reference promotion. This is not a
new frozen depth or a final-test result. Both learners continue their declared
budgets; the queued SIL20 experiment remains unlaunched.

The secondary audit subsequently exited **0**, all **50/50 complete**:
**121.82 / 113 / 295 / level 5**, counts **35/11/6/1**. The sole level-5
game is seed **10140**, score **295**. The
[comparison audit](results/level10/balanced-sil-27m5-validation-audit.json)
recomputes all four candidate/reference primary/secondary summaries and
checks seed order, game-over flags, unchanged checkpoint hashes and exact
primary-file identity. Combined reused-set mean is **125.44**, counts
**49/17/9/4**, versus reference **148.47**, **60/26/13/6**. Do **not** promote
this candidate or record another selected replay for its unchanged depth.

Across the first ten matched primary checkpoints, SIL has **nine level-5
games at five checkpoints**, versus control **two games at two checkpoints**.
This is a favorable depth-consistency signal on repeatedly reused primary
seeds, not independent generalization evidence. The secondary comparison
above is against the stronger, later 28.5M reference, not a same-budget
secondary ablation. Preserve the candidate and finish the declared trial
before deciding on continuation. Level 10 remains unachieved and the final
test stays unused.

The next SIL primary at **27,750,400** finishes all 20 games:
**157.85 / 125 / 289 / level 5**, counts **18/8/6/1**. This is its best
primary mean so far, not its best depth-count rank; the already-audited
three-reach checkpoint remains that candidate. Original continuation
**33,751,040 / 34,000,896** also finishes 20/20, respectively
**119.25 / 115 / 277 / level 4**, counts **13/4/4/0**, and
**88.9 / 108.5 / 133 / level 3**, counts **12/1/0/0**. All seed/termination
and summary checks pass; no reference promotion follows.

Six further bounded checks confirm both actual learner handles live and
advancing, through roughly **34.17M / 27.98M**. Latest checked progress has
finite PPO/SIL losses and exact action-origin totals. At SIL **27,996,160**,
there are **2,916 auxiliary updates**, **802,409 restored transitions
committed**, replay size 32,768, pending 39,855 screens, zero discarded
truncations, and MLX peak **2,034,602,200 bytes**. The bounded observer ended
normally without stopping training. Committed milestone integrity still
passes; disk space is 29 GiB available, and SIL20 remains queued.

### First frozen level-6 candidate

The SIL **28,250,112** primary completes all 20 games:
**130.25 / 112.5 / 326 / level 6**, reach counts **12/6/4/2/1** for levels
2 through 6. Seed **10006** reaches level 6 and ends normally at score
**326** after **15,034 actions**, boot offset **72,093**. This is a frozen,
from-boot evaluation result, not restored practice or changing-policy training.
All earlier frozen candidates in this project reached at most level 5.

The [selection record](results/level10/balanced-sil-28m25-audit-selection.json)
checks seed order, completion flags, raw score rewards and summary statistics
and records checkpoint hashes. The
[raw primary](results/level10/validation-balanced-sil-28m25.json) is copied
exactly before running a 50-game secondary audit on **10100-10149** and an
uncapped replay of **10006**. Do not presume replay reproduction, secondary
transfer or reliable level-6 play before those checks finish. The level-10
goal and untouched final seeds remain unchanged. Review this new-depth
candidate before launching the queued SIL20 comparison.

The secondary audit exited **0**, all **50/50 complete**:
**146.4 / 124.5 / 337 / level 6**, counts **43/15/9/2/1** for levels 2-6.
Seed **10141** also reaches 6, score **337**, after **14,932 actions**;
seed **10108** reaches 5, score **276**. The
[comparison audit](results/level10/balanced-sil-28m25-validation-audit.json)
checks all four candidate/previous-reference sets, their statistics and
termination flags, unchanged checkpoint hashes, and exact primary identity.

The uncapped [primary replay](results/level10/balanced-sil-28m25-primary-level6-replay.html)
also exited **0**, **15,035 frames / 15,034 actions**. All primary-game fields
match; decoded screens start at score 0 / level 1 and finish at score 326 /
level 6 / GAME OVER. It enters level 6 at frame **14,709**, score **319**,
then plays **325 actions** and earns **seven** further points. The
[replay verification](results/level10/balanced-sil-28m25-replay-verification.json)
records all level entries and HTML SHA-256
`ad150bf0aaad7a2fdabb2adbafb791056c7ac7fc026c1ed26eef8d5622eb3a34`.
The [policy inspection](results/level10/balanced-sil-28m25-policy-inspection.json)
exited **0** with **zero seeded-action mismatches**, not just matching final
metadata. Neither the replay nor these validation observations are training input.

Promote this checkpoint to **depth reference**: complete level 6 is reproduced
on both reused suites. Across 70 reused games: **141.79 / 123.5 / 337**,
counts **55/21/13/4/2**. This is a rare first-reach result, not robust level-6
play. Preserve the older 28.5M original model for its higher combined mean
**148.47** and level-5 count **six versus four**; it has no level-6 games.
Do not claim all metrics improved. The new model SHA-256 is
`f58fb1f033105b148b8418d513f15c170c05c908f4594a64fb9b9b876c28e471`.

The new-depth review is complete. Continue the current SIL4 tranche so it
can exploit its live own-training archives. The queued SIL20 comparison
still uses the same predeclared 25,006,080 source for an interpretable dose
comparison, not the selected validation checkpoint; launch only after the
original continuation ends and its final tranche is reviewed. Level 10 is
still unachieved; final seeds **40000-40099** and committed milestone files
remain untouched. No remote push or public-hosting change was made.

The [level-6 comparison chart](results/level10/balanced-sil-level6-comparison.png)
shows the original continuation through primary 34,750,464 and SIL through
28,504,064, with the new frozen level-6 reach starred. It excludes restored
practice and adds the earlier 500,000 DQN actions to the lineage x-axis.
The historical 100-game random baseline (mean 1.54) is not a new matched
validation baseline. The plotted records are retained in the JSON companion.
Reproduce this separate chart with the prior two-run report command, changing
`--highlight-level 6` and output to `results/level10/balanced-sil-level6-comparison`.

### Original continuation completed; higher replay dose launched

The original unchanged continuation exited **0** at **35,008,512**, after
**10,002,432 new actions** (whole-rollout rounding), wall **12,148.65 seconds**,
target false. All **40/40 primary suites** completed all 20 games; **18
checkpoints / 23 games** reached level 5, none level 6. Its best primary-ranked
checkpoint remains **28,504,064**, already audited and retained as the
higher combined-mean comparison. The final scheduled primary at 35,000,320
is **119.4 / level 5**, counts **14/3/2/1**; the final resumable weights are
two rollouts later and are not substituted for that evaluated checkpoint.

The [completion audit](results/level10/balanced-lr1e5-long-completion.json)
verifies every full/restored episode's origins, exact worker/publication
offsets, raw score rewards, same-run archive sources, all logged action totals,
worker reset roles, and all primary seed lists/termination/statistics.
There are **1,086 complete from-boot training games**, highest 6, and **3,528
complete restored practice segments**, highest 8. The previously documented
100,000-action serve stall is the only training truncation and is excluded
from complete-game performance. Terminal origins total **6,296,842 from
boot / 3,705,590 restored** actions.

The full raw log is copied byte-identically: **5,909 lines / 3,218,815 bytes**,
SHA-256 `bba5bfe8f9e4753d6e28b0c77fd2046a60f31f8bca6cae20be8100a5c3ed19e5`.
Final resumable model SHA-256:
`aba565a247edc901bde29531e2bf3faca947d63aed4abc25ab520bcbf3d0c412`;
optimizer/state hashes are in the audit. No checkpoints were removed. A
process check confirmed the old learner was gone before starting its replacement.

The reviewed, predeclared dose comparison is now active:

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-curriculum-balanced-sil20 \
  --resume runs/level10-curriculum-balanced-lr1e5/latest \
  --additional-steps 5000000 --gae-lambda 0.95 \
  --sil-updates 20 --sil-batch-size 512 --sil-capacity 32768 \
  --sil-suffix-steps 2048 --sil-loss-weight 0.1 --sil-value-weight 0.01 \
  --sil-priority-alpha 0.6 --sil-priority-beta 0.1
```

The [launch audit](results/level10/balanced-sil20-launch-verification.json)
through **25,070,889** verifies exact common-source hashes, and only two
configuration differences versus SIL4: run directory and **4 to 20** updates.
All 32 emulator workers omit MLX; the first 16 are protected from restored
resets. All **52 checked suffixes** contain newly earned, correctly accounted
score rewards. Logged origin/storage accounting and losses pass. At
**25,063,424**, replay size is **25,871**, pending screens **31,473**, auxiliary
updates **260**, and MLX peak **1,577,597,648 bytes**. All collected experience
so far starts from boot; verify restored collection once this trial creates
and reuses its own archives. There is no complete SIL20 validation yet and
no performance improvement is presumed from successful initialization.

Both training handles are verified live after the handoff; memory pressure
reports 43% free. SIL4 continues with its existing training buffers. The new
trial starts with empty training buffers and does not import the selected
level-6 replay, validation states, or any archive file. The immutable prelaunch
protocol retains its original `predeclared_not_launched` status as a dated
record; this launch audit documents its subsequent execution. The depth
reference is still the verified SIL 28.25M model, and the final test remains unused.

The first SIL20 primary at **25,251,840** subsequently completes all 20 games:

| Same-source setting | Mean / median / best | Highest level | Reach counts 2/3/4/5 |
|---|---|---|---|
| No SIL | 133.3 / 116 / 306 | 5 | 15/7/3/1 |
| Four SIL updates | 98.6 / 89.5 / 264 | 4 | 11/2/1/0 |
| Twenty SIL updates | 114.2 / 108.5 / 286 | 4 | 15/4/2/0 |

The [first-validation audit](results/level10/balanced-sil20-first-validation-audit.json)
recomputes every setting's seed order, completion and summary statistics, and
verifies SIL20 checkpoint hashes plus saved SIL RNG / non-persisted replay
flags. The [raw primary](results/level10/validation-balanced-sil20-25m25.json)
is copied exactly. This initial result improves on matched SIL4, but not the
no-SIL control; it is not new depth or evidence of a generally superior
setting. Continue the declared comparison without a secondary audit or
reference promotion for this checkpoint. Both learners remain live; the
verified level-6 checkpoint and final-test reservation are unchanged.

### Conditional continuation from verified level 6

After the verified 28,250,112 checkpoint, the next four SIL4 primary suites
are all complete but reach only levels **4 / 4 / 3 / 4**:

| PPO actions | Mean / median / best | Reach counts 2/3/4/5 |
|---|---|---|
| 28,504,064 | 128.5 / 120.5 / 282 | 16/6/1/0 |
| 28,753,920 | 126.8 / 118.5 / 275 | 13/6/3/0 |
| 29,003,776 | 96.1 / 107 / 185 | 14/1/0/0 |
| 29,253,632 | 128.3 / 116 / 273 | 11/6/3/0 |

The saved level-6 model, optimizer and state hashes are rechecked unchanged.
Predeclare a [conditional continuation](results/level10/sil-level6-continuation-protocol.json)
from **28,250,112**, five million additional actions, all learning settings
unchanged, **only if** the remaining SIL4 tranche finishes without a stronger
audited depth candidate. Finish and audit the existing run first; if a stronger
candidate appears, review it and record the revised source decision. Do not
automatically prefer the last weights merely because they are later.

Use the existing SIL4 slot after terminal completion, keeping SIL20 unchanged.
Restore model/optimizer and both saved RNG streams, but start fresh curriculum,
replay and pending buffers as required by the current checkpoint format.
This restarts emulator trajectories; it is not uninterrupted training with
preserved buffers. No replay or validation observations are imported. The
conditional run is not active yet, and level 10 remains unachieved.

The second SIL20 primary at **25,501,696** is also 20/20 complete:
**96.3 / 103 / 190 / level 3**, counts **13/2/0/0**. Matched SIL4 is
**127.2 / 117 / 285 / level 5**, counts **14/5/3/1**; the no-SIL control is
**100.35 / 104 / 226 / level 4**, counts **11/3/1/0**. The
[two-checkpoint comparison](results/level10/balanced-sil20-first-two-primary-audit.json)
verifies seeds, completion and recomputed statistics for all six suites.
The higher dose has mixed early results; do not promote or additionally
audit either initial checkpoint on the secondary set.

A pre-completion SIL4 audit through **29,536,128** checks **3,506 learning
suffixes**, **1,943 full/practice episode suffix coverages**, exact pending
and dropped-prefix counts, raw scores, source references, and all logged
action totals. It contains **488 complete from-boot games / 1,455 complete
practice segments**, zero truncations, and **18 complete primary suites**.
The best depth checkpoint remains 28,250,112. This is not a terminal record;
finish the tranche before archiving and making its final continuation choice.

### SIL4 completed; verified-depth continuation launched

The initial SIL4 learner exited **0** at **30,007,296**, after **5,001,216**
new actions, wall **6,835.09 seconds**, target false. All 20 primary suites
completed all 20 games. Eight checkpoints reached level 5, totaling 13 such
games, including one primary level-6 game. The matched no-SIL control had
eight checkpoints / twelve level-5 games and no level-6 game. These are
repeatedly reused suites, not independent tests. The best depth checkpoint
remains **28,250,112**, already verified on both reused sets and in the replay.
The final primary is **97.25 / 63.5 / 288 / level 5**, counts **8/2/2/1**;
the last resumable weights are one rollout later and are not selected.

The [completion audit](results/level10/balanced-sil-completion.json) checks
**533 complete from-boot training games / 1,595 practice segments**, highest
levels **6 / 8**, zero truncations, **3,828 learning suffixes / 2,128 episode
coverages**, exact origins/rewards/pending counts and all primary statistics.
SIL applied **4,880 updates**; **35,565** pending transitions remain uncommitted.
The log is copied byte-identically: **6,777 lines / 3,457,807 bytes**, SHA-256
`1904da334dd0a47f53701eb82fe632c300822d9d56ccf1a6eeefbe0591902c96`.
Final resumable model SHA-256 is
`c737bbf14453c8cbe689b6edf41d1bc50bb02f1cbcf91e74f5d12289aae05022`;
all checkpoint files are preserved.

After final review and confirmation that the old process stopped, the
conditional continuation launched in its freed slot:

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-sil-level6-continuation \
  --resume runs/level10-curriculum-balanced-sil/step-028250112 \
  --additional-steps 5000000 --sil-updates 4
```

The [launch audit](results/level10/sil-level6-continuation-launch-verification.json)
through **28,375,261** verifies exact source hashes and only metadata/budget
configuration differences, all 32 CPU-only workers and 16 protected-boot
roles, and **55 initial suffix score checks**. At 28,372,992 it has **100
applied SIL updates**, replay size 32,768, pending 41,232 screens, finite losses,
exact logged action accounting and MLX peak **1,577,597,648 bytes**. No complete
validation was available at that launch check. Both RNG streams are present
in the source and restored by the unchanged resume path; buffers start empty.

This is an advancement run, not a same-source ablation of SIL20. Compare the
unchanged SIL20 trial against the archived original SIL4 tranche. Both current
learner handles are verified live; memory pressure reports 39% free. The dated
conditional protocol remains preserved, with this record documenting execution.
The depth reference and fresh final-test reservation are unchanged.

### Higher-dose own-source collection and first resumed validation

SIL20's third and fourth primary suites both complete all 20 games:
**126.85 / 106.5 / 290 / level 4**, counts **11/7/4/0**, at 25,751,552;
**100.2 / 109 / 192 / level 3**, counts **16/1/0/0**, at 26,001,408.
These do not establish a stronger frozen policy.

Its first own level-5 archive entry is generated by protected-boot worker
**3**, action **27,567**, global **25,888,196**, score **280**, boot offset
**167,765**. That full training game ends at **25,908,356**, **315 / level 5**.
Peer worker **18** completes the first restored episode at **25,914,163**:
532 actions, score **314**, exactly **34 newly earned points** from 280.
Worker **22** then publishes level 6 at **25,925,655** and ends that restored
episode at **25,937,911**, score **327**, 1,359 actions, **47 new points**.
These are own-policy practice episodes, not frozen/from-boot level-6 milestones.

The [restored-collection audit](results/level10/balanced-sil20-first-restored-audit.json)
through **26,061,918** verifies all **60 completed practice sources**, **776
learning suffixes / 267 episode coverages**, exact action offsets, raw-score
and pending/dropped-prefix accounting, with zero truncations. The starting
score is not replay reward, and no validation or recorded-game states enter
training. Later practice progress remains separate from performance evaluation.

The resumed Level-6 run's first primary at **28,504,064** completes all 20
games: **93.2 / 65 / 286 / level 4**, counts **8/2/1/0**. Its validated source
has mean 130.25 / level 6; the prior uninterrupted SIL4 checkpoint at the same
action count has mean 128.5 / level 4. The
[first-validation audit](results/level10/sil-level6-continuation-first-validation-audit.json)
checks all three sets, checkpoint hashes and saved SIL-state flags; the
[raw primary](results/level10/validation-sil-level6-continuation-first.json)
is copied exactly. Do not promote or additionally audit this weaker checkpoint.
Fresh worker trajectories and buffers differ from the uninterrupted run, so
one result does not prove a restart penalty or isolate a method change.
Continue the declared tranche with the original Level-6 reference preserved.

### Resumed own-source collection and a bounded decoding recheck

The resumed run's second and third primary suites complete all 20 games:
**104.85 / 108 / 261 / level 4**, counts **13/2/1/0**, at 28,753,920;
**150.8 / 123.5 / 315 / level 5**, counts **13/8/6/3**, at 29,003,776.
The latter recovers later-level play but does not exceed the frozen source's
depth. Keep the existing Level-6 reference and continue the tranche.

The [own-source collection audit](results/level10/sil-level6-continuation-first-restored-audit.json)
through **29,253,632** verifies **151 complete from-boot games / 43 practice
segments**, **575 learning suffixes / 194 episode coverages**, exact source,
score, action-offset, pending and dropped-prefix accounting, and zero guards.
The first level-5 publication comes from worker **21**, action **25,060**,
global **29,052,022**, score **276**, boot offset **74,517**. Its complete
from-boot training game ends at **29,068,982**, score **310 / level 5**.
Worker **17** completes the first restored segment at **29,063,282** with
score **277**, 164 actions and exactly **one newly earned point**. Subsequent
own practice reaches levels 6 and 7; these are not frozen/from-boot milestones.
No validation observations, archived replay or external states are imported.

SIL20's fifth through seventh primaries also complete all 20 games:

| PPO actions | Mean / median / best | Highest level | Reach counts 2/3/4/5 |
|---|---|---|---|
| 26,251,264 | 118.05 / 116 / 280 | 4 | 13/4/3/0 |
| 26,501,120 | 99.2 / 89 / 250 | 4 | 11/2/1/0 |
| 26,750,976 | 117.2 / 119.5 / 261 | 4 | 16/3/2/0 |

The [early validation audit](results/level10/sil-continuations-early-validation-audit.json)
checks all seven SIL20 and three resumed suites. No SIL20 primary has yet
reached level 5; matched original SIL4 has three such checkpoints by this
point. The higher dose is not promoted, and its declared tranche remains active.

A separate [argmax protocol](results/level10/argmax-sil-28m25-protocol.json)
was recorded before launching this diagnostic on the verified Level-6 model:

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-curriculum-balanced-sil/step-028250112/model.safetensors \
  --games 20 --seed 10000 --max-steps 50000 --envs 20 --deterministic \
  --output results/level10/argmax-sil-28m25-diagnostic.json
```

Earlier 10M and 20.5M models stalled in this mode. This recheck uses the newer
SIL-trained model without changing either live learner or sampled-validation
protocol. Every action remains the network's argmax, with no scripted serve.
The 50,000-action guard is diagnostic only: any incomplete game disqualifies
the entire suite from performance selection or the target. Final test seeds
40000-40099 remain unused. The diagnostic is active at this record; report its
terminal result separately rather than treating initialization as success.

The argmax diagnostic subsequently exits **1** as intended: **13 complete
games / 7 truncations**, all seven guards waiting for a serve. Seed **10013**
does finish at **486 / level 7**, 18,798 actions, but that does not rescue the
incomplete suite or qualify a selectable Level-7 milestone. The
[audit](results/level10/argmax-sil-28m25-audit.json) checks seeds, recomputed
statistics, model/result hashes, mode and disqualification. Do not report its
completed-only mean as performance across all requested games.

Before another diagnostic, record the [1% epsilon-greedy protocol](results/level10/epsilon001-sil-28m25-protocol.json).
At every action, independent of the screen or parsed game status, select a
uniform random action with probability **0.01**, otherwise the learned-logit
argmax. This is a separate stochastic policy, not sampled PPO and not a
serve-specific fallback. [Mnih et al. (2015)](https://doi.org/10.1038/nature14236)
used the generic epsilon-greedy evaluation mechanism at epsilon 0.05 for DQN;
this diagnostic adapts it at a preselected 0.01 to PPO-trained logits, not a
replication or a claim that this setting will improve Breakdown.

After the argmax evaluator exits, launch:

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-curriculum-balanced-sil/step-028250112/model.safetensors \
  --games 20 --seed 10000 --max-steps 100000 --envs 20 \
  --deterministic --epsilon 0.01 \
  --output results/level10/epsilon001-sil-28m25-diagnostic.json
```

All 20 games must complete; only a strictly stronger descending level-reach
rank than the sampled f58 primary warrants a 50-game secondary check in the
same epsilon-greedy mode. No training or sampled-validation settings change,
no game-specific action rule is added, and fresh final-test seeds remain
unused. The epsilon-greedy diagnostic is active at this record.

The epsilon-0.01 diagnostic finishes with exit **0**, **20/20 complete**:
**109.4 / 107.5 / 250 / level 4**, reach counts **17/2/1/0**. It has no serve
truncations, but its complete-suite depth rank is below the sampled Level-6
source. The [result audit](results/level10/epsilon001-sil-28m25-audit.json)
verifies seeds, termination, recomputed statistics, checkpoint/result hashes,
mode and rank. Do not promote this setting or run its secondary audit.

One [predeclared lower-dose counterpart](results/level10/epsilon0001-sil-28m25-protocol.json)
then tests **epsilon 0.001**, otherwise the identical command, checkpoint,
primary seeds and guard, with output `epsilon0001-sil-28m25-diagnostic.json`.
It finishes with exit **0**, **20/20 complete**, **129.05 / 118 / 260 / level 4**,
counts **15/6/3/0**. Seed 10013 eventually leaves its long serve wait and ends
at 212 / level 3, rather than the isolated pure-argmax Level-7 outcome.
The [result audit](results/level10/epsilon0001-sil-28m25-audit.json) verifies
seeds, complete-game statistics, hashes, mode and the weaker rank. Neither
exploration rate merits promotion or a secondary audit. End this bounded
decoding comparison, keep sampled neural decoding, and continue both training
tranches. No source code, live policy protocol or committed package changed.

### Conditional replay-capacity comparison

The [next protocol](results/level10/sil-level6-cap128k-protocol.json) tests
capacity **131,072 versus 32,768** from the same verified 28,250,112 source
as `level10-sil-level6-continuation`. Keep four SIL updates, batch 512,
suffix 2,048, losses, sampling priorities and all PPO/curriculum settings
unchanged. This is a capacity comparison against the current restarted
continuation, not the differently sourced SIL20 trial. Both start with empty
training buffers and the same saved learner/SIL RNGs.

Earlier FIFO accounting measured short retention of own experience; a larger
history may help preserve rare later-level observations, or dilute useful
recent examples. It does **not** increase the nominal 0.5 replay draws per
new action. No improvement is assumed and no archived or validation states
may enter the buffer.

This trial is **not launched**. First finish and audit both current tranches.
If a stronger depth reference replaces f58, review and revise the source
decision before launching. Otherwise use five million new actions:

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-sil-level6-cap128k \
  --resume runs/level10-curriculum-balanced-sil/step-028250112 \
  --additional-steps 5000000 --sil-updates 4 --sil-capacity 131072
```

Initially use the sole full-learner slot. The maximum replay-plus-pending
screens become **768 MiB**, **384 MiB** more than the current setting, excluding
array/queue overhead and emulator/model/GPU memory. A read-only precheck reports
42% memory-pressure free and **8,863.31 MiB swap in use**; later VM snapshots
also show some paging. These numbers do not establish ample usable memory.
Recheck post-completion free/compressed memory and swap trends before launch.
Do not stop unrelated processes or add a third learner. Source hashes are
rechecked unchanged, complete sampled validation and final-test reservation
remain unchanged, and the current two learners keep their existing budgets.

### Resumed full-training level 7; first SIL20 level-5 validation

The resumed run completes a from-boot training game at **30,047,948**:
protected worker **11**, **379 / level 7**, 13,439 actions, boot offset 91,939.
Its previous episode ends at 29,617,900, exactly 32 times the episode length
earlier. It publishes levels 5/6/7 at **29,963,724 / 29,994,924 / 30,044,204**.
Four retained learning suffixes account for 68/10/83/3 new points; their
dropped prefixes contain the other 215 points. The
[training audit](results/level10/sil-level6-continuation-first-full7-audit.json)
through 30,080,932 checks all **232 complete full games / 268 practice
segments**, **1,167 suffixes / 500 episode coverages**, origins, raw rewards
and logged-action accounting, with no truncations. This is a changing-policy
training outcome, not frozen level 7 or a source for evaluation-state imports.
The latest frozen primary remains **152.85 / 121 / 298 / level 5**, counts
**17/7/4/2**, all 20 complete, at 30,003,200.

A screen-only check of existing recorded games explains why a fixed
three-boundary assumption is unsafe: the verified f58 replay's reserve count
rises **1 to 2 at frame 9,113**, score 181 / level 3, and stays at two until
frame 13,813. It has three reserve decreases plus GAME OVER. The two older
checked replays have two decreases plus GAME OVER. The
[observation audit](results/level10/reserve-icon-observation-audit.json)
finds no reason to alter the reserve-decrease detector; it does not infer the
game's internal award rule or prove every possible redraw case. No code or
live training setting changes.

SIL20's tenth primary, **27,500,544**, is its first to reach level 5:
**126.4 / 116 / 305**, counts **13/5/3/2**, all 20 complete. Matched SIL4 is
**134.5 / 117 / 315**, counts **14/6/3/3**; no-SIL is **148.55 / 125 / 286**,
level 4, counts **17/6/5/0**. The
[same-source audit](results/level10/balanced-sil20-first-level5-primary-audit.json)
checks all three suites and the SIL20 checkpoint hashes. This improves on
SIL20's earlier depth but does not surpass matched SIL4 or the retained
Level-6 reference. No promotion or secondary audit; both budgets continue.

### Intermediate tranche review, 2026-09-14 08:17 UTC

The [review](results/level10/sil-continuations-review-20260914-0817.json)
checks the continuation through **31,153,631**: **342 complete from-boot
games / 559 practice segments**, highest **7 / 8**, **1,951 learning suffixes /
901 episode coverages**, no truncations, all origins/rewards/pending accounting
and 11 complete primary suites. Six checkpoints reach level 5, totaling twelve
such games, but none reaches level 6 in this restarted run's frozen primaries.
The strongest Level-5 count is **4/20** at **30,752,768**, mean **150.3**;
the following primary at **31,002,624** is **127.5 / 123.5 / 199 / level 3**,
counts **16/7/0/0**. Preserve the original f58 depth reference.

SIL20 through **28,307,001** has **491 complete from-boot games / 844 practice
segments**, highest **5 / 8**, **2,955 suffixes / 1,335 episode coverages**, no
truncations, and thirteen complete primaries. The latest at **28,250,112** is
**131.35 / 122 / 294 / level 5**, counts **16/4/3/1**. Recomputing all 39 suites
at the thirteen matched action counts gives:

| Same-source setting | Checkpoints reaching 5 | Total level-5 games | Total level-6 games |
|---|---|---|---|
| No SIL | 3 | 4 | 0 |
| Four SIL updates | 7 | 12 | 1 |
| Twenty SIL updates | 2 | 3 | 0 |

These repeatedly reuse primary seeds and are not independent test games.
The dose comparison is still intermediate, but gives no reason to replace
the Level-6 reference. Both learner handles are confirmed live; do not mistake
an observer's completed window for a stopped learner. Finish and review the
remaining budgets before the conditional capacity experiment, which remains
unlaunched. The committed packages and fresh final-test reservation are unchanged.

### Supplemental consistency audit at 31,252,480

The continuation's twelfth primary completes all 20 games at
**169.2 / 122.5 / 328 / level 5**, counts **15/9/8/6**. Model SHA-256:
`e0da769e9cfd3bff2ee5c60273d4da4c07482389a14fc748d530569aeca7ec05`.
All seeds, game-over flags, recomputed statistics and saved model/optimizer/
state hashes are checked; the [raw primary](results/level10/validation-sil-continuation-31m25.json)
is copied exactly.

Six Level-5 reaches justify a **supplemental consistency audit**, explicitly
beyond the strict deepest-primary-rank trigger. This candidate does not beat
f58's primary depth, and the live learner's selection rule is unchanged.
The [prelaunch protocol](results/level10/sil-continuation-31m25-audit-selection.json)
requires all 50 secondary games complete. To replace the old Level-5
consistency reference, require strictly more than its four secondary Level-5
games, not merely a larger primary count. Keep the Level-6 depth reference
unless stronger complete depth evidence warrants review and full replay
verification, and report any mean/depth tradeoff separately.

```bash
venv/bin/python -m rl.evaluate \
  runs/level10-sil-level6-continuation/step-031252480/model.safetensors \
  --games 50 --envs 20 --seed 10100 --max-steps 100000 \
  --output results/level10/validation-audit-sil-continuation-31m25.json
```

The audit is active at this record. Only the status observer was stopped to
review this result; both learner handles were re-polled live and continue
unchanged. Use one bounded evaluator, no third learner. This uses reused
secondary seeds, never final seeds 40000-40099, and no evaluation states or
recorded observations enter training. No reference promotion is presumed.

### Verified Level 8 and revised continuation source

The supplemental audit exits **0** with all 50 games complete. The
[validation audit](results/level10/sil-continuation-31m25-validation-audit.json)
recomputes all six candidate/previous-reference suites and verifies the
candidate's mode, seeds, checkpoint identity and hashes:

| Candidate set | Mean / median / best | Highest level | Reach counts 2/3/4/5/6/7/8 |
|---|---|---|---|
| Primary, 20 games | 169.2 / 122.5 / 328 | 5 | 15/9/8/6/0/0/0 |
| Secondary, 50 games | 182.66 / 148 / 555 | 8 | 42/27/18/10/4/2/1 |
| Combined reused validation, 70 games | 178.81 / 140 / 555 | 8 | 57/36/26/16/4/2/1 |

Secondary seed **10128** reaches level 8 at **555 points**, 22,439 actions,
boot offset **90,186**. Seed **10100** reaches 7 at **518**; seeds 10102 and
10146 finish at level 6. The
[full Level-8 replay](results/level10/sil-continuation-31m25-secondary-level8-replay.html)
is recorded with `--max-steps 0`; it matches every secondary-game field,
all 22,440 decoded screens have monotonic score/level displays, and GAME OVER
appears only at the end. Level entries are:

| Level | First frame | Score |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 2,713 | 61 |
| 3 | 5,482 | 127 |
| 4 | 7,454 | 199 |
| 5 | 10,486 | 277 |
| 6 | 13,064 | 317 |
| 7 | 17,531 | 368 |
| 8 | 20,180 | 510 |

It plays **2,259 actions / 45 newly earned points** after entering level 8;
it does not clear level 8. Replay SHA-256 is
`1fd6a5a1cf901275f148311dcd65acf9590225095cc4de8431ef81e611293d2f`.
The [replay verification](results/level10/sil-continuation-31m25-replay-verification.json)
and [policy inspection](results/level10/sil-continuation-31m25-policy-inspection.json)
confirm **zero mismatches across all 22,439 seeded neural actions**. Recording
and inspection both exit 0. There is no action override, restored start, or
validation-data import into training.

Promote **31,252,480 / e0da769e...** as the depth, later-level-consistency
and combined-mean validation reference. Its combined mean 178.81 and sixteen
Level-5 games exceed the old references' 141.79/four and 148.47/six, and it
reaches deeper levels. It does not improve every metric: the old mean reference
has 60 Level-2 games versus 57. Keep all earlier checkpoints and replays.
One Level-8 reach in 70 reused games is not reliable mastery or fresh-test
performance. **Level 10 remains unmet; final seeds 40000-40099 are unused.**

The new source triggers the old capacity protocol's source-review condition.
Preserve that dated record, but **do not launch its old f58-source command**.
The revised [unchanged Level-8 continuation protocol](results/level10/sil-level8-continuation-protocol.json)
gets priority: five million new actions from the verified 31,252,480 source,
all learning settings unchanged, only after SIL20 finishes, is audited and
frees its slot. Keep the existing Level-6-source continuation on its declared
budget. Check memory after the old SIL20 exits; if needed, wait for the other
old run to finish rather than adding pressure or a third learner.

The [revised capacity protocol](results/level10/sil-level8-cap128k-protocol.json)
uses the same Level-8 source and compares 131,072 versus 32,768 entries,
but remains conditional on reviewing the unchanged Level-8 continuation.
If Level 10 is still unmet and no stronger source supersedes it, start capacity
testing as the sole learner after all prior tranches finish and memory is
rechecked. This preserves a same-source comparison. Both new runs are
**unlaunched**; a newly stronger audited source requires another explicit review.

### Both old tranches complete; unchanged Level-8 continuation launched

Both learner sessions exit **0**, observed at 09:09:58 UTC on September 14.
The [Level-6-source completion audit](results/level10/sil-level6-continuation-completion.json)
verifies 33,251,328 total / 5,001,216 new actions, 20 complete primary suites,
522 complete from-boot games and 1,078 complete practice segments, no training
guards, and all 3,293 learning suffixes / 1,600 episode coverages. Changing-
policy full training reaches 7; restored practice reaches 8. The final primary
is 133.25 / 120 / 242, highest 4, so retain the audited 31,252,480 Level-8
checkpoint rather than selecting the final resumable weights.

The [SIL20 completion audit](results/level10/balanced-sil20-completion.json)
verifies 30,007,296 total / 5,001,216 new actions, 692 complete full games,
1,328 complete practice segments and three guarded practice segments. All
4,381 learning suffixes / 2,023 episode coverages pass; the three guarded
suffixes discard 6,144 retained transitions. Nineteen primary suites are
eligible; only two reach 5, totaling three such games, versus eight suites /
thirteen games for the original SIL4 trial. Do not adopt SIL20. Its final
primary is **disqualified**: seed 10017 ends at score 25 / level 1 after
100,000 actions, with `waiting=false` and no GAME OVER. Do not describe that
particular guard as a serve stall or the completed-only statistics as a full
20-game performance estimate. Its three training guards do show waiting.

Both complete logs are copied without overwriting and independently hash-
verified against their run logs:

| Log | Lines / bytes | SHA-256 |
|---|---|---|
| Level-6-source continuation | 5,830 / 3,062,241 | `601eaac4096ea2ae2d5bdcc77288d0ec4ffad0aaabe9414eeccae76e23691fcb` |
| SIL20 | 7,228 / 4,991,150 | `f4a5e9c171b58b3ba5890f4219060c011914e692542473566f93755fb958bbd2` |

No old learner/evaluator/recorder process remains before the next launch.
The post-exit check reports 57% memory free and 9,649.44 MiB swap used; the
percentage alone is not a physical-RAM headroom estimate. Launch the existing
32k-buffer configuration as the sole learner, leaving capacity testing deferred.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-sil-level8-continuation \
  --resume runs/level10-sil-level6-continuation/step-031252480 \
  --additional-steps 5000000 --sil-updates 4
```

Session **96673**, parent PID **45971**, launched at approximately 09:13:45
UTC. Target 36,252,480 rounds to 36,253,696. The
[launch audit](results/level10/sil-level8-continuation-launch-verification.json)
verifies the source's four hashes, metadata-only configuration differences,
32 CPU-only workers / 16 protected-from-boot workers, fresh training buffers,
92 complete full games / 71 practice segments, no guards, and all first 388
learning suffixes / 163 episode coverages. Resume code restores model,
optimizer and learner/SIL RNGs; no evaluation or replay states enter training.
Initial MLX peak allocation is 1,577,597,136 bytes. The later memory check
reports 47% free, 9,577.44 MiB swap used and unchanged cumulative swap-outs.

The first primary is 20/20 complete, 121.5 / 118.5 / 324, highest 5. The
second, **31,752,192**, is 20/20 complete, **175.1 / 139.5 / 356**, highest
6, counts 2-6 **17/10/7/2/1**. Its primary rank beats the selected reference's
primary rank, despite fewer Level-5 reaches. The
[predeclared secondary audit](results/level10/sil-level8-continuation-31m75-audit-selection.json)
uses reused seeds 10100-10149, default sampled decoding and a 100,000-action
guard. It is running at this record; no reference promotion is presumed.
The verified Level-8 checkpoint remains selected and the fresh final seeds
remain unused. The sole learner continues unchanged throughout the audit.

The 31,752,192 secondary audit subsequently exits **1**: 49 complete games
and one guarded serve stall, seed **10135**, score 22 / level 1, boot offset
103,345. It remains waiting at 100,000 actions without GAME OVER. The
[verified result](results/level10/sil-level8-continuation-31m75-validation-audit.json)
disqualifies the suite; its completed-only mean 124.55 is not a full 50-game
estimate. Do not promote the checkpoint or use final-test seeds to retry it.

The third primary, **32,002,048**, completes all 20 games at
**139.8 / 121.5 / 439 / level 7**, counts 2-7 **13/6/3/1/1/1**. Seed
10000 reaches 7, 14,862 actions, boot offset 33,283. Model SHA-256:
`49d341ed615e8b82fc3c18480a7de13437ba04fd20229e7840d8471e807fb637`.
It outranks both prior primaries by depth, but its lower mean/intermediate
counts are a tradeoff. The [predeclared audit](results/level10/sil-level8-continuation-32m-audit-selection.json)
starts as session **93838** only after the previous evaluator exits; it uses
the same reused 50-game set, sampled decoding and diagnostic action guard.
The Level-8 reference remains selected pending this complete secondary review.

The 32,002,048 secondary audit exits **0**, all 50 complete, at
**142.56 / 121 / 302 / level 5**, counts 2-5 **39/17/10/3**. The
[validation audit](results/level10/sil-level8-continuation-32m-validation-audit.json)
verifies all seeds, termination flags, statistics and source hashes.
Combined reused sets: **141.77 / 121 / 439 / level 7**, reach counts 2-8
**52/23/13/4/1/1/0**. This trails the retained Level-8 reference's mean and
later-level counts; do not promote or spend fresh-test games on it.
The sole learner continues its declared unchanged budget. Its fourth primary
completes 20/20 at mean 160.5, highest 5, three Level-5 reaches; it does not
pass the reference-primary screening gate. Capacity testing stays deferred.

The restarted learner rebuilds its own practice through 8. Its first complete
Level-8 segment ends at **32,319,411**, worker 18, **545 points**, 5,941
actions / **231 new points**, from its own peer-sourced Level-6 entry at
score 314 (worker 27, action 26,909, boot offset 169,920). Within this episode
it publishes Level 7 at 32,166,387 / score 366 and Level 8 at
32,237,267 / score 506. The
[fresh-run origin and suffix audit](results/level10/sil-level8-continuation-first-practice8-audit.json)
passes through 32,440,019: 155 complete full games, 249 complete practice
segments, no guards, 801 learning suffixes / 404 episode coverages, and four
complete primary suites. Changing-policy full training still reaches only 5
at this snapshot. Restored Level 8 is not a new performance milestone.

The fifth primary, **32,501,760**, completes 20/20 at
**151.5 / 124 / 328 / level 6**, counts **17/6/5/2/1**. It passes the
reference-primary screening gate and receives the
[predeclared secondary audit](results/level10/sil-level8-continuation-32m5-audit-selection.json).
That evaluator (session **29820**) exits **1**: 49 complete games and one
serve-stalled guard, seed 10135. The
[audited result](results/level10/sil-level8-continuation-32m5-validation-audit.json)
is disqualified, with no reference promotion or fresh-test use. A second
long-waiting game eventually completes; it is not misclassified as truncated.
The sixth primary, 32,751,616, completes all 20 at mean **123.1**, highest
4, so it does not qualify for secondary review. The learner continues the
same budget, and the verified Level-8 reference remains selected.

### Level-8-source continuation midpoint review

At **33,812,390** logged actions, the
[midpoint audit](results/level10/sil-level8-continuation-midpoint-audit.json)
verifies **270** complete from-boot training games, **612** complete practice
segments, **zero** training guards, and all **1,710** learning suffixes /
**882** episode coverages. All reward, source-publication, protected-worker,
action-origin, pending-prefix and replay counts pass, and logged updates are
finite. Changing-policy full training reaches 5 and restored practice 8;
these remain separate from frozen performance.

All first **ten primary suites complete**, with six checkpoints reaching 5,
ten Level-5 games in total, and highest primary level 7. The seventh through
tenth means/highest levels are **109.3/5**, **121.4/4**, **124.2/4**, and
**94.05/3**. None passes the selected reference's primary screening gate.
The first five checkpoints contain nine Level-5 games, the next five only
one. These are repeated reused seeds across changing weights, not independent
game samples or a pooled performance estimate. The three secondary audits
already reviewed do not replace the verified source.

Only the status observer was stopped for this review; learner session 96673
was re-polled live. Memory reports 43% free, 9,545.44 MiB swap used and
unchanged cumulative swap-outs (2,491,021). MLX peak is 1,577,597,728 bytes.
Continue the declared five-million-action tranche; retain the Level-8 source,
keep the larger-buffer trial conditional on completion/source/memory review,
and leave final seeds 40000-40099 untouched. No learning or environment
configuration changes are made at this midpoint.

The next primary, **34,000,896**, is **disqualified**: 19 complete games and
one 100,000-action serve guard. Seed **10006**, boot offset **72,093**, is
still waiting at score **265 / level 4**, with no GAME OVER. The
[guard audit](results/level10/sil-level8-continuation-first-primary-guard-audit.json)
verifies all seeds, completion flags, statistics and checkpoint hashes; the
[raw suite](results/level10/validation-sil-level8-continuation-34m-disqualified.json)
is copied exactly. Its completed-only mean 131.11 is not a full 20-game
performance estimate. No secondary evaluation or reference promotion follows.
The learner resumes updates; this is a validation failure, not process exit
or a reason to restart training. Final seeds remain unused.

### Later qualifying primaries and first training guard

The twelfth primary, 34,250,752, completes 20/20 at mean 113.95, highest 4.
The thirteenth, **34,500,608**, completes 20/20 at
**132.2 / 123 / 349 / level 6**, counts 2-6 **18/3/2/1/1**. Seed 10008
reaches 6 in 16,973 actions, boot offset 91,003. Its
[predeclared audit](results/level10/sil-level8-continuation-34m5-audit-selection.json)
uses the existing reused secondary protocol. Evaluator **36683** exits **1**:
48 complete games and two serve guards, seeds **10117 / 10137**, at
scores **63 / 64**, both level 1 and waiting at 100,000 actions. The
[verified result](results/level10/sil-level8-continuation-34m5-validation-audit.json)
disqualifies the suite; completed-only mean 133.5 is not a full-set estimate.
No reference promotion or final-test use follows.

The learner's first training guard ends at **34,640,319**, worker **30**,
from boot at **33 points / level 1**, 100,000 actions, boot offset 13,076,
still waiting without GAME OVER. The
[training-guard audit](results/level10/sil-level8-continuation-first-training-guard-audit.json)
verifies that its retained **2,048**-transition terminal suffix is discarded,
not committed to replay; 97,132 older transitions in that learning segment
had already been trimmed. Earlier legitimate completed learning boundaries
are distinct. Through 34,744,578, all **2,276** suffixes / **1,187** episode
coverages and source/reward/action/buffer checks pass: 337 complete full games,
849 complete practice segments, one training guard. The guard receives no
performance credit. The learner remains live and resumes normal collection.

The fourteenth primary, **34,750,464**, completes 20/20 at
**123.9 / 105 / 353 / level 6**, counts **13/4/4/1/1**. Seed 10007 reaches
6 in 11,339 actions, boot offset 29,999. Model SHA-256:
`ee310d879bacb03589dbdee43e109b5cb4341ecc51d642b70d42b21f90b8420e`.
The [next predeclared audit](results/level10/sil-level8-continuation-34m75-audit-selection.json)
starts as session **74298** only after the preceding evaluator exits. It
passes the reference-primary screening gate, not an automatic promotion.
Keep the verified Level-8 source, unchanged learner and untouched final seeds.

The 34,750,464 secondary audit exits **1** with 49 complete games and one
serve-stalled guard, seed **10117**, still at level 4 after 100,000 actions.
The [verified result](results/level10/sil-level8-continuation-34m75-validation-audit.json)
disqualifies the suite; no reference promotion or full-set performance estimate
follows it. All five standalone secondary evaluators for this continuation
are now terminal. The fifteenth primary, **35,000,320**, completes 20/20 at
mean **133.95**, highest 5, two Level-5 reaches. It does not pass the
reference-primary screening gate. Continue the declared tranche, retaining
the verified Level-8 source and the conditional capacity plan.

The sixteenth primary, **35,250,176**, completes 20/20 at
**138.1 / 124 / 323 / level 6**, counts **15/6/4/1/1**, qualifying for the
[declared secondary check](results/level10/sil-level8-continuation-35m25-audit-selection.json).
Evaluator **5048** exits **1**: 49 complete games and one serve guard,
seed **10109**, score **254 / level 4**, still waiting at 100,000 actions,
boot offset 171,226. Another long-waiting game eventually completes and is
not counted as truncated. The
[audited result](results/level10/sil-level8-continuation-35m25-validation-audit.json)
disqualifies the suite; completed-only mean 124.88 is not full-set performance.
All six standalone audits are terminal and none replaces the Level-8 source.
The seventeenth primary, **35,500,032**, completes 20/20 at mean **143.65**,
highest 5, two Level-5 reaches; it does not qualify for a secondary audit.
The unchanged learner continues toward its declared budget endpoint.

### Unchanged Level-8 continuation completes; capacity comparison launches

Learner **96673** exits **0**, observed at **10:26:53 UTC** on September 14,
at **36,253,696** total / **5,001,216** new actions. The
[completion audit](results/level10/sil-level8-continuation-completion.json)
verifies 447 complete full training games, 1,253 complete practice segments,
two from-boot serve guards, all 3,205 learning suffixes / 1,702 episode
coverages, and 4,096 discarded retained transitions. The second guard is
protected worker 2 at 35,971,619, score 193 / level 3, boot offset 124,611.
No guard receives performance credit. Source/reward/origin/buffer accounting
passes, and the final learner saves RNGs with replay explicitly not persisted.

Of 20 primary suites, **19** are eligible; 11 checkpoints reach 5, totaling
17 Level-5 games on reused seeds. Highest eligible primary level is 7.
Five of six secondary audits are disqualified; the complete 32,002,048 audit
does not outperform the retained Level-8 source. The last primary completes
20/20 at **132.65 / 131.5 / 263 / level 4**, so the final resumable weights
are preserved, not selected. This unchanged continuation has not improved
the verified reference. Level 10 remains unmet.

The full log is copied without overwrite and independently hash-verified:
**5,428 lines / 2,584,730 bytes**, SHA-256
`f1a14c1cda6e5674c9dc34760fcf63d6ef00435e76d2f6860a389807a9e006fc`.
No learner/evaluator/recorder process remains before the next launch.

The [capacity launch decision](results/level10/sil-level8-cap128k-launch-decision.json)
rechecks the exact source hashes and the completed same-source comparator.
Post-exit samples at 10:27:38 and 10:28:24 show stable swap use, 8,986.81 MiB,
and unchanged cumulative swap-outs, 2,575,213. The free-memory percentages
(47/48%) are not treated as physical-RAM estimates. The larger buffer adds
384 MiB of bounded maximum screen storage; use the sole learner slot and
monitor actual paging. The old f58-source plan stays superseded.

```bash
caffeinate -i venv/bin/python -m rl.ppo \
  --run runs/level10-sil-level8-cap128k \
  --resume runs/level10-sil-level6-continuation/step-031252480 \
  --additional-steps 5000000 --sil-updates 4 --sil-capacity 131072
```

Session **60642**, parent PID **69997**, launches around **10:29:13 UTC**.
Target 36,252,480 rounds to 36,253,696. The
[launch verification](results/level10/sil-level8-cap128k-launch-verification.json)
confirms the only control-configuration differences are run location and
capacity; all 32 workers omit MLX and the first 16 remain protected from boot.
Model, optimizer and learner/SIL RNGs resume from the same source; curriculum,
replay and pending suffixes are fresh. At 31,268,864, the first learning
metrics, action/suffix counters and SIL updates match the control exactly,
excluding timing/allocation telemetry. At the initial audit through 31,357,639,
12 complete full games / 63 suffixes pass accounting, with no restored
segments or training guards yet. The buffer has already retained more than
32,768 entries, so the capacity change is active.

Post-launch memory at 10:30:37 reports 37% free and 8,978.81 MiB swap used,
with no new swap-outs. MLX peak is 1,577,597,648 bytes. These are startup
checks, not performance results; continue matched validation and memory
monitoring. Keep the verified Level-8 source and final seeds 40000-40099 unused.

### Larger-buffer first validation and secondary audit

The first checkpoint, **31,502,336**, completes 20/20 primary games at
**159.25 / 120 / 318 / level 6**, counts **17/7/6/3/1**. The matched
unchanged control was 121.5 / 118.5 / 324 / level 5, counts 12/5/2/1.
Mean/depth/later-level counts improve at this checkpoint, not best score;
one checkpoint is not proof of a general or whole-trial improvement. Its
[primary and full-buffer audit](results/level10/sil-level8-cap128k-first-primary-audit.json)
verifies 34 complete full games / 139 learning suffixes, no training guards,
and a fully populated 131,072-entry replay. At 10:33:25, memory reports 36%
free, 8,987.31 MiB swap used and 2,576,269 cumulative swap-outs; continue
monitoring memory rather than assuming the startup sample remains current.

The [predeclared secondary audit](results/level10/sil-level8-cap128k-first-audit-selection.json)
exits **0** (session **53712**), with all 50 complete at
**145.78 / 123.5 / 515 / level 7**, counts 2-7 **39/16/10/4/2/1**.
Seed 10145 reaches 7 in 18,574 actions, boot offset 36,520; seed 10128 reaches
6 at score 372. The [validated result](results/level10/sil-level8-cap128k-first-validation-audit.json)
checks seeds, termination, statistics, decoding mode and unchanged hashes.
Combined reused validation: **149.63 / 122 / 515 / level 7**, counts 2-8
**56/23/16/7/3/1/0**. Do not promote: the retained Level-8 source has mean
178.81, sixteen Level-5 reaches and greater depth. The larger-buffer learner
continues its declared budget, with no fresh final-test use or replay-state
imports. The first secondary evaluator is terminal; only the learner is live.

The [first restored-training audit](results/level10/sil-level8-cap128k-first-restored-audit.json)
passes through **31,812,596**: 66 complete full games, 25 complete practice
segments, no training guards, and all 324 learning suffixes / 91 episode
coverages. Protected worker **8** first publishes its own from-boot Level-5
entry at 31,595,721, score 276, action 10,727, boot offset 182,447. Worker
31's first restored segment ends at 31,604,160, score 278 / level 5, 214
actions / two new points. Worker 16 first publishes a restored Level-6 entry
at 31,648,177, score 315. Completed restored practice reaches 7; full changing-
policy training still reaches 5. These are not frozen performance results.

The second primary, **31,752,192**, is **disqualified**: 19 complete games
and one serve guard, seed **10005**, score **204 / level 3**, still waiting
at 100,000 actions, boot offset 67,063. The
[guard audit](results/level10/sil-level8-cap128k-first-primary-guard-audit.json)
verifies all seeds, flags, statistics and hashes; the raw suite is copied
exactly. Completed-only mean 162.68 / highest 6 is not a full 20-game result.
No secondary audit, reference promotion or final-test use follows. The
larger-buffer learner continues unchanged, with the Level-8 source retained.

The larger-buffer learner's first completed own Level-8 practice ends at
**31,863,066**, worker 25, **531 points**, 4,010 actions / **165 new points**,
from its current-run Level-7 entry at score 366 (worker 18, source action
14,862, boot offset 182,447). It publishes Level 8 at 31,823,322 / score 509.
The [practice audit](results/level10/sil-level8-cap128k-first-practice8-audit.json)
passes through 32,034,920: 85 complete full games, 74 complete practice
segments, zero guards, and all 452 suffixes / 159 episode coverages. Restored
changing-policy Level 8 is not a new frozen or from-boot performance result.

The third primary, **32,002,048**, completes 20/20 at
**166.55 / 130.5 / 370 / level 6**, counts **17/9/7/3/2**, qualifying for
the [declared secondary audit](results/level10/sil-level8-cap128k-32m-audit-selection.json).
That evaluator, **89482**, exits **0**, all 50 complete, at
**140.4 / 121.5 / 356 / level 6**, counts **38/15/11/5/1**. The
[verified audit](results/level10/sil-level8-cap128k-32m-validation-audit.json)
checks all seeds, termination flags, statistics and checkpoint hashes.
Combined reused sets: **147.87 / 122 / 370 / level 6**, counts 2-8
**55/24/18/8/3/0/0**. This improves matched control mean/counts through 6
(141.77, four Level-5 games), but not its level-7 depth. Neither displaces
the retained Level-8 source (178.81, sixteen Level-5 games). Keep that source,
continue the capacity trial, and leave final seeds untouched. Both completed
capacity-trial secondary evaluators are terminal; the learner remains live.

### Larger-buffer first complete from-boot training Level 7

Protected worker **6** completes the run's first from-boot Level-6-or-higher
training game at **32,471,271**, ending at **507 points / level 7**,
17,903 actions, boot offset **57,119**, with normal GAME OVER. No restored
start or archive source is attached to this game. Its previous episode ends
at 31,898,375; the difference is exactly 32 times its own action count.
It publishes Levels 5/6/7 at 32,293,191 / 32,365,191 / 32,421,223,
scores 293 / 333 / 383. The
[from-boot audit](results/level10/sil-level8-cap128k-first-full7-audit.json)
verifies its five legitimate learning suffixes, including trimmed prefixes,
and all source/reward/action accounting through **32,587,776**: 126 complete
full games, 163 complete practice segments, no training guards, 747 suffixes /
289 episode coverages. Multiple learning boundaries are allowed because
screen-visible reserve increases have already been observed; do not impose
an artificial three-boundary limit.

This is changing-policy training, not frozen performance or a reference
promotion. The completed unchanged control never exceeded full-training
level 5, but one later training game does not establish reliable improvement.
The fourth and fifth primaries complete 20/20 at **139.2 / level 4** and
**141.55 / level 5**; neither passes the reference-primary screening gate.
Continue the capacity trial and monitor memory/paging. The Level-8 reference
and untouched final-test seeds remain preserved.

The sixth primary completes 20/20 at mean 133, highest 4. The seventh,
**33,001,472**, completes 20/20 at **129.45 / 112.5 / 366 / level 6**,
counts **13/7/4/1/1**, qualifying for the
[declared secondary audit](results/level10/sil-level8-cap128k-33m-audit-selection.json).
Evaluator **79906** exits **0**, all 50 complete, but only reaches 4:
**112.84 / 115 / 279**, counts **34/9/4/0**. The
[verified result](results/level10/sil-level8-cap128k-33m-validation-audit.json)
gives combined reused validation **117.59 / 115 / 366 / level 6**, with one
Level-5-or-higher game. Do not promote. All three capacity secondary audits
are terminal; continue the learner with the Level-8 reference retained.

The [capacity comparison chart](results/level10/sil-level8-capacity-comparison.png)
and SVG/JSON companions show the completed control and the capacity trial
through primary **33,001,472** / progress **33,161,216**. Its
[verification](results/level10/sil-level8-capacity-comparison-verification.json)
recomputes all 447 plotted rolling summaries from complete from-boot game
records. Incomplete suites have no mean/depth point; their completion rates
remain visible. Restored practice is excluded. Stars mark complete primary
Level-6-or-higher reaches and the dotted depth line marks the Level-10 goal.
Separate secondary audits, including the retained Level-8 result, are not
plotted. The historical random baseline uses seeds 30000-30099, not fresh
final-test games. Both lineages include their shared 500,000 prior DQN actions;
do not sum fork counters. This is a dated snapshot, not a live chart.

```bash
venv/bin/python -m rl.report \
  runs/level10-sil-level8-continuation runs/level10-sil-level8-cap128k \
  --target-level 10 --highlight-level 6 --baseline results/level5/random.json \
  --output results/level10/sil-level8-capacity-comparison
```

The eighth capacity primary, **33,251,328**, is **disqualified**: 19 complete
games and one serve guard, seed **10017**, score **58 / level 2**, still
waiting at 100,000 actions, boot offset 188,429. The
[guard audit](results/level10/sil-level8-cap128k-33m25-primary-guard-audit.json)
verifies seeds, flags, statistics and checkpoint hashes; the raw result is
copied exactly. Its completed-only mean 99.26 is not a full-suite estimate.
No secondary audit or promotion follows. The comparison chart above remains
the explicitly dated seven-primary snapshot, not an updated live chart.

The [capacity midpoint audit](results/level10/sil-level8-cap128k-midpoint-audit.json)
through **33,958,366** verifies 238 complete full games / 495 complete practice
segments, no training guards, and all 1,689 suffixes / 733 episode coverages.
Eight of ten primary suites are eligible, four reach 5, and their highest
level is 6. The 101 completed practice segments starting on **Level 8** all
end on 8; mean newly earned score is **34.05**, best **62**. This is evidence
of the current practice bottleneck, not a frozen-policy result or proof of a
timing problem. The declared capacity trial continues unchanged.

The [first training-guard audit](results/level10/sil-level8-cap128k-first-training-guard-audit.json)
through **34,925,064** passes all 2,269 suffixes / 1,014 episode coverages.
Restored worker 21 hits its 100,000-action guard at **34,842,646**, waiting
at score 353 / Level 6. Its 2,048-step retained suffix has zero reward and
is correctly discarded; this is not a complete game or practice success.
Primaries 11-14 all complete and reach 4, 4, 4, 5 respectively; none qualifies
for another secondary audit against the retained source.

### Isolated action/history diagnostic (predeclared)

The [protocol](results/level10/temporal-e0da-protocol.json) fixes the verified
e0da weights and compares two **evaluation-only** 50k-action arms on reused
primary seeds 10000-10019. Adjacent frames span nominally 150k T-states;
frames spaced two actions apart span 300k, matching the original history
span. The action guard is 200k, matching the original nominal 10-billion-
T-state guard (HUD settling remains additional). Both retain ordinary
categorical sampling; no seed, history identifier, coordinates, scripted
steering or serve override enters the model. More frequent sampling itself
changes action RNG consumption and trajectories, so this is not perfect
physical-time invariance.

`rl/temporal_probe.py` wraps observation history without modifying the live
learner, emulator, shared evaluator, snapshot format or recorder. Six new
tests cover exact stride-1 passthrough, stride-2 padding/copying, interleaved
and duplicate-seed games, native serial/parallel equivalence, invalid inputs
and rejection of fresh final seeds. **All 89 tests pass** (7.214 seconds).
A two-game 100k/stride-1 compatibility gate must exactly match archived source
games before either 50k arm runs. Then run adjacent followed by spaced, with
one diagnostic process at a time. No result can automatically be promoted;
end-to-end support and standard revalidation would be needed first. Final
seeds 40000-40099 remain untouched.

The [joint temporal audit](results/level10/temporal-e0da-two-arm-audit.json)
verifies both predeclared arms, their unchanged source/implementation hashes,
identical seeds and boot offsets, termination flags and all statistics. The
two-game compatibility gate exactly matches the archived original games.
All three evaluators exit 0; neither arm has an incomplete game.

| Same frozen e0da weights; same 20 reused seeds | Mean | Median | Best | Highest level | Level 5+ games |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 100k actions, adjacent frames | 169.2 | 122.5 | 328 | 5 | 6 |
| 50k actions, adjacent frames | 144.7 | 119.5 | 255 | 4 | 0 |
| 50k actions, two-action-spaced frames | 269.4 | 276 | 543 | 8 | 10 |

Spaced-frame reaches at levels 2-8 are **20 / 18 / 14 / 10 / 2 / 2 / 1**.
Seed **10015** completes at **543 / Level 8**, **44,325 actions**, boot offset
173,912; seed 10009 completes at 464 / Level 7. Scores improve on 17/20 games
versus original timing and 18/20 versus adjacent 50k. These comparisons are
descriptive evidence on reused validation seeds, not a fresh estimate of
generalization or proof of a single timing mechanism. **No reference promotion
or final test follows this diagnostic.** The original-timing verified Level-8
model remains the reference.

This warrants end-to-end stride support after the unchanged capacity learner
finishes: default-stride compatibility, exact reproduction of the full spaced
primary suite, an uncapped replay with all actions checked, then a fixed
50-game reused secondary audit at 50k/stride 2 with a 200k-action guard. No
timing/history sweep or new fine-tuning run precedes those checks.

The [integration protocol](results/level10/temporal-e0da-integration-protocol.json)
records the pre-edit dependency hashes and all gates before shared-core edits.
It also fixes the 70-game promotion rule: complete suites only, lexicographic
level-reach counts from 10 down to 2 and then mean score versus the retained
reference. A copied evaluation-only package, if needed, must retain identical
weights and explicit timing/stride provenance; it cannot resume training.

The [capacity completion audit](results/level10/sil-level8-cap128k-completion.json)
confirms session **60642 exits 0** at **36,253,696**, adding **5,001,216**
actions in 4,323.82 seconds. There are 421 complete full games (highest 7),
970 complete practice segments (highest 8), and one guarded practice segment.
All 3,043 suffixes / 1,392 episode coverages pass accounting; the one guarded
2,048-transition suffix is discarded. Of 20 primary suites, 18 are eligible
and two disqualified; eight eligible checkpoints reach 5 (15 games total),
highest primary 6. The final primary completes 20/20 at mean 145.7, median
127, best 295, highest 5; long-waiting seed 10014 eventually completes.

No larger-buffer secondary replaces the original-timing Level-8 source.
The final checkpoint is retained, not promoted; its complete 5,091-line log
is copied with SHA-256 `1e52f97bfe95fbc768f14f134313e151d302f43f8b981786e25e3a280e283b1a`.
All learner/evaluator processes have exited, opening the shared-core edit gate
for the predeclared timing/history integration. No new learner or final test
has launched.

The [integration implementation audit](results/level10/temporal-e0da-integration-implementation-audit.json)
records the applied source hashes, **96 passing tests** (7.172 seconds), and
the ordinary evaluator's exact reproduction of all 20 original game
dictionaries (session 14813 exits 0). The default stride remains 1. Seven new
tests cover native action/reward preservation with spaced histories, equality
to the isolated probe across serial/parallel scheduling, full seven-frame
snapshot continuation in another process, atomic rejection of incompatible
snapshot/peer histories, replay offsets and CPU-only vector workers.

`runs/level10-temporal-e0da-eval` contains byte-identical e0da weights and a
new evaluation-only state declaring 50k actions / stride 2, zero additional
training and disabled resume. No optimizer is copied; the source is unchanged.
State SHA-256 is `948d4bef41f45b024f1a8c6f3c35ae7a65b39863f2bfb8e24ea2e8e7c97a20bb`.
The ordinary spaced primary reproduction uses these checkpoint defaults;
the isolated TemporalPolicy wrapper is no longer on this evaluation path.

The [spaced reproduction audit](results/level10/temporal-e0da-integrated-spaced-audit.json)
confirms session 14242 exits 0 with **all 20 complete game dictionaries exactly
matching** the diagnostic, not merely its aggregate score. The ordinary
recorder then produces the uncapped
[543-point Level-8 replay](results/level10/temporal-e0da-spaced-primary-level8-replay.html):
seed 10015, 44,326 raw frames / 44,325 actions, matching every original game
field. Recording session 85211 and inspection session 92885 both exit 0.
The [replay verification](results/level10/temporal-e0da-spaced-replay-verification.json)
and [policy inspection](results/level10/temporal-e0da-spaced-policy-inspection.json)
verify **zero mismatches across all 44,325 actions**, using replay stride 2
and batch size 128. The replay reaches 8 at frame 38,298 / score 498, then
earns another 45 points before GAME OVER; it does not clear Level 8.

These completed gates open the single fixed secondary audit: **50 games,
seeds 10100-10149, 20 workers, 50k actions / stride 2, 200k-action guard**.
Session **76080** is running it. No learner is active. The original-timing
Level-8 reference remains selected until the full, complete 70-game comparison
is audited; final seeds and the committed packages remain unchanged.

The [secondary audit](results/level10/temporal-e0da-spaced-secondary-audit.json)
confirms session **76080 exits 0**, all **50 games complete**, mean **218.98**,
median **239**, best **404**, highest **7**; counts 2-7 are **50/36/30/11/4/1**.
Together with the exactly reproduced primary, all **70** games complete at
**233.39 / 244.5 / 543 / Level 8**, counts 2-8 **70/54/44/21/6/3/1**.
The old timing gives 178.81 / 140 / 555 and counts **57/36/26/16/4/2/1**.
The predeclared lexicographic rank improves, all integration gates pass, and
the fixed timing configuration is **promoted as the validation reference**.
It does not improve highest level or best score. Both original source and
timing configurations remain intact; no Level 9/10 or fresh-test claim follows.

### Fine-tuning the verified timing configuration

The [training protocol](results/level10/timing50k-stride2-sil4-protocol.json)
predeclares **4,000,000 additional 50k actions** from the original e0da learner,
not the evaluation-only package. Rounded rollout end is **35,258,368**.
Gamma and GAE lambda become their square roots; rollout/PPO batch sizes,
evaluation intervals/action guards, SIL capacity/suffix/batch sizes double
to retain nominal emulated-time windows and update/sample ratios. Learning
rate 1e-5, entropy .001, four epochs/four SIL updates, loss/PER settings and
the 32-worker/16-protected curriculum stay fixed. This is an adapted training
recipe, not a one-factor ablation or exact trajectory/time invariance.

The [launch verification](results/level10/timing50k-stride2-sil4-launch-verification.json)
checks all configured parameters/source hashes, 32 CPU-only workers / 16
protected roles, fresh buffers and the first 14 suffixes through 31,371,340.
Session **48013** is the sole live learner; no standalone evaluator remains.
The promoted model stays frozen, no evaluation/replay states are imported,
and final seeds remain unused until a verified Level-10 policy is frozen.

The [first fine-tuned primary audit](results/level10/timing50k-stride2-sil4-first-primary-audit.json)
at **31,506,432** is **20/20 complete**, mean **224.8**, median **225.5**,
best **579**, highest **8**, counts 2-8 **18/12/10/7/2/2/1**. Seed **10006**
completes at 579 / Level 8 in **48,490 actions**; seed 10001 reaches 7.
Its primary rank trails the frozen timing reference's 20-game rank, so no
secondary audit or promotion follows. The full checkpoint hashes and saved
selection/RNG state are verified. Training accounting through **31,577,262**
passes all 48 suffixes / six full-game coverages, with no guards. No Level
9 or 10 is established; the sole learner continues its declared tranche.

The [first same-run archive audit](results/level10/timing50k-stride2-sil4-first-archive-audit.json)
through **31,956,992** passes all 106 suffixes / 20 completed full-game
coverages, with no guards. Worker 19 publishes Level 5 at **31,877,428**
(score 278, local action 19,530, boot offset 50,304), then Level 6 at
**31,938,420**. Protected worker 6 independently publishes Level 5. These
entries all originate in this run's fresh from-boot games under changing
weights, not evaluation/replay imports. They are available for subsequent
practice; no completed restored segment or new frozen milestone is claimed.

The [second-primary audit](results/level10/timing50k-stride2-sil4-32m-primary-audit.json)
at **32,006,144** completes **20/20** games: **279.3 / 271 / 558 / Level 8**,
counts 2-8 **19/17/16/11/4/3/2**. Seed 10008 reaches 8 at score 558 in
50,374 actions; seed 10012 reaches 8 at 546 in 60,462 actions. The model
hash is `0ac3de98f31dfa3c1f8ce514d6b762124ff4d9d0b1f294130feef947163e286a`.
Its primary rank improves over the preserved timing reference and qualifies
for the [predeclared secondary audit](results/level10/timing50k-stride2-sil4-32m-audit-selection.json).
Session **24318** runs 50 reused games (10100-10149), 20 workers,
50k/stride 2, 200k-action guard. No promotion follows partial results; after
all 70 games pass comparison, a full selected replay and all-action check
are required before promotion. Learner 48013 continues independently.

The same source/reward audit through **32,086,590** verifies 129 suffixes /
28 episode coverages: 26 complete full games, two complete restored segments,
no guards. The first restored segment uses protected worker 6's own Level-5
entry and earns only four new points; the second starts from the learner's
own Level-6 entry and earns eight new points. These are legitimate self-play
practice records, not complete-game performance or imported demonstrations.

### Fine-tuned 32M reference verified and promoted

The [secondary audit](results/level10/timing50k-stride2-sil4-32m-secondary-audit.json)
finishes **50/50 complete**, mean **245.06**, median **256**, best **572**,
highest **8**. All **70** reused games together give **254.84 / 261 / 572**,
counts 2-8 **66/57/45/26/11/8/3**. The declared depth/count rank improves over
the preceding timing reference, although Level-1 clears fall from 70 to 66.
This remains validation-only selection, not reliable mastery or a fresh test.

The selected seed **10126** exactly reproduces **572 / Level 8** uncapped in
**55,422 actions / 55,423 frames**; recorder 61864 and inspector 85729 both
exit 0. Every action matches the frozen sampled neural policy. The replay
enters Level 8 at frame 36,767 / score 514 and earns 58 further points,
without clearing it. [Full verification](results/level10/timing50k-stride2-sil4-32m-replay-verification.json)
and [promotion decision](results/level10/timing50k-stride2-sil4-32m-promotion.json)
preserve all hashes and limitations. Previous references remain untouched.

The [33M progress audit](results/level10/timing50k-stride2-sil4-33m-progress-audit.json)
also verifies the third/fourth primaries: **32,505,856**, all 20 complete,
mean **181.7**, highest **6**; **33,005,568**, all 20 complete, mean **203.25**,
highest **7**. Neither qualifies for a secondary audit. Through **33,118,302**,
all 366 suffixes / 129 episode coverages pass source/reward/action checks:
66 full games and 63 restored segments, no guards. Full changing-policy
training and restored practice now reach 8; neither is a frozen-model depth
milestone. The declared training stage continues unchanged.

The fifth primary at **33,505,280** completes all 20 games at mean **197.45**,
median **155**, best **561**, highest **8**, counts 2-8 **18/10/8/5/2/1/1**.
The [checkpoint audit](results/level10/timing50k-stride2-sil4-33m5-primary-audit.json)
verifies hashes and its lower rank; no secondary audit follows. The sixth
primary at **34,004,992** is also 20/20 complete, mean **188.7**, median
**189.5**, best **438**, highest **7**, counts 2-7 **16/13/6/3/2/2**. The
32M reference remains selected. Source/reward/action accounting through
**34,083,965** passes all **607 suffixes / 233 episode coverages**, including
98 complete full games and 135 restored segments, with no guards.

### Fixed finer-action diagnostic of the 32M model

The [predeclared diagnostic](results/level10/temporal-32m-25k-stride4-protocol.json)
tests exactly one alternative: **25k actions / stride 4**, preserving the
nominal 300k-T-state four-frame history. It uses the unchanged 32M weights,
sampled neural actions, reused primary seeds and a doubled 400k-action guard.
No training settings change; this is not a pure causal timing ablation because
different action frequency also changes decisions and policy RNG consumption.

Evaluator **74974 exits 0**: all **20** complete, mean **217.95**, median
**225**, best **559**, highest **8**; counts 2-8 **18/11/10/7/4/1/1**.
Seed 10013 reaches 8 in 104,161 actions. The selected 50k/stride-2 baseline
has mean 279.3 and counts **19/17/16/11/4/3/2**. The alternative's declared
rank is lower, so there is **no secondary audit, timing-package promotion,
or further timing sweep**. The [completed audit](results/level10/temporal-32m-25k-stride4-primary-audit.json)
verifies source hashes, seeds, boot offsets, timing and recomputed statistics.
Learner 48013 remains the sole active process; final-test seeds stay unused.

### Isolated sampling-temperature diagnostic

The [protocol](results/level10/temperature-32m-protocol.json) predeclares one
fixed **temperature 0.8** diagnostic of the selected 32M logits, with timing,
screen inputs and model weights unchanged. It tests reduced sampling noise,
not an action heuristic; there is no state-dependent override or sweep.
The new `rl.temperature_probe` is isolated from the live learner and refuses
fresh-test seeds and result overwrites. Its output cannot itself promote a
model or configuration.

All **101 regression tests** pass, including five new tests for exact unit-
temperature behavior, unchanged screen inputs/RNG draws, native serial/parallel
agreement, invalid temperatures and fresh-seed rejection. The
[compatibility audit](results/level10/temperature-32m-compatibility-audit.json)
confirms exact complete dictionaries for baseline seeds 10000 and 10001:
272/4/29,564 actions and 250/4/23,870 actions. Session 52533 exits 0.
Only then does session **91516** start the one 20-game temperature-0.8 primary.
All games must complete and improve the selected depth/count rank before
any further integration or secondary audit. The training stage remains fixed.

Meanwhile the seventh training primary at **34,504,704** completes 20/20,
mean **200.35**, best **377**, highest **6**, counts 2-6 **16/12/10/5/3**.
It does not qualify for a secondary audit. No Level 9 or 10 is established.

The [temperature-0.8 audit](results/level10/temperature-32m-08-primary-audit.json)
finishes **20/20 complete**, mean **220.4**, median **264**, best **577**,
highest **8**, counts 2-8 **17/14/11/6/2/1/1**. Session91516 exits0.
Seed10014 temporarily waits for a serve at577/Level8, then the neural policy
serves and finishes unassisted at79,119 actions. No guard fires. The declared
rank trails the selected unit-temperature reference; no integration, secondary
audit, promotion or sampling sweep follows. Original categorical sampling
and training remain unchanged. This result does not establish Level9/10.

### Initial timing fine-tuning completed; lower-rate continuation launched

Session **48013 exits 0** at **35,258,368**, after **4,005,888 additional
actions** and **3,367.31 seconds**. All **eight primary suites** complete;
seven reach Level 5, with 37 Level-5 reaches across them and highest level 8.
The final scheduled primary at **35,004,416** gives **157.5 / 127.5 / 274 /
Level 4**, below the selected 32M reference. Terminal weights are preserved,
not automatically promoted or used for the next stage.

The [completion audit](results/level10/timing50k-stride2-sil4-completion.json)
verifies **139 complete from-boot games / 237 complete restored segments**,
highest 8 in each, with **zero guards**. All **922 learning suffixes / 376
episode coverages** pass source, reward, action and storage accounting.
The complete [log](results/level10/logs/level10-timing50k-stride2-sil4.jsonl)
is copied byte-for-byte: **1,739 lines / 1,032,855 bytes**,
SHA-256 `cc238b8da59d6406c6bd66f5daa31d3c9e077343fa46b2805d7ac4cb993f4bbd`.

The [next protocol](results/level10/timing50k-stride2-lr5e6-long-protocol.json)
resumes verified **step-032006144** with learning rate **5e-6** and
**8,000,000 additional actions**, rounded end **40,009,728**. All other
learner/timing/sampling/curriculum/SIL settings stay fixed. Adam and saved
learner/SIL RNG states resume; archives, replay and pending histories restart
empty. Because the source and transient buffers differ from uninterrupted
training, this is a directional experiment, not a matched one-factor ablation.
Review on complete target, budget, faults, or five consecutive complete
primaries with no Level-5 reach. Final-test seeds remain unused.

Only after previous processes exit and source/log hashes pass does sole
session **86865** launch (`runs/level10-timing50k-stride2-lr5e6-long`). Its
[startup verification](results/level10/timing50k-stride2-lr5e6-long-launch-verification.json)
confirms the exact inherited configuration and explicit overrides, 32 CPU-only
workers / 16 protected roles, empty buffers and accounting through **32,109,877**
(15 suffixes / one completed full game, no guards). The selected reference
stays frozen; no fresh-test result or new depth milestone is claimed.

The [first lower-rate primary audit](results/level10/timing50k-stride2-lr5e6-long-first-primary-audit.json)
at **32,505,856** completes **20/20 games**, mean **272.4**, median **266**,
best **544**, highest **8**, counts 2-8 **18/16/13/8/6/4/1**. Seed10000
finishes at544/Level8 in46,915 actions. Levels6/7 are more frequent than in
the selected reference's primary (6/4 versus4/3), but Level8 occurs only once
versus twice. The declared depth/count rank therefore does not qualify for a
secondary audit or promotion. No Level9/10 is established.

Checkpoint hashes, exact configuration, boot offsets, saved learner/SIL RNG
and inherited selection records all verify. Source/reward/action accounting
through **32,576,507** passes **64 suffixes / nine complete full-game
coverages**, with no guards, archived entries or completed restored segments
yet. The sole lower-rate learner continues its unchanged stage; selected
32M weights and fresh-test seeds remain protected.

The [first same-run archive audit](results/level10/timing50k-stride2-lr5e6-long-first-archive-audit.json)
through **32,853,916** passes **110 suffixes / 25 complete full-game
coverages**, six publications and no guards. No completed restored segment
is claimed yet. The first Level-5 entry is worker30's own from-boot game at
**32,756,383** (score280, localaction23,445, bootoffset85,875). Protected
worker10 independently publishes Level5 at **32,768,299**, then worker29 at
**32,795,934**. These are actual same-run changing-policy experiences, not
validation/replay imports or new frozen-model milestones.

The [33M lower-rate audit](results/level10/timing50k-stride2-lr5e6-long-33m-primary-audit.json)
at **33,005,568** is **20/20 complete**: mean **238.75**, median **218.5**,
best **557**, highest **8**, counts 2-8 **18/14/10/6/5/3/1**. Seed 10004
reaches Level 8 in 48,537 actions. The rank trails the selected reference and
the first lower-rate checkpoint, so no secondary audit or promotion follows.
Configuration, checkpoint hashes, boot offsets and saved RNG/selection state
verify. Accounting through **33,093,709** passes **160 suffixes / 39 episode
coverages**: 34 full games and five completed restored segments, no guards.
The lower-rate training stage continues unchanged.

The [third primary audit](results/level10/timing50k-stride2-lr5e6-long-33m5-primary-audit.json)
at **33,505,280** is **disqualified**: seed **10006** reaches the 200,000-action
guard at **425 / Level 7**, still playing rather than waiting for a serve,
without GAME OVER. Nineteen other games complete, including seed 10002 at
578 / Level 8 in 51,775 actions. Their mean 243.68 and median 274 describe
only those completers, not an eligible twenty-game performance result.
There is no secondary audit or promotion. Hashes, boot offsets, saved state
and unchanged selection records verify; training resumes normally.

Source/reward/action accounting through **33,606,215** passes **254 suffixes /
69 episode coverages**: 46 full games and 23 restored segments. Restored
practice reaches 8, while complete from-boot training reaches 7. There are
no training guards or runtime errors; the validation guard is separate and
its trajectory never enters training. Final-test seeds remain unused.

The [first full Level-8 training audit](results/level10/timing50k-stride2-lr5e6-long-first-full-level8-audit.json)
confirms worker 22's first from-boot game finishes at **33,635,511** with
**559 / Level 8**, **50,918 actions**, boot offset 135,458. All four learning
suffixes exactly cover its actions, including correctly excluded prefixes.
The first completed restored Level-8 segment is worker 28 at **33,504,765**:
it starts at 371 / Level 7 from worker 18's same-run entry and earns **188
new points**. No archive or game state is imported from validation.

Accounting through **33,895,812** verifies **332 suffixes / 105 episode
coverages**: 59 complete full games, 46 complete restored segments, no
training guards. These games use changing weights; they confirm legitimate
training experience, not a new frozen-policy milestone or Level-9/10 reach.

The [fourth lower-rate primary](results/level10/timing50k-stride2-lr5e6-long-34m-primary-audit.json)
at **34,004,992** is **20/20 complete**, mean **262.8**, median **272**,
best **498**, highest **7**, counts 2-7 **20/17/13/11/4/2**. Seed10005 stays
at421 / Level7 for an extended period, then finishes at **146,223 actions**
without additional scoring or intervention. No validation guard fires.
The complete suite trails the selected reference's depth/count rank; no
secondary audit or promotion follows. Checkpoint/configuration/boot/RNG and
selection records verify. Accounting through **34,125,784** passes **376
suffixes / 124 episode coverages** (62 full games, 62 restored), no training
guards. The eight-million-action lower-rate stage continues unchanged.

### Lower-rate control completed; Level-8-focused practice trial

The [lower-rate control](results/level10/timing50k-stride2-lr5e6-long-completion.json)
exits **0** at **40,009,728**, after **8,003,584 additional actions** and
**6,518.75 seconds**. Fourteen of sixteen primaries are eligible; all fourteen
reach Level 5, with 74 such games and highest level 8. None exceeds the
selected 32M primary rank, so no secondary audit or promotion is warranted.
Besides the earlier 33.5M guard, the 36M suite has seed10008 incomplete at
397 / Level7 after200,000 actions. Final primary: **143.85 / 119.5 / 425 /
Level7**, all20 complete. Terminal weights remain preserved, not selected.

The complete [archived log](results/level10/logs/level10-timing50k-stride2-lr5e6-long.jsonl)
has **3,401 lines / 2,008,975 bytes**, SHA-256
`b3caf6fddc25e744140b0f439e6c1f98264a884f74ad031cb4a1ed31b6fd44cc`.
The audit verifies **239 complete full games / 439 complete restored segments**,
highest8 in each, plus one guarded full game (worker13,264/Level4,waiting for
serve,200k actions). All1,778 learning suffixes /679 episode coverages reconcile;
the guarded final4,096-transition suffix is discarded, not treated as success.

Only **98** practice segments start at Level8: mean **41.70** new points,
median46,best62, **zero clears**. First/last20 means are33.8/34.8. Earlier
practice levels are cleared in75/119 Level5 starts,52/108 Level6 starts and
35/114 Level7 starts. These observations motivate a matched practice-allocation
trial, not a claim that running longer alone has improved the policy.

The [Level-8-focused protocol](results/level10/timing50k-stride2-min8-protocol.json)
keeps the same source,8M-action budget,5e-6 learning rate,32 workers/16 protected,
timing,sampling and SIL settings. Only **minimum practice level5→8** changes.
Archives and buffers begin empty; all workers boot until the policy itself
discovers eligible Level8+ entries. No validation/replay states are imported.
All **103 tests** pass, including new threshold/atomic-peer/protected-boot tests.

Sole learner **74142**, `runs/level10-timing50k-stride2-min8`, launches only
after control exit/log/source checks. Its [startup audit](results/level10/timing50k-stride2-min8-launch-verification.json)
verifies the exact matched configuration except run path and minimum level,
32 CPU-only worker roles, and initial accounting through **32,109,877**:
15 suffixes /one completed full game, no guards or imported practice. Check
the expected common prefix against the control before the threshold affects
training. Final-test seeds remain untouched; Level9/10 is not reached.

The [first focused-trial checkpoint audit](results/level10/timing50k-stride2-min8-first-primary-audit.json)
at **32,505,856** verifies the expected common prefix: model and optimizer
files are byte-identical to the matched control, all non-configuration saved
learner/RNG state agrees, and the entire evaluation dictionary matches.
All **20 games complete** at **272.4 / 266 / 544 / Level 8**, counts 2-8
**18/16/13/8/6/4/1**. This is reproducibility, not a new improvement or
promotion; the selected 32M reference still has the stronger depth/count rank.
Accounting through **32,541,955** verifies **61 suffixes / eight full games**,
no archives, restored segments or guards. The practice-allocation comparison
continues as each run develops its own permitted training experience.

The [33M focused-trial primary audit](results/level10/timing50k-stride2-min8-33m-primary-audit.json)
at **33,005,568** is **20/20 complete**, mean **213.4**, median **221**,
best **530**, highest **8**, counts 2-8 **17/13/10/6/3/2/1**. Seed 10014
finishes at Level 8 in 48,386 actions. The suite trails the selected reference;
no secondary audit or promotion follows. Checkpoint hashes, configuration,
boot offsets, reward totals, saved RNG and selection records verify.
Accounting through **33,109,893** passes **153 suffixes / 37 full games**,
with no archives, restored segments or guards. Complete changing-policy
training reaches 6, not yet an eligible Level-8 practice entry. Validation
remains isolated from training; continue the declared focused trial.

### Focused trial completion and self-reference retention comparison

The [focused trial completion audit](results/level10/timing50k-stride2-min8-completion.json)
verifies exit 0 at **40,009,728**, after **8,003,584 additional actions** and
6,452.16 seconds. All 16 primary suites complete and reach at least Level 5,
but none exceeds the saved 32M reference's depth/count rank. Highest level
remains 8; final primary is **120.3 / 102 / 278 / Level 5**. No promotion.

Training completes **273 full games / 548 Level-8-start practice segments**,
with no guards. All 1,500 learning suffixes / 821 episode coverages reconcile.
Three genuine from-boot Level-8 entries seed the same-run archive. Practice
earns mean **32.88**, median **32**, best **62** new points, with **zero clears**;
first/last 100 means are **31.22 / 29.05**. The byte-identical archived log has
2,918 lines / 1,858,150 bytes, SHA-256
`93d7a2166c9aafa025448101b34ec275d3d87a7641b98efb897a5a061e52424c`.

The [next predeclared protocol](results/level10/timing50k-stride2-min8-refkl01-protocol.json)
tests retention, motivated by declining full-game validation despite focused
practice. PPO adds **0.1 × KL(frozen source policy || learner)** on current
learner rollout screens. The reference is the agent's own selected 32M RL
checkpoint; it never acts or updates. SIL remains unchanged. This adapts the
student-trajectory auxiliary objective in [Kickstarting Deep Reinforcement
Learning](https://arxiv.org/abs/1803.03835), using a fixed coefficient and PPO,
not the paper's IMPALA/PBT schedule. It is a hypothesis, not a demonstrated
fix for forgetting. No demonstrations or validation trajectories are imported.
Disabled exact-compatibility, enabled smoke, and unit/full-suite gates precede
the otherwise matched 8M-action trial. Final-test seeds remain unused.

The [implementation gates](results/level10/reference-kl-implementation-verification.json)
pass: disabled before/after 65,536-action smokes produce byte-identical model
and optimizer files, identical non-configuration learner state and identical
learning/episode/validation records. The enabled smoke exits 0 with finite
KL/losses, nine reconciled learning suffixes and unchanged source weights.
Its deliberately guarded validation is ineligible; no smoke weights are
selected or resumed. Five new tests cover analytic gradients, frozen targets,
metadata/hash validation, additive loss, optimizer resume and standalone
inference. The full suite passes **108 tests** in 11.150 seconds.

The [performance launch audit](results/level10/timing50k-stride2-min8-refkl01-launch-verification.json)
confirms sole learner **20492** (Python PID **32532**), matching the completed
focused control except the declared reference settings and run path. All
32 workers are CPU-only; 16 stay protected for from-boot play. Initial
accounting through **32,110,651** verifies ten learning suffixes, no completed
games yet, no archives/restored practice/guards. Mean PPO-minibatch reference
KL is finite (latest 0.00320); throughput is about 1,319 actions/second at
this startup check. First primary is **32,505,856**. No performance gain is
claimed from these implementation checks, and the saved reference remains
the same standalone model.

### Reference-penalty completion, new selection, and persistent supervision

On September 16, a fresh process check found the reference-penalty learner had
finished normally about46 hours earlier; it was not still training. The
chat goal reported `usageLimited`. Leaving a finite learner alive had not
provided ongoing agent supervision, and the earlier status messages should
have made that distinction clear. No usage-limit setting was changed.

The [completion audit](results/level10/timing50k-stride2-min8-refkl01-completion.json)
verifies exit0 at **40,009,728**, **8,003,584** new actions, **6,970.67 seconds**,
all16 primary suites complete,112 Level5-or-higher games and highest8. Training
completes222 full games /331 restored Level8-start segments, zero guards;
all1,394 learning suffixes /553 episode coverages reconcile. Practice earns
mean48.60, median53, best62 new points, zero clears; first/last100 means are
47.58/47.02. The archived log has2,564 lines /1,733,943 bytes, SHA-256
`e66ebad64f6b8e2287b08f93613be4795be30f83b3da04c8350adfab58764264`.

Checkpoint **35,504,128** is the sole primary qualifier against the saved32M
reference: mean240.95, median196.5, best576, three Level8 reaches in20 games.
Its [secondary audit](results/level10/refkl01-35m5-secondary-audit.json) completes
50/50 at209.64 /197 /552, one Level8 reach. Combined70-game counts2-8 are
64/41/32/23/10/7/4, mean218.59, median197, best576. The earlier reference has
66/57/45/26/11/8/3 and mean254.84. This is a depth-first ranking improvement
with regressions elsewhere, not uniform improvement or fresh generalization.
An initial audit script accidentally loaded the old primary in place of its
secondary; its40-game comparison was caught and replaced before promotion
using strict20/50 seed checks. The corrected audit compares70 games per model.

The [promotion record](results/level10/refkl01-35m5-promotion.json) includes
the complete seed10019 replay at576 /Level8 /49,715 actions; all actions
match the frozen neural policy, with no serve overrides. No Level9/10 or
fresh final tests occur. Previous models and replays remain preserved.

The [supervised continuation protocol](results/level10/supervised-continuation-protocol.json)
keeps the selected learner's training recipe and32M frozen reference unchanged,
but removes the arbitrary action-budget cutoff. A local Python supervisor
checks liveness, logs, disk and code identity, automatically audits new
primary records, verifies promotions/replays, and reserves the fresh100-game
test for a verified Level10 policy. It reports explicit review conditions
instead of silently ending a batch; it does not call Codex or bypass limits.

The first launch was rejected before training because two mutually exclusive
budget flags were supplied. The supervisor detected the exit and wrote
`needs_attention`; that record is preserved. A real CLI regression test now
guards the fix, and all **119 tests** pass. The corrected launch uses only
`--steps0` (the non-inherited additional limit defaults to0).

The [startup verification](results/level10/supervised-continuation-startup-verification.json)
confirms session **98423**, supervisor PID3280, learner PID3283, sole learner
under `runs/level10-supervised-refkl01-v2/learner`, with both parsed action
limits0. All learning settings match the verified35.5M source. Through
**35,694,614**,41 learning suffixes /five full-game coverages reconcile;
no guards, archives or restored segments yet. First primary is36,003,840.
The live heartbeat and automatic selection records, not this dated narrative,
are authoritative as training proceeds.

The [first complete supervised cycle](results/level10/supervised-continuation-first-cycle-verification.json)
was verified at23:46 PDT on September16. Primary36,003,840 finishes20/20 at
mean192.35 /median193.5 /best524 /Level8; counts2-8 are17/12/6/5/2/1/1.
It trails the selected primary, so no secondary or promotion occurs. The
supervisor records the result and the same learner resumes updates, advancing
to36,477,239. All162 suffixes /37 full-game coverages reconcile, zero guards;
both processes are live and selected/teacher hashes remain unchanged. This
verifies the live primary-to-resumed-training path in addition to119 passing
tests; automatic promotion and final-test gates remain unit-tested but have
not yet been triggered by this new performance run.

### September19: long-run plateau, durable replays, within-level curriculum

The [completed-run audit](results/level10/supervised-v2-completion-review.json)
verifies200,499,200 new actions,175,807.4 seconds,400 finished primaries
(363 complete),4,395 complete full training games and11,644 complete restored
segments. All33,173 learning suffixes /16,060 episode coverages reconcile;
21 truncated suffixes are discarded. The last scheduled validation was
cancelled by the intentional operator stop, not counted as a finished suite.
The supervisor exits1 for the operator interruption and records
`operator_stopped`; the learner saves its terminal checkpoint. No data deleted.

Automatic audits promoted106.5M then116M. No further promotion occurred over
roughly120M later actions. Selected116M's model SHA-256 is
`5b5ea6721db0284ce7bac56db2920144225f083eb2f4e11455464052c3849b85`.
Its primary mean334, secondary mean308.06, combined mean315.47; ten of70
reach8. Replay581/Level8/seed10120 is fully reproduced and inspected. A scan
of19,117 complete frozen-game records (including reused/duplicate records,
not independent tests) identifies598/Level8/seed10008 from66.5M as the best
single effort; its56,764 actions reproduce with zero mismatches. Neither
result establishes Level9 or10. The new versioned exports retain both notions
of best separately and automatically update standalone stable replay files.

The [new protocol](results/level10/progress16-protocol.json) extends only the
training-state archive: also capture after16 new visible score points within
an eligible level, using the same reservoir capacity. Rebase the progress
offset on every reset/restore/level change, never reuse prior score as reward,
never load archived states from files, and never modify native state contents.
This adapts the return-to-own-promising-states principle of
[Go-Explore](https://arxiv.org/abs/2004.12919), not its full algorithm or an
established explanation of the plateau. Protected boot workers and ordinary
from-boot frozen validation remain unchanged. The selected replay loses balls
with Level8 partially cleared; that observation motivates this test but does
not prove that state coverage is the only limitation.

[Implementation gates](results/level10/progress16-implementation-verification.json):
129 tests pass. Disabled before/after65,536-action runs have byte-identical
model/optimizer files, identical nonconfig learner state and learning events.
Both smokes exit0 and neither becomes a source of training states or weights.
The performance run resumes the real selected116M checkpoint directly, with
empty training archives and replay, preserved learner/optimizer/RNG state,
and the unchanged frozen32M reference penalty. Fresh seeds remain unused.

The [first live cycle](results/level10/progress16-first-cycle-verification.json)
completes20/20 at116,506,624: mean272.65, median282.5, best524, highest8;
counts2-8 are19/17/16/11/4/1/1. It does not qualify for promotion. The
supervisor consumes the result and the same learner resumes updates. Through
116,629,504 all59 learning suffixes /11 full-game coverages reconcile, zero
guards. No training archive yet at that snapshot; the curriculum cannot use
Level8 states from validation. It must rediscover them during training.

Post-hoc rendered frames are retained for the
[selected581-point game](results/level10/selected116m-level8-frames.png) and
[single-best598-point game](results/level10/best598-level8-frames.png). The
latter enters Level8 at538 points and earns60 more before losing its last
ball with blocks remaining. In the selected replay, every non-ASCII playfield
cell is within the model's supported128–191 graphics range (no unencoded
high-byte cells); this rules out that specific encoding omission for this
game, not all possible perceptual limitations. These diagnostics introduce
no gameplay, training states or policy labels.

At07:52UTC, [live archive verification](results/level10/progress16-live-archive-verification.json)
passes through117,291,131: two own Level8 entrances and four within-level
progress publications, each after16 newly earned points. Worker20's publication
at534 traces to worker26's same-run518-point entry; it earns16 new points,
does not reuse the prefix reward, and is not a full-game result. Original
worker26 also reaches a550-point archive. All159 learning suffixes/35 full-game
coverages reconcile, zero guards. Protected boot workers retain their roles.
This proves enabled data routing and accounting, not improved performance.

### September 19: confirmed eight-level cap; goal decision required

The [game's author](https://pski.net/breakdown-a-new-trs-80-game/) says the game
ends after all eight levels. Read-only static analysis confirms that behavior
in this repository's `var/breakdown.cmd`, SHA-256
`0eada4c36fbdb135ca14f390cdb3e01a3a3da121a1328a531fcc6c1e2e931ee6`.
The existing CMD parser loaded a separate Python bytearray, not a live emulator.
No policy input, reward, training state, game code or emulator was changed.

The verified level-transition instructions are:

```text
5869  LD A,(6316h)     ; current level
586C  INC A
586D  LD (6316h),A
5870  CP 9
5872  JP Z,5D17h       ; GAME OVER
5875  CALL 573Bh       ; level HUD update (skipped at 9)
```

Routine 573Bh formats the level into video addresses 3C3Bh–3C3Fh; routine
5D17h writes GAME OVER on row 10. Restart resets the counter to 1. The binary
contains eight level titles, ending with BLOCK CITY. Exact bytes and addresses
are preserved in the [cap audit](results/level10/game-level-cap-audit.json).

This invalidates the assumption behind the requested Level-10 target. It also
invalidates interpreting zero transitions to displayed9 as zero final wins:
both victory and defeat use GAME OVER, and the current tooling does not
distinguish them. The two inspected best replays end with blocks remaining;
no all-eight-level victory has been verified. No claim of Level10 completion.

The [progress16 completion audit](results/level10/progress16-completion-review.json)
verifies the intentional stop at120,102,912 (+4,096,000), after eight complete
20-game primary suites. Last suite: mean283.9, median282.5, best527, highest8.
No suite qualifies for secondary promotion; selected116M remains unchanged.
All877 learning suffixes /546 episode coverages reconcile across114 complete
from-boot games and432 complete restored segments, with zero guards. Five own
Level8 entrances and235 score-progress publications have checked provenance.
These are training practice counts, not independent complete-game evaluations.

The supervisor, learner and all32 workers have exited; final model, optimizer,
state and byte-identical log archive are retained. Both replay bundles' file
hashes and stable HTML copies are rechecked. All130 tests pass, including a
four-point-spacing curriculum unit fixture; no progress4 experiment began.

The user has been asked to choose beating all eight original levels or
explicitly extending the game. Until that decision, training is paused and
GOALS.md/game code remain unchanged. A revised all-eight target first needs a
verified screen-derived win criterion, not a cosmetic target-level change.
Final-test seeds40000–40099 remain untouched. No new public deployment or push.
