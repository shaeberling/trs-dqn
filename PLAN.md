# Execution plan

The emulator already exists; use its Python API and compile its existing C
library only as required by README setup. Stay on the current branch. No prior
solutions, demonstrations, privileged policy inputs, or scripted paddle control.

User decisions: use the Mac Mini's available CPU/GPU, no wall-clock training
limit, and target clearing level 1 as the first completion milestone. All final
evaluation games must run to GAME OVER (at least ten).

1. Review and validate the existing emulator, screen observations, score and
   terminal detection; measure emulator and neural-network throughput.
2. Implement a screen-only RL environment and replay-based neural agent, with
   explicit reproducible seeds, logs, checkpoints, and resume support.
3. Train from the agent's own experience and score rewards, evaluating periodic
   checkpoints on a fixed validation protocol. Record experiments and changes.
4. Evaluate the chosen trained model on ten or more held-out complete games;
   report mean, median, best score, highest level, and number of level-1 clears.
5. Provide saved weights, watch mode, reproducible commands, and results notes.

## Review findings

- `TRS.boot`, `run_for_tstates`, `Keyboard`, and video RAM expose the necessary API.
- The native emulator has global state: parallel environments require processes.
- The graphics screenshot omits text; video RAM contains score/status text.
- Clean checkout had no `libtrs.so` or installed Python dependencies.
- The host is Apple Silicon with ten CPU cores. Benchmark Metal acceleration
  and CPU execution before committing to the training backend.

## Progress

- [x] Read GOALS.md and README.md; agree scope and completion target.
- [x] Review existing emulator implementation.
- [x] Validate environment and benchmark compute (MLX selected on Apple M4).
- [x] Implement training/evaluation/watch workflows; numerical and environment tests pass.
- [x] Train a model that clears level 1; frozen checkpoint verified by recorded level transition.
- [x] Run final evaluation: 50/50 held-out complete games, mean 44.3, median 45.5,
  best 88, highest level 2, three level-1 clears. Document results and save artifacts.

The final evaluation was expanded from the minimum ten games to 50 before
running it. Training stopped cleanly after the requested first-clear milestone;
the trainer's default, stronger automatic stop of three clears in five validation
games was not required for this milestone. That initial run was stopped.

## Level-5 follow-up (completed)

Approved follow-up: reach displayed level 5 (clear levels 1-4), then report
the selected checkpoint's performance on 100 fresh complete games. Preserve
the level-2 model and published replay. See [LEVEL5.md](LEVEL5.md).

- [x] Add configurable target levels, complete-game level reach rates, and a
  separate best-by-level checkpoint, with tests.
- [x] Measure the original checkpoint on the expanded validation suite:
  20/20 complete, mean 51.5, median 55.5, best 66, level 2 in 1/20.
- [x] Run the initial unchanged continuation (planned cap: 20 million additional
  actions), investigate its terminal-parser failure, and preserve its checkpoints.
  The cap was not exhausted; experiments adapted from the measured results.
- [x] Compare learning rate, then exploration strength, while preserving control
  runs. The lower-rate/lower-entropy continuation produced a frozen level-5 game.
- [x] Freeze the selected checkpoint and evaluate 100 fresh complete games:
  mean 83.32, median 72.5, best 167, highest level 3, 0/100 level-5 reaches.
- [x] Record the 278-point level-5 validation replay and package weights, optimizer,
  evaluation records, archived logs, checksums, and reproduction instructions
  without overwriting the original model or public replay.

The first-reach target is verified, not reliable mastery: the selected model
reached level 5 in 1/20 validation games, but not in the 100-game final test.
Training stopped automatically at the validation target and was not resumed
using the test results. Details and provenance are in [LEVEL5.md](LEVEL5.md).

## Level-10 follow-up (in progress)

The user requested committing the level-5 milestone, then improving toward
displayed level 10 (clear levels 1-9). The milestone is committed as `f4c77d6`;
the original and level-5 packages remain immutable. See [LEVEL10.md](LEVEL10.md).

- [x] Commit level-5 code, model, replay, logs, and results after 25 passing tests.
- [x] Reserve fresh final-test seeds 40000-40099 before training; do not use
  them for model selection. Earlier published test suites are not fresh tests.
- [x] Start continuing the MLX PPO policy from the frozen level-5 checkpoint, initially
  for five million additional actions, evaluating every 250,000 actions.
- [ ] Track complete-game level reach rates and mean/median/best score. Compare
  promising checkpoints on secondary validation seeds 10100-10149; adapt
  experiments if improvement stalls.
- [ ] Verify a frozen checkpoint reaches displayed level 10 in a complete
  validation game, then evaluate 100 fresh complete games without reselection.
- [ ] Record a full-game replay and package the model, logs, tests, and honest
  results separately from the existing milestones.

The five-million-action first tranche is an experiment checkpoint, not a wall
clock limit or completion condition. Level 10 is a stretch target, not promised
performance; a first reach and reliable later-level play will be distinguished.

Current experiment decision: the control was paused after 2,080,768 additional
actions. The smaller-learning-rate trial completed 2,002,944 actions and
improved primary/secondary validation, with a reproduced level-5 game. Its
best-by-level checkpoint's extension was paused after 2,179,072 additional
actions without a new validation depth record. A controlled lower-entropy trial
completed 2,002,944 actions and improved the primary mean to 132.6, reproducing
level 5 in 1/20 complete games. Continue its final validated checkpoint for a
ten-million-action tranche. Its secondary 50-game validation audit completed
with mean 122.6 and two level-5 games; the 283-point game independently
reproduced in a full replay. The longer-reward-horizon comparison completed its
two-million-action budget without an improvement. Optional parallel validation exactly reproduced all
20 preserved serial game records with both 8 and 20 workers; the new
continuation uses 20, with separate selection-protocol records. Its first stage
was recovered at 22,253,568 after a confirmed serve-stalled evaluation. The
recovery kept the original 30,500,480 budget endpoint and unchanged
learning settings, adds evaluation progress/cancellation, and uses a 100,000-action
validation guard that disqualifies every incomplete suite. Final test and target
replay remain uncapped. After seven consecutive complete validations missed
level 5, paused it cleanly at 27,598,848 (7,098,368 total actions in both long
stages), preserving its final checkpoint and full raw log. Its freed capacity
now runs the verified self-generated curriculum trial from the stronger
20,500,480 source. See LEVEL10.md for results, provenance, and active commands.

A separate finer-timing training trial now adapts the verified 20,500,480
checkpoint to 50,000 T-states per action. It preserves nominal temporal
discount horizons and rollout/update cadence through matched coefficient,
rollout, minibatch, interval, and guard adjustments; four million new actions
match two million original actions' nominal emulation time. This is a distinct
protocol comparison. It completed at 24,506,368 (4,005,888 additional actions):
seven complete validation suites and one serve-stalled, disqualified suite;
none of the complete suites reached level 5. Preserve its checkpoints/log and
retain original timing for the curriculum experiment.

Prepare an optional self-generated start-state curriculum after inspecting the
very short level-5 portions of existing replays. Exact native snapshot
restoration, unchanged complete-game evaluation, and integration checks passed
(55 tests including deferred worker GPU imports). The separate two-million-action
trial is now active after the declining original-timing long run paused,
avoiding a third full learner under tight host memory.
Only newly self-reached training states may enter its archive; restored segments
must never count as complete games or target success. Keep from-boot validation,
score-only rewards, screen-only inputs, and the held-out final test unchanged.
The curriculum's first validation was disqualified by one serve-stalled game;
learning continued without an action override.
Its next checkpoint completed 20/20 games, mean 143.95, highest level 4;
fresh self-play has also generated a level-5 entry for curriculum practice.
It later completed its tranche at 22,503,424, with seven complete validation
suites and one disqualified suite; none reached level 5. Its eight completed
level-5 practice starts averaged only 1.375 new points. Preserve its snapshots
and raw log; the peer-shared comparison is now the only active learner.

At roughly 21.35 million actions only two workers had level-5 entries and one
completed practice segment started there. A matched peer-sharing comparison
now distributes same-agent discoveries among all workers. Cross-process
restoration and policy-input isolation pass (59 tests); real shared-reset and
unchanged-local smoke checks passed before launch. Both comparisons start from
the same 20,500,480 source with empty archives and score-only PPO unchanged.
The shared trial's first validation completed 20/20 games, mean 122.75, highest
level 4. Its 21,000,192 checkpoint subsequently reached level 5 in two of 20
complete games (mean 130.3); the 21,504,000 checkpoint reached it once (mean
136.35). The former's secondary audit completed 50/50, mean 135.12, median 121,
best 304, with two level-5 reaches. This improves the source's secondary mean
and intermediate-level counts, but not its level-5 count or highest level.
Continue with a separate focused curriculum from that audited checkpoint:
archive only newly self-reached levels 5+ and choose a restored start with 95%
probability once available, preserving from-boot validation and score-only
learning. The reset fraction does not equal the fraction of training actions.
The ordinary shared trial completed at 22,503,424: all eight validation suites
were complete, with no new highest level. Its full log/checkpoints are archived;
all 242 restored segment origins and score differences were checked. The focused
trial is now the only active learner. Its first complete validation mean was
129.5, highest level 4, before any eligible training archive existed. All 59
regression tests pass. Level 10 remains unverified and the fresh final test
remains unused.

The focused trial subsequently completed at 23,003,136: seven complete
validation suites and one disqualified serve stall, no new highest level.
Actual same-start level-5 practice improved, with the last 100 segments
averaging 10.22 new points versus 1.04 for the first 100; all 2,110 segments
still ended at level 5. Archive the full log and continue its final weights
for ten million additional actions with unchanged settings. Preserve the
stronger audited shared checkpoint as the public performance reference.
The resumed learner starts with fresh archives and must rediscover its own
entries; no validation/replay states are imported. The full level-10 objective
and untouched final test remain unchanged.

The long run has reached levels 6-8 in restored training segments, but not in
frozen from-boot validation. Its restored-action share rose from 36.20% to
76.98% while earlier-level validation weakened. Add and verify an optional
fixed reservation of 16/32 workers for from-boot play, then compare from the
same 23,003,136 source with other settings unchanged. Keep the current learner
running as the control. All 64 tests and the disabled-option exact-update smoke
pass; both real reserved-reset integration checks also exited successfully,
including completed practice sourced from protected collectors. Launch the
matched `level10-curriculum-balanced` comparison with fresh archives and 16
reserved workers. It is now active: worker roles, source hash, exact saved
configuration differences, finite updates, and initial action-origin counters
are verified. Training-only depth remains separate from the level-10 success
gate. Leave both learners running and compare complete-game validation at
matched action counts before selecting a continuation.

The balanced run reproduced the first three model/optimizer/evaluation files
exactly and independently rediscovered the same level-5 training entry. Real
practice now exercises the protected-reset difference with verified provenance
and action counters. The control's latest eight validations reach only level 2;
847 level-8 practice segments have not reached level 9. Unless new depth arrives,
pause the control cleanly after its next scheduled validation near five million
additional actions, retain its evidence, and give compute to the balanced run.
This is an adaptive shortening of the control budget, not a wall-clock cutoff.

The control subsequently exited cleanly at 28,045,312 (+5,042,176). Its last
validation still reached only level 2; all 5,435 raw log lines are archived
byte-identically and all 4,454 completed practice origins/rewards are checked.
The balanced run is now the sole learner. Its first two post-divergence
validations have not established an improvement, so continue monitoring its
full-game results and deeper practice without selecting a new final model.

The balanced run also reaches restored level 8, but its complete-game gains
are not consistent: a stronger 24.75M suite is followed by a lower 25M mean.
Add an otherwise matched protected-boot comparison at learning rate 1e-5,
from the same 23,003,136 source and fresh archives, with an initial two-million-
action tranche. Leave the existing balanced learner unchanged; compare at
matched action counts and retain the stronger audited model as the reference.

The 1e-5 trial is now active and its source/config differences, 32 worker
roles, finite updates, and initial origin counters are verified. The original
balanced run's 25.25M validation improves on the matched unprotected control
(103.7 / level 4 versus 76.45 / level 2), still without new full-game depth.
Continue both trials and keep restored reaches separate from the success gate.

The smaller-rate run's first checkpoint reaches level 5 / score 313 in primary
validation, with a full replay and exact seeded-action verification. Its
secondary 50-game mean is 111 with one level-5 game, weaker than the retained
reference; do not promote it. The learner also completes a from-boot level-6
training game (334 points), but its policy changed during the episode. A later
frozen checkpoint's secondary audit reaches only level 4. Keep training-level
progress distinct from frozen success, continue both runs, and use subsequent
complete validations to decide the next continuation. SIL was reviewed as a
conditional alternative, not implemented; no training objective has changed.

The smaller-rate tranche completed at 25,006,080 with a second full training
game reaching level 7 / score 411. Exact episode offsets and all practice
origins/rewards pass; frozen validation still tops out at level 5. The fixed
last-five checkpoint average completed 20/20 validation at mean 112.25 and
level 4, so do not promote or resume it. All 71 tests pass.

The 5e-5 balanced run was paused cleanly at 28,815,360 after 19 consecutive
complete validations missed level 5. Both finished logs are archived exactly.
The smaller-rate final model/optimizer/RNG now continues in
`runs/level10-curriculum-balanced-lr1e5-long`, with unchanged learning settings,
fresh archives, and a ten-million-action initial ceiling. Its first complete
validation reaches level 5 / score 306. Continue toward 10 and review frozen
validation, not changing-policy training depth, before selecting any final model.

Alongside that continuation, isolate longer GAE credit assignment: lambda 0.99
versus 0.95, same final 25,006,080 checkpoint, original gamma/timing and all
other learning/curriculum settings unchanged. Use a five-million-action initial
tranche, compare matched complete primary validations, and audit any deeper
frozen checkpoint separately. This is a bias/variance experiment, not a claimed
diagnosis or new reward. No final-test games are used for selection.

Both runs remain active. The unchanged continuation completed another full
training game at level 6 / score 348 and rebuilt restored level 8; its latest
complete primary reaches level 5. The longer-trace run completed a from-boot
training game at level 7 / score 381. Audit its nearby frozen 25,751,552
checkpoint on the existing secondary 50-game set to test transfer. Do not
promote changing-policy training outcomes or use the fresh final set.

That frozen longer-trace audit finished 50/50 complete, mean 140.1 and three
level-5 reaches. Its selected 286-point replay reproduces every action exactly.
Preserve it as a candidate, but keep the reference: combined reused-set mean
improves while the level-5 count declines from four to three. Neither model
has frozen level 6. Continue both existing learning budgets; the fresh final
test set remains unused, and the Level 10 objective is unchanged.

The unchanged continuation's 28,504,064 checkpoint improves both reused sets:
primary 166.7 with two level-5 reaches; secondary 141.18 with four, all complete.
Its 327-point full replay matches exactly. Adopt it as the validation comparison
reference while retaining previous candidates; no new frozen depth or fresh-test
generalization is claimed. Continue both existing budgets unchanged.

Optional self-imitation is now implemented, disabled by default. All 83 tests
and three emulator smokes pass: disabled byte compatibility, real auxiliary
updates with exact own-score accounting, and complete rejection of truncated
suffixes. No diagnostic checkpoint may seed performance training. Plan a
separately named, matched own-experience SIL trial once the active longer-trace
comparison is reviewed; do not change either live learner mid-run or use
validation/replay files as training input.

Predeclare that future SIL comparison from the same 25,006,080 learner as the
two current trials, fresh buffers and unchanged lambda-0.95 control settings,
with four auxiliary updates/rollout, batch 512, capacity 32,768, suffix 2,048,
loss/value weights 0.1/0.01 and priority alpha/beta 0.6/0.1. Review its initial
five-million-action tranche against the matching control lineage. Keep the
stronger 28.5M frozen model separately; no SIL performance process is active yet.

The longer-trace trial's best primary-ranked checkpoint among its first 15
complete suites (27,000,832) was additionally audited on the reused 50-game
set: mean 119.38, three level-5 reaches, all complete. Its combined results
are weaker than the retained 28.5M reference, so do not promote it. Let the
remaining planned tranche finish, then review and use that slot for the
predeclared SIL comparison. The current two-run chart is saved separately
from the older comparison chart; it excludes restored practice.

The longer-trace tranche finished cleanly at 30,007,296 (+5,001,216), all 20
primary suites complete, none beyond level 5. It reached level 5 at four
checkpoints versus eight for the matched original continuation; the final
suite reached only 3. Archive its full log and learner state; do not extend it
now. The predeclared SIL comparison is now active from the same 25,006,080
source, while the original continuation remains unchanged. Initial source,
configuration, worker-role, own-score, finite-update and replay-bound checks
pass. Next verify the first complete SIL validations and restored-experience
accounting when its fresh curriculum discovers later-level starts. Keep the
28.5M model as reference and keep the final test unused.

The first SIL primary completes 20/20 at 25,251,840: mean 98.6, highest level 4,
below the matched control's 133.3/level 5. Do not promote or audit it further
on the secondary set merely because the implementation works. Continue the
declared trial, check later matched validations and eventual restored SIL
collection, and retain the stronger reference. The original learner's first
100,000-action training serve stall was truncated and excluded from performance.

Restored SIL collection is now exercised and verified on this run's own first
level-5 archive entry. All 38 completed practice sources/life boundaries and
330 suffix score sums checked so far pass; inherited score is not reward.
The corresponding full training game reached 5 with changing weights, not a
frozen milestone. Continue the declared trial and compare complete validations;
implementation correctness alone does not justify a reference promotion.

The second and third SIL primaries both reproduce level 5; the third reaches
it twice with mean 150.5 and six level-4 reaches. Its primary depth-count rank
exceeds the current reference's, so a frozen 50-game secondary audit is now
running on the existing validation seeds. Do not presume transfer or promote
it before that audit. Own restored practice also reaches 6, then 7 and 8 in
a single later segment; source/new-score and exact-offset checks pass. Keep
these training outcomes separate from the Level 10 completion gate.

That secondary audit finished 50/50 complete, mean 129.32 with one level-5
reach, below the reference's 141.18/four. The combined reused sets also favor
the reference, so do not promote the third SIL checkpoint. Continue its
declared budget and the original continuation, monitoring deeper frozen
results separately from restored level-8 practice. No final-test games are
used and no new replay is needed for this weaker audited candidate.

The current SIL run has also completed a verified from-boot training game at
level 6 / score 317, with changing weights. Frozen validation remains at 5.
Replay-retention accounting finds median full eviction after about 44,687 new
actions among already-evicted suffixes, while four 512-state updates per
4,096-action rollout supply only 0.5 nominal replay draws/new action. This is
a tuning hypothesis, not a demonstrated bug or performance explanation.

Predeclare a one-factor SIL-dose comparison: **20 versus 4 replay updates**,
same 25,006,080 source/model/optimizer/RNG, fresh buffers, unchanged capacity,
suffix, losses and all PPO/curriculum settings, five million new actions.
Use the original long continuation's slot only after its tranche completes and
is reviewed; do not add a third learner or change either active process.
Check source/configuration/origins/finite losses/memory, then compare complete
primary suites at matched action counts. Keep the 28.5M reference, secondary
audit rule and untouched final seeds. The exact prelaunch protocol is in
`results/level10/balanced-sil20-protocol.json`; this experiment is not yet running.

The current SIL run's tenth primary (27,500,544) reaches level 5 three times,
mean 134.5, all 20 games complete. It now leads this trial's primary depth-count
ranking and exceeds the reference's two primary reaches, although its mean
and intermediate-level counts are lower. Freeze this candidate for a secondary
50-game audit on reused seeds 10100-10149 before deciding on promotion; exact
primary records and source hashes are saved. Continue both live budgets and
keep the final test untouched.

The tenth SIL checkpoint's secondary audit is complete: 50/50 games, mean
121.82, one level-5 reach. Its combined reused sets trail the retained
reference (125.44 versus 148.47 mean; four versus six level-5 reaches), so
do not promote it or create another selected replay. Matched primary depth
counts across the first ten checkpoints favor SIL (nine level-5 games versus
two), but these reuse the same seeds and do not overturn the secondary audit.
Continue the remaining budgets and the queued, separately controlled replay-
dose comparison; no fresh final-test games have been used.

New depth candidate: SIL 28,250,112 completes 20/20 primary games with one
level-6 game (seed 10006, score 326, 15,034 actions). Seed order, completion,
score rewards and frozen checkpoint hashes pass initial verification.
Record that complete game uncapped and run the reused 50-game secondary audit
before reporting replay-verified level 6 or changing reference roles. Review
this candidate before the queued SIL20 launch; Level 10 is still the objective
and final seeds 40000-40099 remain reserved.

Frozen level 6 is now verified: the uncapped primary replay reproduces all
15,034 actions exactly, and the 50-game secondary audit also reaches 6 once
(337 points; mean 146.4, all games complete). Adopt SIL 28,250,112 as the depth
reference. Keep the original 28.5M model as the higher combined-mean and
level-5-consistency comparison; do not claim reliable level-6 play from two
reaches in 70 reused games. The new-depth review is complete. Continue SIL4
with its live buffers and launch the already-declared same-source SIL20
comparison only after the original long continuation ends and is reviewed.
The final Level 10 test remains unused, and no milestone package is overwritten.

The original long continuation has finished cleanly at 35,008,512, all 40
primary suites complete and none beyond level 5. Exact log archival and the
completion audit pass; preserve its 28.5M higher-mean checkpoint and final
learner state. The verified level-6 SIL checkpoint remains the depth reference.

The freed slot now runs the predeclared SIL20 comparison from the same
25,006,080 source as SIL4. Exact source hashes, the two intended configuration
differences (directory and update count), all 32 worker roles, initial score
suffixes, finite updates and bounded storage pass. Keep SIL4 running unchanged;
next assess SIL20's complete matched primary validations and verify its first
restored sources when generated. No validation/replay data enters training,
no final tests are used for selection, and Level 10 remains the goal.

The first SIL20 primary is now verified, all 20 complete: mean 114.2, highest
level 4, counts 15/4/2/0. This exceeds matched SIL4 (98.6, 11/2/1/0) but not
the no-SIL control (133.3, level 5). Preserve its exact raw evaluation and
checkpoint; do not promote or run a secondary audit for this initial result.
Continue the matched sequence and verify own restored SIL20 collection once
it generates later-level entries. The depth reference remains verified level 6.

The four complete SIL4 primary suites after the verified 28.25M checkpoint
reach only levels 4/4/3/4 through 29,253,632. Let the remaining tranche finish
and review its final candidates. If no stronger audited checkpoint emerges,
continue from the preserved 28,250,112 depth reference, not unselected later
weights: five million additional actions, all learning settings unchanged.
Model/optimizer/learner RNG/SIL RNG restore, but archives and replay start
empty; this is not uninterrupted trajectory continuation. The conditional
source, hashes and command are in `results/level10/sil-level6-continuation-protocol.json`.
Use only SIL4's freed slot after its verified completion, leave SIL20 unchanged,
and keep final seeds reserved. This continuation is not running yet.

SIL20's second primary also completes all 20 games: mean 96.3, level 3,
below matched SIL4's level 5 and no-SIL control's level 4. The first two
comparisons are mixed; continue the declared dose trial without promoting
an early checkpoint. SIL4's pre-completion audit through 29.54M passes all
own-source, suffix, raw-score and logged-action checks. Its final tranche
review and conditional continuation remain pending.

SIL4 completed cleanly at 30,007,296. All 20 primary suites, 3,828 learning
suffixes, 2,128 episode coverages, origins/rewards/pending counts and exact log
archival pass; no later candidate surpasses the verified 28.25M checkpoint.
Its final resumable state is preserved, not selected for continuation.

The conditional Level-6 continuation is now active from that checkpoint:
five million new actions, unchanged settings, fresh training buffers. Exact
source hashes, only metadata/budget configuration differences, worker roles,
initial suffix scores and finite updates pass. Next verify its first complete
primary and eventual own restored collection. Leave SIL20 unchanged and use
the archived original SIL4 run for its matched dose comparison, not this
differently restarted continuation. Keep the final test reserved for Level 10.

SIL20's own first level-5/6 restored collection is verified: all 60 completed
practice sources and 776 learning suffixes through 26.06M have correct raw
scores, origins, offsets and pending accounting. These practice reaches are
not frozen milestones. The Level-6 continuation's first primary is weaker
(93.2, level 4, all complete); preserve the verified source and continue the
declared tranche without promoting this initial checkpoint or treating the
single result as proof of a restart effect. Both learners remain active.

The resumed run's own restored collection is now verified, with later practice
through level 8. It has also completed a from-boot training game at level 6 /
score 366; the five frozen primary suites checked so far reach at most 5.
Keep changing-policy training outcomes separate from the selected f58 reference.
Source, raw-score, suffix, pending and logged-action accounting pass.

A bounded decoding comparison on f58 is finished: pure argmax has one complete
level-7 game but seven serve-stalled truncations, disqualifying the suite.
Uniform epsilon 0.01 and 0.001 each finish 20/20 games but reach only level 4.
Neither merits promotion or secondary evaluation; end the rate comparison and
retain sampled decoding. Both active training protocols are unchanged.

Predeclare the next capacity comparison in
`results/level10/sil-level6-cap128k-protocol.json`: 131,072 versus 32,768 replay
entries, four updates unchanged, same f58 source as the current restarted
continuation, fresh buffers and five million new actions. Finish and audit
both current budgets before launch; if a stronger depth source emerges, revise
the source decision first. Initially use the sole learner slot and recheck
memory/paging after the old runs exit. Screens alone rise from 384 to 768 MiB;
existing swap usage makes the free-percentage reading insufficient evidence
of headroom. This trial is not running. Preserve source hashes, sampled full-
game validation, model packages and the untouched Level-10 final-test seeds.

The continuation now has a verified from-boot **training** Level-7 game
(379 points, protected worker 11), not a new frozen-policy milestone. A
screen-only replay check also observes a sustained reserve-ball increase;
keep the existing reserve-decrease learning boundaries, not a hard-coded
three-boundary limit. No environment or policy change is warranted by that check.

The 08:17 UTC intermediate review passes all source/reward/suffix/pending
accounting with zero training truncations: continuation 342 full / 559 practice
games and eleven complete primaries; SIL20 491 full / 844 practice games and
thirteen complete primaries. Neither current run's frozen primaries exceeds 5.
At thirteen matched same-source checkpoints, SIL20 has three level-5 games
versus SIL4 twelve (including one level 6), on reused validation seeds.
Keep f58 as the verified depth reference and finish both live budgets before
the already-declared capacity comparison; its launch conditions remain unchanged.

The continuation's 31,252,480 primary improves to **six Level-5 games out of
20**, mean 169.2, but no level 6. Predeclare one supplemental consistency audit
on the reused 50-game secondary set, explicitly outside the deepest-primary-
rank trigger rather than silently promoting this candidate. It is now running;
require all games complete and more than the old reference's four secondary
Level-5 reaches before changing the consistency reference. Preserve f58's depth
role, report mean/depth tradeoffs and verify any new depth with a full replay.
Both learners stay active; no final-test games or evaluation observations are
used for training. See `results/level10/sil-continuation-31m25-audit-selection.json`.

That audit completes **50/50**, mean 182.66, ten Level-5 games, four Level-6,
two Level-7 and one **Level-8** game. The uncapped 555-point replay reproduces
seed10128, and every one of22,439 seeded neural actions matches. Promote
31,252,480 as the new depth/later-level-consistency/combined-mean reference;
70 reused games average178.81, with sixteen Level-5 and one Level-8 reach.
Preserve earlier references; this is not reliable Level8 or fresh-test evidence.

The old f58-source capacity command is superseded, not launched. The new
`sil-level8-continuation-protocol.json` prioritizes an unchanged five-million-
action continuation from the verified Level8 source after SIL20 finishes and
is reviewed, keeping the other existing learner on its declared budget and
checking memory before using the freed slot. If pressure is high, wait for
the other old run too. No third learner and no validation/replay state imports.

The revised `sil-level8-cap128k-protocol.json` is conditional on the unchanged
Level8 continuation's completed review: same source, only capacity changed,
sole learner initially after memory/paging checks. If a stronger source emerges,
revise the source/comparator before launch. At that declaration neither new
protocol was running; the subsequent launch is recorded below.
Compare future candidates against the new reference's PRIMARY rank (highest5,
six reaches), not its deeper secondary maximum, before secondary audits.
The full objective remains verified Level10 followed by100 fresh complete
games on untouched seeds40000-40099, an uncapped replay and honest documentation.

Both old tranches now exit 0: the Level-6-source continuation at 33,251,328
and SIL20 at 30,007,296. Completion audits and byte-identical archived logs
are saved. SIL20 has 19 eligible primary suites and one disqualified final
suite (one non-waiting incomplete game); do not adopt its higher update dose.
Its three guarded training-practice suffixes are correctly discarded.

The unchanged Level-8 continuation launched as the **sole learner**, session
96673, after source/log hash checks and a post-exit memory review. Its target
is 36,252,480, rounded to 36,253,696 actions. All learning settings remain
unchanged; model/optimizer and learner/SIL RNGs resume, while curriculum,
replay and pending suffixes start fresh. Launch and early accounting checks
pass. The 128k-capacity trial remains conditional and unlaunched.

The second primary, 31,752,192, finishes 20/20 at 175.1 / 139.5 / 356,
highest level 6, verified counts 2-6 of **17/10/7/2/1**.
It passes the reference-primary screening gate, not a promotion. Predeclare
and launch one 50-game secondary audit on reused seeds 10100-10149; retain
the verified Level-8 reference pending that result. Final seeds stay unused.

That secondary suite is disqualified: 49 complete games and one waiting
serve guard (seed 10135, score 22 / level 1 at 100,000 actions). Preserve its
audit; do not promote it. The third primary, 32,002,048, finishes 20/20 at
139.8 / 121.5 / 439, highest 7, and qualifies for the next secondary review.
Its predeclared audit launches only after the previous evaluator exits,
session 93838. Training remains session 96673 alone; reference remains Level 8.

The third checkpoint's secondary audit completes 50/50, mean 142.56, highest
5 and three Level-5 reaches. Combined reused sets average 141.77, highest 7,
four Level-5 reaches: not a reference improvement. Preserve both audits and
continue the unchanged learner; no further evaluation helper is active.

The fifth primary, 32,501,760, qualifies with one Level-6 reach, but its
secondary audit is disqualified by one serve-stalled game (49 complete /
one incomplete). The sixth primary completes 20/20 at mean 123.1, highest 4,
and does not qualify. No reference change; continue the declared unchanged
tranche, then review the conditional capacity experiment. All three standalone
secondary evaluators are now terminal; session 96673 remains the sole learner.

The Level-8-source midpoint audit passes through 33,812,390: 270 complete
full games / 612 complete practice segments, no training guards, all 1,710
learning suffixes / 882 episode coverages, and ten complete primary suites.
Later-level primary performance has weakened (first five checkpoints nine
Level-5 games, next five one, on reused seeds). Keep the verified Level-8
source; finish the declared tranche before the conditional capacity review.
Memory/paging remain under observation. No code or configuration changes;
no fresh final-test games. The observer stopped for review, not the learner.

The eleventh primary (34,000,896) is disqualified by one waiting serve guard:
seed 10006, score 265 / level 4 at 100,000 actions, no GAME OVER. Its 19
completed games do not form a full-suite estimate. Raw result and guard audit
are saved; do not send this checkpoint to secondary evaluation. The learner
resumes updates normally, reference remains Level 8 and final seeds are unused.

The thirteenth primary (34,500,608) qualifies with a Level-6 reach, but its
secondary suite is disqualified: 48 complete games / two serve guards.
The fourteenth (34,750,464) also qualifies; its predeclared secondary audit
runs as session 74298 after the previous evaluator exits. No reference change.
The first training guard, from-boot worker 30 at 34,640,319, is audited:
discarded terminal suffix of 2,048 transitions, no performance credit, exact
reward/source/action accounting. Continue the learner; do not restart it.

The fourteenth checkpoint's secondary audit is disqualified by a waiting
Level-4 guard on seed 10117 (49 complete / one incomplete). Its audit is
saved, and all five standalone evaluators are terminal. The fifteenth primary
completes 20/20 at mean 133.95, highest 5, two Level-5 reaches: no secondary
review. Finish the unchanged learner's remaining budget; keep the Level-8
reference and conditional larger-buffer comparison, with final seeds unused.

The sixteenth primary (35,250,176) qualifies with one Level-6 reach, but its
secondary audit is disqualified by seed 10109 waiting at level 4 (49 complete /
one incomplete). The seventeenth primary completes all 20 at mean 143.65,
highest 5, two Level-5 reaches: no secondary review. All six standalone
audits are terminal, the Level-8 source stays selected, and the learner is
finishing its declared five-million-action tranche before capacity review.

The unchanged Level-8 continuation now exits 0 at 36,253,696. Completion
audit passes: 447 complete full games / 1,253 practice segments, two full-game
serve guards with 4,096 retained transitions correctly discarded, all 3,205
learning suffixes / 1,702 episode coverages, and 19 eligible primary suites
plus one disqualified suite. No secondary audit replaces the Level-8 source.
The final primary is complete at mean 132.65, highest 4. Preserve the final
learner and hash-verified complete log; do not promote final weights.

The predeclared 128k-capacity comparison has launched as the **sole learner**,
session 60642 / PID 69997, after source/comparator/hash and post-exit paging
checks. Same e0da source and five-million-action budget, only capacity changed
from 32,768 to 131,072. Startup verifies 32 CPU-only workers / 16 protected
workers, fresh buffers, exact first-update match against control, and first
63 suffixes / 12 complete full games. Monitor matched complete validations
and actual memory as the larger buffer fills. Final seeds remain untouched;
the verified Level-8 reference is preserved. No other learner is active.

The larger-buffer first primary completes 20/20 at 159.25 / 120 / 318,
highest 6, three Level-5 reaches, ahead of the matched control's first primary
in mean/depth/counts but not best score. The full 131k buffer and first 139
suffixes pass accounting. Its secondary audit exits 0, all 50 complete, mean
145.78, highest 7. Combined 70-game mean 149.63 / seven Level-5 reaches still
trail the retained Level-8 source's 178.81 / sixteen reaches. No promotion;
continue the declared capacity trial and memory checks. Session 53712 is
terminal; learner 60642 remains live.

The second primary is disqualified by one Level-3 serve stall, seed 10005;
exclude the whole suite from selection and do not launch a secondary audit.
The first restored-training audit verifies 66 full games / 25 practice
segments, no training guards, and all 324 suffixes / 91 episode coverages.
Own restored practice reaches 7, not a frozen/full-game milestone. Keep the
learner running unchanged and preserve the verified Level-8 reference.

Own larger-buffer practice now reaches 8; source/reward/suffix audit passes
through 32,034,920 with 85 full games / 74 practice segments, zero guards,
452 suffixes / 159 episode coverages. This is not frozen performance.
The third primary qualifies with two Level-6 reaches; its secondary audit
exits 0, all 50 complete, mean 140.4, highest 6. Combined 70-game mean147.87 /
eight Level-5 reaches improves matched-control counts but not depth, and
still trails the Level-8 reference. No promotion. Both secondary evaluators
are terminal; continue the declared capacity trial with final seeds unused.

Protected worker 6 now completes a verified from-boot training Level-7 game,
507 points / 17,903 actions, at 32,471,271. The origin/suffix audit passes
through 32,587,776: 126 full games / 163 practice segments, no training guards,
747 suffixes / 289 episode coverages. Weights changed during the game, so
this is not a frozen-policy milestone. The fourth/fifth complete primaries
reach only 4/5 and do not qualify for secondary audits. Continue the same
capacity trial; monitor memory and keep the Level-8 reference unchanged.

The seventh larger-buffer primary qualifies with one Level-6 reach, but its
secondary audit, though 50/50 complete, reaches only 4 (mean 112.84). No
promotion. All three standalone capacity evaluators are terminal. A verified
comparison chart now shows complete from-boot training and eligible primary
validations through the seventh capacity checkpoint; it excludes restored
practice and separate secondary results. Continue the fixed trial, preserving
the Level-8 reference and untouched final seeds.

The eighth capacity primary is disqualified: seed 10017 stalls waiting at
level 2, score 58, while 19 games complete. Raw result and guard audit are
saved; no secondary evaluation or reference promotion. Keep the larger-buffer
learner active and treat the comparison chart as its dated seven-primary
snapshot, not a live or fresh-test report.

The capacity midpoint audit through 33,958,366 passes all 1,689 suffixes /
733 episode coverages: 238 complete full games, 495 complete practice
segments, no training guards. Eight of ten primaries are eligible; their
highest level is 6. All 101 completed practice segments starting on Level 8
end on Level 8 (mean new score 34.05, best 62). This identifies a bottleneck,
not its cause. Finish the declared capacity budget unchanged.

Prepare an isolated, evaluation-only temporal-history probe using the retained
e0da weights: compare 50,000-T-state actions with adjacent versus two-action-
spaced input frames, on the same reused primary seeds. The latter preserves
the original nominal 300,000-T-state history span. No learner, emulator,
curriculum, recorder or selected checkpoint changes; no final-test seeds.
Predeclare the two bounded arms, test bookkeeping/RNG equivalence, then run
them sequentially. Any result remains diagnostic until end-to-end support
and standard validation exist; no timing/history sweep or automatic promotion.

The isolated probe is implemented in `rl/temporal_probe.py`, with six new
tests (89 total passing). Its saved `temporal-e0da-protocol.json` precedes all
frozen diagnostic games. Session 59372 runs the two-game, 100k/stride-1
compatibility gate; require exact archived game dictionaries before starting
the sequential 20-game 50k arms. Only this new module/tests were added; no
live learner or shared emulator/evaluation implementation was modified.

Capacity accounting through 34,925,064 passes 2,269 suffixes / 1,014 episode
coverages. The first training guard is restored worker 21 at 34,842,646:
100,000 actions, score 353 / Level 6, waiting; its 2,048-step zero-reward
suffix is discarded. Primaries 11-14 all complete (levels 4, 4, 4, 5) and
none qualifies against the retained source. Keep the fixed trial unchanged.

The temporal compatibility gate exits 0 and exactly matches both archived
game dictionaries. Both predeclared 20-game arms also exit 0, all games
complete. Adjacent 50k: mean 144.7, median 119.5, best 255, highest 4, no 5.
Spaced 50k: mean 269.4, median 276, best 543, highest 8, ten Level-5 reaches.
The joint audit verifies all source/implementation hashes, seeds, boot offsets
and recomputed statistics. Spaced improves score on 17/20 games against the
original 100k configuration and 18/20 against adjacent 50k; these are reused
validation comparisons, not fresh-test estimates or proof of causality.

All temporal evaluators are terminal (59372 / 66316 / 68803, exit 0). Preserve
the original-timing Level-8 reference. After the fixed cap128k learner exits,
add end-to-end observation-stride metadata/support with default 1 unchanged:
environment, vector workers, snapshots/curriculum, training, evaluation,
recording and replay-policy inspection. Require exact original-game
compatibility and exact reproduction of the 20-game spaced diagnostic before
secondary validation. Verify its uncapped full Level-8 replay and every action.
Then evaluate this single fixed configuration on reused secondary seeds
10100-10149 with the matched 200k-action guard. No rate/history sweep, fresh
final seeds, automatic promotion or new training run before those checks.

The detailed `temporal-e0da-integration-protocol.json` is now saved before any
shared-core edit. It fixes default compatibility, ordinary-evaluator exact
20-game reproduction, uncapped seed-10015 replay/all-action verification, and
the 50-game secondary audit. A new evaluation-only package may copy unchanged
model bytes with explicit timing/stride provenance, never mutate the source.
Only fully verified 70-game depth/count rank improvement can replace the
original-timing validation reference; no new learner or final seeds yet.

Capacity trial session 60642 exits 0 at 36,253,696 (+5,001,216 actions).
Completion audit passes: 421 complete full games / 970 complete practice
segments, one guarded practice segment, all 3,043 suffixes / 1,392 episode
coverages, 18 eligible primaries / two disqualified suites. Eight eligible
primaries reach 5 (15 games total), highest primary 6; no secondary replaces
the Level-8 source. Final primary completes 20/20 at mean 145.7, highest 5;
its long-waiting seed 10014 eventually finishes, not a guard. The complete
5,091-line log is copied and hash verified. All learner/evaluator processes
are absent; shared-core integration may now begin under the saved protocol.

Stride integration is applied after the clean exit. Seven new tests pass,
and all 96 regression tests pass (7.172 seconds, session 5853 exit 0).
The ordinary evaluator is running the fixed 20-game original-timing
compatibility gate. Do not start spaced reproduction until its complete
game dictionaries exactly match the archived source primary.

The original compatibility gate exits 0 and matches all 20 archived game
dictionaries exactly. A new `runs/level10-temporal-e0da-eval` package copies
unchanged e0da weights with explicit 50k/stride-2 metadata, no optimizer and
resume disabled. Its state/model and implementation hashes are audited.
Session 14242 now runs the ordinary-evaluator spaced reproduction using
checkpoint defaults, not the TemporalPolicy wrapper. Require exact equality
with the original 20-game diagnostic before the uncapped replay gate.

Session 14242 exits 0 and exactly reproduces all 20 spaced diagnostic game
dictionaries through the ordinary evaluator (mean 269.4, ten Level-5 reaches,
highest 8). The saved reproduction audit opens the uncapped replay gate.
Session 85211 records seed 10015 using the adapted checkpoint defaults and
no action limit. No learner or secondary evaluator is active yet.

Replay session 85211 exits 0 with exact 543/8/44,325-action outcome;
inspection 92885 exits 0 with zero action mismatches at batch 128. Full
replay/inspection hashes and decoded level entries are saved. All integration
gates pass. Session 76080 now runs the one predeclared 50-game secondary audit
on reused seeds 10100-10149, checkpoint defaults 50k/stride 2, guard 200k,
20 workers. No learner or fresh final-test evaluation is active. Await every
game and audit the full 70-game rank before reference promotion.

Secondary session 76080 exits 0, all 50 complete, mean 218.98, highest 7.
The full 70-game audit gives mean 233.39 / median 244.5 / best 543 / highest 8,
with counts 2-8 of 70/54/44/21/6/3/1. This improves the predeclared rank;
promote the fixed50k/stride2 evaluation-only package, preserving original
source/timing and committed milestones. No Level9/10 or final tests yet.

The next training stage is predeclared and launched as sole session 48013:
`runs/level10-timing50k-stride2-sil4`, original e0da model/Adam/RNG resume,
4M additional50k actions, stride2, sqrt gamma/lambda, doubled rollout/batches,
guards/eval interval and SIL storage windows. Other learner/curriculum/loss
settings stay fixed. No imported replay/validation trajectories, archives
and buffers start empty. Startup audit verifies parameters/source hashes,
32CPU-only workers/16protected roles and first14 suffixes. The selected
evaluation package remains frozen; monitor first complete primary and memory.

The first fine-tuned primary at 31,506,432 is 20/20 complete: mean 224.8,
median 225.5, best 579, highest 8; counts 2-8 are 18/12/10/7/2/2/1.
Seed 10006 reaches 8 in 48,490 actions; seed 10001 reaches 7. This rank trails
the frozen timing reference's primary, so no secondary audit or promotion.
Checkpoint hashes, saved RNG/selection records and training accounting
through 31,577,262 pass (48 suffixes / six complete full games, no guards).
Session 48013 remains live and alone; paging is stable. Continue its declared
budget with the reference and final-test seeds protected.

First same-run archive audit through 31,956,992 passes 106 suffixes / 20
completed full-game coverages, no guards. Worker 19 publishes its own Level-5
entry at 31,877,428 and Level-6 entry at 31,938,420; protected worker 6 also
publishes Level 5. All entries originate in fresh from-boot training games,
not imported validation states. No completed restored segment is claimed yet.
Keep training unchanged and review the imminent second primary.

Second primary at 32,006,144 qualifies: all 20 games complete, mean 279.3,
median 271, best 558, highest 8, counts 2-8 of 19/17/16/11/4/3/2. The frozen
model/state/optimizer and primary hashes are verified. Source/reward audit
through 32,086,590 passes 129 suffixes / 28 episode coverages: 26 full games,
two completed restored segments, no guards. Restored practice reaches 6.

Session 24318 runs its predeclared 50-game secondary audit on reused seeds
10100-10149 at 50k/stride2 with a 200k-action guard. Compare all 70 complete
games against the frozen timing reference; any incomplete suite disqualifies
the candidate. If it qualifies, record the highest-level/highest-score game
uncapped and verify every action before promotion. Learner 48013 continues
unchanged; no final-test seeds are used.

The 32M secondary exits 0: all 50 games complete, mean 245.06, highest 8.
Combined 70-game mean/median/best are 254.84/261/572; counts 2-8 are
66/57/45/26/11/8/3. Its declared rank improves, while Level-1 clears decline
from 70 to 66. Recorder 61864 and inspector 85729 both exit 0: seed 10126
exactly reproduces 572 / Level 8 in 55,422 actions, all actions verified.
Promote immutable step-032006144 and preserve preceding references.

The 32.5M and 33M primaries both complete, mean 181.7 / highest 6 and
203.25 / highest 7; neither qualifies against the new reference. Accounting
through 33,118,302 verifies 366 suffixes / 129 episode coverages: 66 full
games and 63 completed restored segments, no guards. Changing-policy full
training and restored practice both reach 8, not a new frozen milestone.
Learner 48013 continues unchanged toward its rounded 35,258,368 stage end.

Fifth/sixth primaries (33,505,280 / 34,004,992) both complete 20/20, with
means 197.45 / 188.7 and highest levels 8 / 7. Neither qualifies for a
secondary audit. Through 34,083,965, all607 suffixes /233 episode coverages
pass source/reward/action checks:98 full games,135 restored,no guards.

One predeclared inference-only diagnostic of the frozen32M weights tests
25k/stride4 with the same nominal screen-history span and doubled guard.
Session74974 exits0:20complete,mean217.95,median225,best559,highest8,
counts2-8 18/11/10/7/4/1/1. This ranks below selected50k/stride2; retain
reference and do not run a secondary audit or further timing sweep.
Solelearner48013 continues its unchanged stage; no final-test seeds used.

Seventh primary34,504,704 is20/20complete,mean200.35/highest6,belowreference.
An isolated one-setting sampling-temperature probe is predeclared separately:
temperature0.8 on frozen32M logits, unchanged50k/stride2, reused20primary.
All101tests pass. Unit-temperature session52533 exits0 and exactly matches
baseline seeds10000/10001 complete-game dictionaries. Session91516 now runs
the fixed0.8 probe. No heuristic actions, live learner edits, promotion from
partial results, sampling sweep, or fresh-test seeds. Learner48013 continues.

Temperature0.8 session91516 exits0:all20complete,mean220.4,median264,best577,
highest8,counts2-8 17/14/11/6/2/1/1. The final game waits atLevel8 but serves
and finishes without intervention. Complete audit verifies source/timing/
seeds/boot offsets/statistics. Rank is below selectedunit-temperature32M;
nointegration/secondary/promotion/sweep. Mainlearner48013 reaches its final
scheduled35,004,416 validation; await complete results and normalstageend.

Initial timing learner48013 exits0 at35,258,368 (+4,005,888),wall3367.31s.
All8primariescomplete,7reach5,total37Level5reaches,highest8. Finalscheduled
primarymean157.5/median127.5/best274/highest4 doesnotreplace32M. Completion
audit passes139full/237restored,922suffixes/376coverages,zero guards. Complete
1,739-line log is copied byte-identically and listed with the finished runs.

After oldprocessesexit and source/loghashes pass, solelearner86865 launches
`runs/level10-timing50k-stride2-lr5e6-long`: selected32M source, half learning
rate5e-6,8Madditional actions,roundedend40,009,728. Other settings remainfixed;
buffers/archives/historyrestartempty. This is not a matched isolatedablation
against the uninterrupted oldrun. Predeclared review: target/budget/fault,
or fiveconsecutivecompleteprimarieswithzero5. All101tests pass; startupchecks
exactparams/sourcehashes,32CPU-only/16protectedworkers andfirst15suffixes/
onefullgame through32,109,877. Selected32M reference staysfrozen and final
seeds40000–40099remainunused. Monitor86865, not terminal48013.

Firstlower-rate primary32,505,856 completes20/20:272.4/266/544/highest8,
counts2-8 18/16/13/8/6/4/1. Fullcheckpoint/config/boot/RNG/selectionrecords
verify. Its depth/count rank is below selected32M (one8versustwo), so no
secondaryaudit orpromotion despite higher6/7counts. Accountingthrough
32,576,507 passes64suffixes/ninefullgames,zero guards, no archives/restored
segmentsyet. Keep86865 running towarddeclaredbudget/target with same settings.

Firstsame-run archiveaudit through32,853,916 passes110suffixes/25complete
fullgames,sixpublications,zero guards; no completedrestoredsegmentyet.
FirstLevel5entries areworker30at32,756,383,protectedworker10at32,768,299,
andworker29at32,795,934, allactualfrom-boottrainingexperiences. They open
same-run practice, notfrozen milestones. Continue86865; nextprimary33M.

The 33,005,568 lower-rate primary completes all 20 games at mean 238.75,
median 218.5, best 557, highest 8; counts 2-8 are 18/14/10/6/5/3/1. Its
rank trails the selected reference and first lower-rate checkpoint. No
secondary audit or promotion. Full hashes/configuration/boot/RNG checks pass;
accounting through 33,093,709 verifies 160 suffixes / 39 episode coverages
(34 full games, five restored), no guards. Keep learner 86865 unchanged.

Third primary 33,505,280 is wholly ineligible: seed 10006 truncates at the
200k guard, score 425 / Level 7, not waiting, no GAME OVER. Nineteen games
finish (one Level 8), but no completer-only average qualifies the suite.
Checkpoint/boot/config/RNG/selection records verify. No secondary/promotion.
Accounting through 33,606,215 passes 254 suffixes / 69 covers: 46 full games,
23 restored segments, no training guards. Restored practice reaches 8, full
training 7. Learner 86865 resumes normally; keep the declared stage unchanged.

First complete from-boot training Level 8 is verified: worker 22 at33,635,511,
559 points /50,918 actions, all four learning suffixes cover the game exactly.
First restored Level 8 at33,504,765 earns188 new points from a same-run Level7
entry. Through33,895,812, all332 suffixes/105 covers pass (59 full,46 restored),
no training guards. These changing-policy games are not frozen milestones.
Keep86865 live; next scheduled primary is34,004,992.

Fourth primary34,004,992 completes20/20:262.8/272/498/highest7, counts2-7
20/17/13/11/4/2. Long seed10005 finishes at421/Level7 in146,223 actions,
without further scoring or intervention; no guard. Rank belowreference,
so nosecondary/promotion. All hashes/configuration/boot/RNG/selectionchecks
pass. Through34,125,784,376 suffixes/124 covers verify (62full/62restored),
zero trainingguards. Keep86865 running under unchangedstage protocol.

User resumed with “Continue”; a fresh handle poll confirms86865 finished
normally at40,009,728 (+8,003,584), not a live/stalled process. Full completion
audit and byte-identical log archive pass. Fourteen eligible primaries, two
disqualified, highest8, no checkpoint qualifies against selected32M. Complete
training totals239full/439restored, plus one guarded full game;1,778 suffixes/
679 covers reconcile and guarded4,096 suffix is discarded. Of98 Level8-start
practice segments, none clears8 (mean41.70,best62). Do not resume finalweights.

The matched `runs/level10-timing50k-stride2-min8` trial changes only minimum
practice level5→8 relative to that control; same32M source,5e-6 rate,8M budget,
32 workers/16 protected, timing/sampling/SIL settings. All103tests pass, and
prior exit/log/source gates pass. Solelearner74142 (PID14930) is live with
exact matching config except run path and threshold; initial15 suffixes/
one full game through32,109,877 verify. Archives remainempty until actual
same-run Level8 discovery. Check expected first-checkpoint common prefix;
keep selected reference and final40000–40099 untouched. Monitor74142, not86865.

Focused trial's first checkpoint32,505,856 passes the predicted common-prefix
test: model/optimizer byte-identical to control; all nonconfig saved state
matches; full evaluation dictionary identical (20complete,272.4/266/544/8).
No improvement or promotion is claimed. Through32,541,955,61suffixes/eight
full games pass accounting, with no archives/restored segments/guards. Keep
74142 live as the minimum-level threshold starts affecting practice later.

Focused primary 33,005,568 completes all 20 games: 213.4 / 221 / 530 / Level 8,
counts 2-8 17/13/10/6/3/2/1. Its rank trails selected32M; no secondary or
promotion. Hashes/config/boot/reward/RNG/selection checks pass. Through
33,109,893, all 153 suffixes / 37 full games reconcile, no archives/restored
segments/guards. Complete training reaches 6; validation Level8 is not a
training archive. Continue sole learner74142 with unchanged protocol;
next primary33,505,280. Final-test seeds remain unused.

Fresh checks after “Great, continue” confirm focused learner74142 exited0
at40,009,728 (+8,003,584), not live. Full audit/byte-identical log archive pass:
16 complete primaries, no qualifying checkpoint;273 full/548 restored games,
1,500 learning suffixes/821 episode covers, zero guards. Three genuine Level8
entries;548 Level8 starts earn mean32.88/best62 new points, zero clears, no
first-to-last100 improvement. Preserve all weights; do not resume terminal.

Predeclared next trial `level10-timing50k-stride2-min8-refkl01` changes only
PPO's objective by adding0.1 KL(own frozen32M source || learner) on current
learner rollout screens. Same source/budget/timing/curriculum/SIL settings;
no teacher actions or demonstrations. This tests retention, not an established
cause of the plateau. Protocol precedes implementation. Disabled before/after
65,536-action smokes match model/optimizer bytes, nonconfig saved state and
all learning/episode/validation events exactly. Five new unit tests pass;
enabled smoke and full suite are required before the performance launch.

All reference-penalty gates passed:108 tests, disabled byte-identical smoke,
enabled finite-loss/immutable-reference smoke. Logs are archived separately
as diagnostics; none selects or seeds a learner. The sole performance learner
is now20492, PID32532, `runs/level10-timing50k-stride2-min8-refkl01`, launched
from original32M source after all diagnostics exited. Full configuration
matches focused control except reference/run fields. Through32,110,651,
ten learning suffixes reconcile, zero full-game completions/archives/guards.
Continue this declared8M-action trial; first primary32,505,856. Select only
complete qualifying primaries, then audited secondary/replay; final seeds
remain unused. Do not poll terminal74142 or any smoke as live training.

September16 user asks for actual status and continuous oversight. Fresh poll:
20492 completed its8M-action budget nearly46h earlier. Goal is usageLimited;
agent cannot resume that product-controlled status. Completed reference-KL
audit/log preserved;16 complete primaries,112 reaches5, no9/10.35.5M primary
qualifies;50-game secondary and complete49,715-action replay/inspection pass.
Select d4217900… by declared70-game depth rank (four8 versus three), while
reporting lower mean218.59 versus254.84 and weaker earlier-level counts.

Implemented local `rl.supervise`: one unchanged continuous PPO learner,
15-second heartbeat, automatic primary-record→50-secondary→uncapped replay
and action verification, frozen100-fresh tests only after verified10. No
Codex/API calls or usage-limit bypass; explicit operator/health/regression
stops, no silent fixed-action batch exit. First launch19639 failed before any
training due to mutually exclusive budget flags; preserved needs_attention
record, corrected argument builder and added real CLI regression.119 tests pass.

Current live supervisor session98423 /PID3280 owns learnerPID3283 at
`runs/level10-supervised-refkl01-v2/learner`. Source is selected35.5M;32M frozen
reference remains unchanged. Both action limits0, learning config identical.
Through35,694,614,41 suffixes/five full games reconcile, no guards. Read live
`status.json` AND process handle, and `selected.json` for automatic promotions;
do not trust old stored selected-reference metadata after unattended progress.
First primary36,003,840. Stop via that supervisor root's STOP file if requested.
No need to restart an arbitrary8M tranche. Automatic chat follow-up still
requires the product goal to be resumed; current local work does not change it.

First real supervised cycle verified September16 at23:46 PDT: primary36,003,840
completes20/20, mean192.35 /median193.5 /best524 /Level8, counts2-8
17/12/6/5/2/1/1. It does not beat the selected primary, so no secondary or
promotion. Supervisor consumes it and the same learner resumes, reaching
36,477,239;162 suffixes/37 full games reconcile, zero guards. Both live PIDs
and unchanged selected/teacher hashes verified;119 tests and diff checks pass.
See supervised-continuation-first-cycle-verification.json. Next primary is
36,503,552; keep monitoring live status/selected records, not this snapshot.

September19 review: supervisor98423 had stayed live for48.8h through236M,
but no Level8 clear. It did automatically promote106.5M then116M; selected116M
scores315.47/280/581 across70 complete reused games, ten reaches8. Graceful
STOP at236,003,328 preserves final learner and all logs;400 finished primaries,
363 complete,11,644 complete restored segments, no9/10. All33,173 SIL suffixes
reconcile. Do not poll98423 as live: it exited1 on intentional interruption.

Preserve selected581-point and single-effort598-point replays in versioned
`results/level10/best` and `best-effort` bundles. Single-game score records do
not replace validation-selected models. Automatic supervision now exports both
and never deletes previous bundles. Original public/Level5 artifacts unchanged.

Predeclared progress16 experiment changes only curriculum snapshot opportunities:
capture live own within-level progress every16 new points, same8-entry/level
reservoir and16 protected boot workers. No external/validation states.129 tests
pass; disabled65,536-action before/after model/optimizer/RNG/events identical.
Live supervisor99933/PID7587 owns learner7593 under
`runs/level10-supervised-progress16`. Source is selected116M, reference32M
unchanged; action budgets0. Configuration/worker/source identities verified.
First primary116,506,624. Verify real progress publications and full first
evaluation/resume cycle before relying on the ongoing experiment. Read live
status/selected/current.json records, not these dated numbers. No fresh tests.

First progress16 primary116,506,624 finishes20/20 at272.65/282.5/524/8,
one reach8, below selected reference. No promotion; supervisor observes it
and training resumes. Through116,629,504 all59 suffixes/11 full games pass
accounting, no guards. Training has not yet rediscovered Level8 at this
snapshot, so no progress archive is live-exercised yet. Continue monitoring
actual same-run publications; do not seed archives from the replay/validation.

User explicitly asks to keep monitoring until Level10. Stay active; do not end
with only a process handoff. Product goal still reports usageLimited, but this
turn is actively monitoring the local learner. First real progress publications
verified at117,291,131: two8-entry/four16-point-progress states, including a
peer-restored chain with exact new-score accounting.159 suffixes/35 full games
reconcile, zero guards. No complete Level8 practice yet; no9/10. Continue
watching practice completions and full frozen validations, adapt only on
evidence, retain stable best/best-effort bundles. Live session99933/PIDs7587/7593.

While actively monitoring, set a first structured review near120,006,912
(four million new actions, not a stop budget). Consider all completed primary
suites and the distribution/clear outcomes of progress-start practice. At
117.93M only61 practice attempts have completed, no clears, so do not infer a
plateau yet. Continue or change a single training factor based on that review;
do not repeat the previous120M-action unattended plateau. Keep monitoring
through the Level10 validation/replay/fresh-test gates, not merely startup.

September 19, 08:35 UTC: confirmed a hard blocker, not another learning plateau.
The author's description and a read-only audit of this exact CMD binary agree:
the game ends after eight levels. At 0x5870, CP 9 / JP Z,0x5d17 enters GAME OVER
before the level-HUD call. Level 10 cannot be reached without changing the game.
The earlier assumption that more training could reach 9/10 was incorrect.

Gracefully paused supervisor99933 at120,102,912 (+4,096,000). It exited1 on the
intentional STOP; learner and all32 workers exited, latest checkpoint saved.
Eight primaries complete20/20, none promotes;114 full/432 restored complete
episodes,877 suffixes/546 episode coverages all reconcile, zero guards.
Log archived byte-identically; selected116M and both stable replay bundles
remain hash-verified.130 tests pass. No progress4 run was launched.

IMPORTANT measurement correction: previous zero Level8-clear counts used a
transition to displayed9. The final-level win path skips that display update,
so those counts cannot establish zero wins. No all-eight-level victory has
been verified. Do not claim success based on reaching8 or reading internal9.

Asked user to choose beating all eight original levels (recommended) or
explicitly extending the game with levels9–10. Do not silently replace the
goal or modify the game; do not continue an unreachable Level10 monitor.
If all-eight is chosen, first define/test a screen-derived win detector and
correct the evaluation/supervisor gate, then resume evidence-led training
with the same learned-policy rules and immutable best-replay preservation.
Fresh final-test seeds40000–40099 remain unused. GOALS.md is unchanged.
Evidence: results/level10/game-level-cap-audit.json and
results/level10/progress16-completion-review.json. All prior live-status notes
are historical; no performance learner is currently running.

User explicitly chose beating all eight original levels. Goal scope updated
in GOALS.md; no game modification. All-eight protocol is predeclared in
results/level10/all-eight-protocol.json. Implemented conservative screen-only
victory proof: terminal Level8 with a visible reserve. Last-ball endings remain
unverified, not presumed losses. Static binary-contract fixtures verify both
loss paths, reserve renderer and final-level branch. Outcomes never enter
policy inputs/reward or change existing learning terminals.

141 tests pass. A65,536-action game-win-mode smoke matches historical disabled
model/optimizer bytes exactly, with unchanged RNG and all other saved state
except declared config/selection rank. Smoke weights/states are not training
sources. Selected116M is being rerun on original20+50 reused seeds for explicit
outcome-labelled baselines; no fresh seeds. If no verified baseline victory,
launch the sole supervisor under runs/all-eight-supervised-progress4, selected
116M source, interval4 and otherwise unchanged learning. Win-first selection,
replay-outcome recomputation, immutable replay bundles, and100-fresh gate are
implemented. Monitor actual learner/validation/practice wins continuously;
first structured review after4M new actions, not an arbitrary stop budget.

September20 07:02UTC: all70 baseline games exactly reproduce prior per-game
fields; zero verified wins and ten unverified last-ball Level8 endings. Both
evaluation helpers and the smoke are terminal. Read-only supervisor preflight
passes and prior-process check is empty. Sole performance supervisor37611
(PID33187) owns learner33193 under runs/all-eight-supervised-progress4.
Action limits0, selected116M source, interval4, target-game-win true and
stop-on-target false; all other learning settings and reference hash unchanged.
First116,041,046 actions account correctly,32 CPU-only workers/16protected,
two SIL suffixes, no full/practice completions yet. First primary116,506,624.
Keep actively monitoring; do not report the old paused run as live or end after
startup. Model bundles and original public/Level5 artifacts remain intact.

07:10UTC: declared reassessment of three strong historical policies discovers
a win in122,503,168 on primary seed10019:576points, one spareball visible.
This was hidden by the former displayed-level-only metric. Pause current
supervisor via STOP;37611 exits1 intentionally at116,506,624, latest preserved.
First validation cancelled, not counted complete. No practice state yet and
no evidence for interval4 performance.48 suffixes/nine full-game covers reconcile.

Winning replay87140 exits0, exactly42,254 actions/42,255frames. Inspector69477
exits0 with zero seeded-action mismatches. Final screens visually checked;
terminal Level8 has a visible reserve. Verified winner bundle published to
results/level10/best-effort; prior598-point version remains intact. Do not
claim final selection or fresh testing yet. Parent reassessment38187 continues
the predeclared122.5M/226.5M/222.5M20+50 checks. Select by verified wins then
depth/count/mean across70 complete games, preserve an uncapped winning replay,
then freeze exact selected weights and run untouched seeds40000–40099 once.
Keep monitoring until that workflow and honest final reporting are complete.

07:20UTC:122.5M completes70/70, one verified win/seven reaches8/mean285.96.
226.5M completes70/70, one verified win/eleven reaches8/mean298.94. Its585-point
seed10001 replay (49,159 actions, one visible reserve) reproduces exactly and
inspector69319 exits0 with zero mismatches. Publish226.5M to both stable bundles;
retain prior598/581/576 versions.142 tests pass, including completion-only
supervision with no new learner. Third222.5M primary20/20 has zero verified wins;
secondary still running in parent38187. Finish declared comparison before any
fresh-test access. No learner live; no wall-clock stop or abandoned monitoring.

September20 07:31UTC: revised goal achieved and final test complete. Final
selection226,500,608 was frozen before fresh access;222.5M was disqualified by
one incomplete secondary game, with no retry or replacement seed. Both stable
replays and models/breakdown-all-eight preserve the verified585-point win;
49,159 actions, zero mismatches, one reserve at final-level GAME OVER.

Supervisor8121/PID40468 ran only the frozen test, never a learner. All100
seeds40000–40099 ended naturally under unchanged sampled policy and max_steps0.
Mean278.75, median277.5, best583, highest8; zero verified wins and three
unverified last-ball Level8 endings. The two apparent stalls eventually ended
at392,602 and447,196 actions. No timeout, truncation, seed replacement, forced
serve, retry, or diagnostic replay was needed. These seeds are now used.

Final phase target_verified_and_tested. Read-only process check finds the
supervisor, evaluator and all worker-family PIDs absent. Raw log records match
all100 result records exactly; frozen/source/package model, optimizer and
state hashes agree. Fresh result/log and stopped progress4 log copies are
byte-identical. Game/emulator unchanged.142 tests pass. No training, helper
or monitor remains active; winning capability is proved, reliable winning
is not. ALL_EIGHT.md and the completion audit contain the final handoff.
No commit, push or public replay deployment performed in this turn.
