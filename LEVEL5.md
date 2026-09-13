# Level-5 continuation experiment

## Final result

**Displayed level 5 is verified in a complete frozen-policy validation game**:
score **278**, seed **10016**, 15,346 model-selected actions. This means clearing
levels 1–4, not clearing level 5. The trainer stopped automatically at
**17,502,208 PPO actions**; no learner remains running.

| Suite | Complete | Mean | Median | Best | Highest level | Reached level 5 |
|---|---:|---:|---:|---:|---:|---:|
| Selection validation | 20/20 | 88.00 | 64 | 278 | 5 | 1/20 |
| Frozen final test, seeds 30000–30099 | 100/100 | 83.32 | 72.5 | 167 | 3 | 0/100 |
| Matched uniform random | 100/100 | 1.54 | 1 | 4 | 1 | 0/100 |

**This is a first-reach milestone, not reliable level-5 play.** The fresh test
had 50 level-2 games, one level-3 game, and no level-4 or level-5 games. The model
was selected and frozen before testing; these results did not trigger further
training or model selection. The original model and public replay are unchanged.

- [Frozen model, optimizer, metadata and checksums](models/breakdown-level5/README.md)
- [278-point level-5 validation replay](results/level5/replay.html)
- [100-game final test](results/level5/trained.json) and [random baseline](results/level5/random.json)
- [Training chart](results/level5/training-curve.png) and [complete experiment logs](results/level5/logs/README.md)

The sections below are the chronological protocol and experiment history;
statements about reserved test seeds describe their status at that point.

## Goal and protocol

Reach displayed **level 5**, meaning levels 1-4 have been cleared. This is a
first-reach milestone, not a claim of reliable mastery or clearing level 5.
The existing `models/breakdown` checkpoint and public replay remain unchanged.
The screen-only observations, six model-selected key combinations, score-only
reward, and existing emulator are unchanged. No demonstrations or gameplay
heuristics are introduced.

- Start: `models/breakdown`, 6,402,048 PPO actions plus 500,000 DQN actions.
- Initial experiment: 20,000,000 additional PPO actions, or the first complete
  validation suite containing a level-5 game. No wall-clock limit.
- Learning settings: unchanged (MLX, 32 environments, learning rate 0.00025,
  entropy coefficient 0.01, 128-step rollouts, gamma 0.995, episodic-life returns).
- Validation: 20 fixed seeds 10000-10019 every 500,000 global PPO actions,
  with no game action limit. These are for selection, not final testing.
- Training retains the existing 30,000-action episode truncation guard;
  truncated episodes bootstrap and cannot count as completed successes.
- Final test reserved in advance: 100 fresh seeds **30000-30099**, complete
  games with no action limit. Do not inspect these during model selection.
- Report mean/median/best score plus reach counts/rates for levels 2-5.
  Rates use complete games as their denominator; incomplete suites never
  qualify for checkpoint selection or successful completion.

## Commands

Baseline on the expanded validation suite:

```sh
venv/bin/python -m rl.evaluate models/breakdown/model.safetensors \
  --games 20 --seed 10000 --max-steps 0 \
  --output results/level5/baseline-validation.json
```

Initial continuation:

```sh
venv/bin/python -m rl.ppo --run runs/level5-baseline \
  --resume models/breakdown --additional-steps 20000000 \
  --target-level 5 --target-clears 1 \
  --eval-games 20 --eval-seed 10000 --eval-every 500000 --eval-max-steps 0
```

`--steps` is an absolute PPO action counter; `--additional-steps` is a budget
relative to the resumed checkpoint. Budget boundaries round up to a complete
rollout. They are mutually exclusive. Omitting both runs until the validation
target is met or the process is interrupted. SIGINT/SIGTERM saves `latest`
after the current rollout/evaluation.

Logs are in `runs/level5-baseline/metrics.jsonl`. Numbered checkpoints preserve
each validation candidate. `best` ranks complete validation suites by mean
score; `best-level` ranks counts reaching level 5, then 4, 3, 2, then mean
score. `latest` preserves resumable weights, optimizer, RNG and counters.
Changing the validation protocol resets best-score comparisons on resume.

Checkpoint bookkeeping was subsequently tightened so both best-mean and
best-level records are calculated before any best snapshot is saved, and a
numbered snapshot's `state.json` is refreshed after its validation. Older
numbered snapshots can retain the preceding selection records in state; their
adjacent `evaluation.json` remains the authoritative measurement. The then-running
extension predated this code change and was not restarted merely to pick it up.
All **21 tests** pass. A separate 4,096-action smoke run completed two diagnostic
games and verified matching post-validation records in numbered, `best`,
`best-level`, and `latest` snapshots. Its weights are not used by the main run.

Regenerate the score, complete-game level reach, and validation-completion chart:

```sh
venv/bin/python -m rl.report \
  results/level5/logs/level5-baseline.jsonl \
  results/level5/logs/level5-terminal-fix.jsonl \
  results/level5/logs/level5-lr1e4.jsonl \
  results/level5/logs/level5-lr1e4-extended.jsonl \
  results/level5/logs/level5-entropy003.jsonl \
  results/level5/logs/level5-entropy003-extended.jsonl \
  --target-level 5 --baseline results/level5/random.json \
  --output results/level5/training-curve
```

[Current continuation chart](results/level5/training-curve.png). The level panel
excludes truncated games, distinguishes training windows from frozen validation
points, and marks the level-5 target. Live plotting tolerates a partial final
log record but still rejects malformed interior records.

## Follow-up decisions

First measure what additional training alone achieves. If level reach rates
continue improving, extend the unchanged run toward 50 million additional
actions. If validation progress stalls, compare a lower learning rate against
the unchanged setting from the same checkpoint; separately test exploration
strength if needed. Keep experiments in distinct directories and compare
equal additional action budgets on the same validation suite. Inspect replays
before testing faster action frequency or a different network.

Do not claim completion based on a training episode, a truncated game, or an
assumed score threshold. Freeze a validation-selected checkpoint before the
reserved final test and report its measured success rate, including zero if
the milestone does not reproduce on that suite.

## Status

Training initially started in `runs/level5-baseline` from the original model, with an
absolute PPO counter target of 26,402,048 (20 million additional actions,
rounded up to a full rollout). `caffeinate -i` keeps the Mac from idle sleep
while the process runs. It stops at the validation target or this experiment's
action budget; reaching the budget is a decision point, not goal completion.

The original model's expanded validation baseline is **20/20 complete games,
mean 51.5, median 55.5, best 66, level 2 in 1/20, no levels 3-5**.
Full records: [baseline-validation.json](results/level5/baseline-validation.json).
These are validation results, not a replacement for the earlier held-out test.

All 14 tests passed, including target-level boundaries, exclusion of truncated
games, deeper-level checkpoint ranking, and validation-protocol compatibility.
A separate 4,096-action resume smoke test completed training, validation,
checkpoint saving, and clean shutdown; it did not become the long run's source.
The original replay's playback checks still pass. No level-5 result yet.

The first scheduled checkpoint at 6,500,352 PPO actions completed all 20
validation games: mean 48.2, median 49, best 109, level 2 in 1/20 and no higher
levels. Checkpoint/optimizer saves succeeded and training continued (a level-2
game no longer triggers the level-5 stop). The higher single-game score does
not yet represent a consistency improvement over the 51.5-mean baseline.

## Recovered terminal-detection failure

The initial continuation stopped at **9,502,720 PPO actions** during validation
after 3,100,672 additional actions. The latest weights and optimizer were saved.
Its best validation checkpoint (8,503,296 actions) had mean 65.3, best 97 and
8/20 level-2 games. Results fluctuated; no level 3 was observed.

The failure was reproduced at seed 10002, action 2065, score 55. The game printed
`GAME OVER` with a pre-existing full graphics cell (0xBF) in the word gap. Python
normalizes graphics to spaces for text detection, but the native stop expected
an empty graphics cell (0x80). Held SPACE therefore dismissed GAME OVER and
restarted the game inside an action; the blank title-screen redraw triggered
the score-settling guard. The guard correctly rejected the ambiguous reward.

The native text stop now uses the same normalization as the screen parser,
matching all letters exactly and treating non-ASCII-text cells as spaces only
where the target text has a space. Its literal-match API remains available.
The fixed seed ends normally at score 55, and the recovered checkpoint completes
20/20 validation games (mean 54.0, best 103, 3/20 level-2 games). See
[recovered-validation.json](results/level5/recovered-validation.json).
All 16 tests pass, including every graphics glyph in the word gap and rejection
of an ASCII letter in that gap. No gameplay controls or rewards were added.

Continuation is now in `runs/level5-terminal-fix`, with the same learning
settings, original action-budget endpoint, and environment version
`normalized-game-over-v2`. Historical logs and checkpoints remain intact.
The environment version participates in validation-protocol comparisons so
pre-fix and post-fix best scores are not silently treated as identical suites.

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/level5-terminal-fix \
  --resume runs/level5-baseline/latest --steps 26402048 \
  --target-level 5 --target-clears 1 --eval-games 20 --eval-seed 10000 \
  --eval-every 500000 --eval-max-steps 0
```

The user explicitly authorized autonomous continuation to level 5, including
repairing failures and making follow-up experiment decisions without waiting
for approval. The 20-million-action point is an assessment point, not a final
stop if the goal remains unmet.

The recovered run recorded its first **level-3 training game** at global action
9,698,140 (score 139). This is a changing training policy, not yet a verified
frozen-checkpoint milestone. The 10,002,432-action checkpoint validated at
mean 58.25, best 94 and 3/20 level-2 games, with no level 3.

A bounded argmax diagnostic on validation seeds 10100-10104 at that checkpoint
completed only 3/5 games; two stalled waiting for a serve at the 10,000-action
limit. It is not a complete-suite performance claim, and sampled-policy
training/evaluation remains unchanged. Records:
[argmax-10m-diagnostic.json](results/level5/argmax-10m-diagnostic.json).

Read current progress without touching the GPU or learner:

```sh
venv/bin/python -m rl.status runs/level5-terminal-fix
```

Further unchanged-training validation improved at 11,501,568 actions to mean
69.85 with 8/20 level-2 games, and at 12,500,992 actions to mean **76.55**, best
126 and **11/20 level-2 games**. All suites completed; none reached level 3 yet.
The intervening 12,001,280-action checkpoint regressed to mean 50.75, illustrating
why the best checkpoints are retained rather than assuming the latest is best.

A validation-only faster-control test used the frozen 11,501,568-action model
at 50,000 rather than 100,000 T-states per decision. It completed 20/20 games,
mean 65.35, best 112 and 5/20 level-2 games, versus mean 69.85 and 8/20 at the
trained frequency. This does not test retraining at the faster frequency; it
showed no immediate benefit, so the main configuration was left unchanged.
[Diagnostic records](results/level5/faster-control-diagnostic.json).

## Lower-learning-rate comparison

A secondary validation audit of the 12,500,992-action checkpoint used seeds
10100-10149 (not the reserved final test). All 50 games completed: mean **72.84**,
median 60.5, best 126, level 2 in **23/50**, no level 3. This supports a real
level-1 consistency improvement, without claiming the level-5 goal is met.
[Audit records](results/level5/validation-audit-12m.json).

Subsequent main-run checkpoints fluctuated below this candidate (13 million:
mean 59.9, 6/20 level-2 games; 13.5 million: mean 68.8, 6/20). A bounded
two-million-action lower-learning-rate continuation was therefore started from
the saved 12,500,992-action model while the unchanged run continues. Only the
learning rate changes, from 0.00025 to 0.0001. Both use the same validation
suite. As with every resume, the trial starts fresh emulator episodes, so
this is not a promise of paired, identical training trajectories.

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/level5-lr1e4 \
  --resume runs/level5-terminal-fix/step-012500992 --additional-steps 2000000 \
  --learning-rate 0.0001 --target-level 5 --target-clears 1 \
  --eval-games 20 --eval-seed 10000 --eval-every 500000 --eval-max-steps 0
```

Resource checks with both learners found combined throughput around 1,900
actions/second and no increase in swap-outs during the measurement window.
Their individual learning curves must be compared by additional actions, not
elapsed time. The trial's smaller approximate KL values confirm smaller policy
updates; they are not themselves evidence of improved game performance.

The lower-rate trial's first validation, at 13,000,704 actions, completed 20/20
games: mean **70.3**, best 115 and **9/20 level-2 games**. The unchanged run at
the same counter had mean 59.9 and 6/20. This is an encouraging early comparison,
not yet proof of a durable advantage or a later-level milestone. Both runs
were active at this point, with no new parser failures. GPU sharing lengthens serial
validation, but each observed long validation pause has completed normally.

After the unchanged run had produced comparison checkpoints through 14,503,936
actions (mean 54.6, 2/20 level-2 games), it was gracefully paused at **14,733,312**.
`latest` was saved and the process exited successfully. This was an experiment
scheduling decision, not a failure or goal completion. The lower-rate trial
continues with the GPU to itself; the unchanged run can be resumed if needed.

The lower-rate trial's 13,500,416-action validation completed 20/20 games:
mean 70.8, best 127, 7/20 level-2 games and no level 3. It still has not improved
on the source checkpoint's peak validation mean or reached the requested goal.

The two-million-action trial finished successfully at **14,503,936** actions.
Its final validation completed 20/20 games: **mean 77.2, median 65.5, best 126,
11/20 level-2 games**. The unchanged run at this same counter had mean 54.6 and
2/20 level-2 games. The trial's last-100 training mean reached 78.57, with 52
level-2 games and one level-3 game. The level-3 training episode was score 134
at action 14,331,952; no frozen validation game has reached level 3 yet.

The lower rate was selected for a further **five-million-action continuation**
from the trial's final checkpoint, keeping all other learning settings unchanged:

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/level5-lr1e4-extended \
  --resume runs/level5-lr1e4/latest --additional-steps 5000000 \
  --learning-rate 0.0001 --target-level 5 --target-clears 1 \
  --eval-games 20 --eval-seed 10000 --eval-every 500000 --eval-max-steps 0
```

This run was subsequently paused as described below. The earlier comparison processes have exited successfully
and their checkpoints remain preserved. The level-5 goal is still unmet; the
five-million-action endpoint is the next assessment point, not a reason to
declare the autonomous goal complete.

The extension's first validation at 15,003,648 actions completed 20/20 games:
mean 57.7, best 127 and 5/20 level-2 games, with no higher levels. This is below
the selected source checkpoint; that checkpoint remains preserved while the
extension gathers more evidence. The full regression suite now passes **18
tests**, and the original replay's codec and playback checks still pass.

## First frozen-checkpoint level-3 game

At **15,503,360** actions, the extension completed 20/20 validation games with
mean **71.95**, median **58.5**, best **196**, level 2 in **8/20**, and level 3
in **1/20**. None reached levels 4 or 5. The numbered checkpoint and
`best-level` both have SHA256
`364cdcebf08504b038365c8648478b0c2332a751d5030148f12cdcba0de50da3`.
This is the first level-3 result from a frozen, saved policy, rather than an
episode played across changing training weights.

The seed-10010 game was recorded separately and reproduced the score 196,
displayed level 3, 8,742 actions, and complete GAME OVER result exactly:
[level-3 validation replay](results/level5/level-3-validation-replay.html).
Replay checks now also compare the highest level in decoded screen frames
with the metadata and can cross-check the seed's evaluation record:

```sh
node tests/test_replay.js results/level5/level-3-validation-replay.html \
  runs/level5-lr1e4-extended/step-015503360/evaluation.json
```

These strengthened checks pass for both this new replay and the untouched
original replay. The original model's SHA256 remains unchanged.

The secondary audit on seeds 10100-10149 completed all 50 games: mean **74.74**,
median **61**, best **126**, level 2 in **20/50**, and no level 3 or higher.
The level-3 validation replay is therefore a reproducible first reach, not
evidence of consistent level-3 performance. [Secondary audit records](results/level5/validation-audit-15m-level3.json).
Final-test seeds 30000-30099 remain untouched.

During the continuing run, a complete training episode reached **level 4**,
score **228**, at action **15,660,689** (episode 11982, 10,177 actions). This
episode used changing training weights and is not a frozen-policy validation
claim. The next scheduled frozen checkpoint, at 16,003,072 actions, completed
20/20 validation games: mean 67.8, best 122, 7/20 level-2 games and no higher
levels. Training continues unchanged; the level-3 checkpoint stays preserved
as the deepest validated candidate.

The **16,502,784**-action checkpoint also completed 20/20 validation games and
reached level 3 in 1/20: mean **73.85**, median **60**, best **154**, level 2 in
8/20. Its SHA256 is
`06124e74778bc0a27ec06b7ae5f0502969eb4a64175198a5382dd877c1caa3da`.
Two more complete training episodes reached level 4 with score 260, at actions
16,468,726 and 16,522,753. These remain training observations, not frozen-policy
level-4 claims. The 50-game secondary validation audit completed all games:
mean **67.14**, median **55.5**, best **125**, level 2 in **15/50**, and no level
3 or higher. [Audit records](results/level5/validation-audit-16m-level3.json).
No learning setting has changed in the extension.

At **17,002,496** actions, all 20 validation games completed: mean **78.85**,
median **66.5**, best **123**, 9/20 level-2 games and none deeper. This is a new
mean-score record on the primary suite, but not a new level milestone.

## Exploration-strength comparison

With later-level successes still rare in frozen-policy validation, a separate
two-million-action trial starts from the preserved **15,503,360**-action
checkpoint. It changes the entropy coefficient from **0.01 to 0.003** while
keeping the learning rate at 0.0001. The hypothesis is that less pressure to
keep actions random may improve long-game survival. It may instead reduce
useful exploration or cause serve stalls; the comparison must establish the
outcome. The unchanged extension initially continued as the control.

This source was chosen over the 16,502,784-action candidate because both had
1/20 level-3 games on primary validation, while the former had stronger
secondary-validation performance (mean 74.74 versus 67.14). This remains model
selection, not final held-out testing. Resuming starts fresh emulator episodes,
so the two training trajectories are not promised to be identical.

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/level5-entropy003 \
  --resume runs/level5-lr1e4-extended/step-015503360 --additional-steps 2000000 \
  --learning-rate 0.0001 --entropy 0.003 --target-level 5 --target-clears 1 \
  --eval-games 20 --eval-seed 10000 --eval-every 500000 --eval-max-steps 40000
```

The trial's diagnostic validation guard is **40,000 actions per game**, to
detect stalls under reduced exploration. This is not a learning change;
completed games are otherwise evaluated on the same 20 seeds with the same
sampled-policy procedure. An incomplete suite cannot win checkpoint selection
or meet the target. The guard changes the validation protocol, so inherited
best-selection comparisons are reset. Compare completed suites explicitly at
equal additional action budgets. The final 100-game test remains unbounded
and its reserved seeds remain unused.

The comparison's first validation, at **16,003,072** actions, completed 20/20
games within the diagnostic guard: mean **72.5**, median **58.5**, best **191**,
level 2 in **5/20** and level 3 in **1/20**. At the same counter, the control
had mean 67.8, 7/20 level-2 games and no level 3. This is an early deeper-game
result, not enough to establish a durable advantage. Both runs were active at
this first comparison point.
An initial resource check found no increase in swap-outs while both trained.

The control's later **17,502,208**-action validation completed 20/20 games,
mean **69.35**, median **59**, best **119**, level 2 in **6/20** and none deeper.
The stronger numbered snapshots and the separate best-mean/best-level snapshots
remain preserved; these fluctuations are not treated as permanent progress.

At **16,502,784** actions, the reduced-entropy trial completed 20/20 games with
mean **86**, median **86**, best **123**, and level 2 in **12/20**. No game
reached level 3 or higher in this suite, and none hit the diagnostic guard.
Its model SHA256 is
`5b66ea122c27572d9349f8efb3757befe1d05c1d59bf0e1ffb06a07cb874a5c8`.
The control at the same counter had mean 73.85 and 8/20 level-2 games, but also
one level-3 game, so this is a consistency gain rather than a deeper milestone.

The control's **18,001,920**-action validation completed all 20 games with mean
**64.65**, median **62**, best **119**, and 4/20 level-2 games, none deeper.
After this regression, and with comparison checkpoints already covering the
trial's full interval through 17.5 million actions, the control was gracefully
paused at **18,366,464**. It saved `latest` and exited successfully; this is an
experiment scheduling decision, not failure or goal completion. The paused
model SHA256 is
`30ef32fe2d1bb671aef7011e9a0843f66ceb6c3254a8ba674dabeb3adb34e869`.
The reduced-entropy trial continues alone to finish the planned comparison.

## First frozen-checkpoint level-4 game

At **17,002,496** actions, the reduced-entropy trial completed 20/20 validation
games with **mean 97.45, median 98, best 241**, level 2 in **14/20**, and levels
3 and 4 in **1/20** each. No game reached level 5. The model SHA256 is
`c971dbf6a2cdaf84bd98e03245c2a206ccdda009b609fd2e73962b0028522a4e`.
The complete level-4 game used seed **10010**, lasted **10,387 actions**, and
ended normally at GAME OVER. The control at this same counter had mean 78.85,
9/20 level-2 games, and none deeper. This is both a consistency improvement and
a new frozen-policy level milestone for the reduced-entropy comparison.

The trial previously had a complete level-4 training episode at action
16,731,325, score 258. Unlike that changing-weights training observation, the
17,002,496-action validation result comes from a single frozen checkpoint.
The checkpoint is preserved while the trial finishes its action budget and
the autonomous level-5 goal continues.

The seed-10010 recording reproduced the validation result exactly, with
10,388 stored frames: [241-point level-4 replay](results/level5/level-4-validation-replay.html).
The replay's decoded highest level, final screen/score, actions, and metadata
match the evaluation record, and all playback-control checks pass. It is a
separate artifact; the original public replay is unchanged. The 50-game secondary
validation audit on seeds 10100-10149 completed all games: mean **79.74**, median
**67**, best **132**, level 2 in **24/50**, and no higher levels. Later-level
success is still rare outside the primary validation suite.
[Audit records](results/level5/validation-audit-17m-level4.json). This does not
use the reserved final-test seeds.

## First level-5 training episode and continued verification

A complete training episode reached displayed **level 5**, score **278**, at
action **17,091,749** (episode 12387, 14,174 actions, start T-states 136615).
It used changing training weights, so it is evidence of progress but not enough
to satisfy the frozen-policy validation and final-test requirements.

The comparison's final scheduled validation at **17,502,208** completed 20/20
games: mean **97.85**, median **110**, best **181**, level 2 in **14/20**, level 3
in **1/20**, and no levels 4-5. The run then finished its action budget and exited
successfully at **17,506,304**, saving `latest`. The last additional rollout was
not separately validated; do not mistake `latest` for the selected candidate.

The preserved **17,002,496**-action checkpoint was selected for continuation
because it has the strongest validated level-reach rank, including a level-4
game. Learning settings stay at learning rate **0.0001**, entropy **0.003**,
and the same screen input, score reward, and network-selected actions. Validation
is increased from every 500,000 to every **250,000 actions** to check snapshots
sooner as training begins to reach the target. Training now announces validation
start and prints each completed validation game, making these GPU-usage dips
easier to distinguish from a stopped process.

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/level5-entropy003-extended \
  --resume runs/level5-entropy003/step-017002496 --additional-steps 5000000 \
  --learning-rate 0.0001 --entropy 0.003 --target-level 5 --target-clears 1 \
  --eval-games 20 --eval-seed 10000 --eval-every 250000 --eval-max-steps 40000
```

This continuation subsequently reached the validation target as described
below. The comparison and its 50-game audit had both exited successfully; the
reserved final test was still untouched at the start of this continuation.

## Frozen level-5 success and final test

The first closer-spaced checkpoint, at **17,252,352** actions, completed 20/20
validation games: mean **91.55**, median **65.5**, best **228**, level 2 in 9/20,
level 3 in 2/20 and level 4 in 1/20. None reached level 5.

At **17,502,208**, the complete validation suite reached **level 5** in 1/20
games: mean **88**, median **64**, best **278**, level 2 in 10/20 and levels
3–5 in 1/20 each. Seed 10016 reached the target and ended at GAME OVER after
15,346 actions. The trainer saved the numbered, best-level and latest snapshots
and exited successfully with `target_met: true`.

This snapshot was copied to `models/breakdown-level5` and its weight, optimizer
and state hashes were checked before the pre-reserved final test. The fixed
model SHA256 is
`f84d6346798330742a1890e9ee6308fbc380c155bac9d24086dde10bf62d3b87`.
The 100-game test and the matched random baseline both completed every game
on seeds 30000–30099 with no action limit. Results are summarized at the top.
The weights remained unchanged through testing and recording. No final-test
result was used to choose or retrain the checkpoint.

The standalone replay records the **validation** game, not the final test's best
game. Re-recording the fixed model without an action limit reproduced all
15,346 actions' resulting score/level/termination metadata. Replay tests decode
all 15,347 frames, verify the highest displayed level and final GAME OVER screen,
cross-check the validation record, and exercise playback controls. The original
model, original evaluation records and original public replay file match the
previously delivered versions.

The six archived continuation/comparison runs consumed **16,699,392 additional
training actions**, excluding small smoke tests. The selected checkpoint's own
lineage is 17,502,208 PPO actions plus 500,000 DQN initialization actions; these
lineage and total-compute counters are different because experiments forked
from earlier snapshots. No further training was needed for this first-reach
goal after the frozen-policy success and the planned final evaluation.

Final checks: **25 numerical, environment, checkpoint and artifact tests pass**;
the original and level-5 replay checks pass; all five packaged file checksums
match; archived logs match their stopped source runs; the final chart regenerates
directly from the archive; and the original model/replay/evaluation files have
no changes against their previously delivered versions. The replay checks use
a mocked DOM/canvas for control logic, not a full browser visual test.
An independent second recording of the frozen level-5 game was byte-for-byte
identical to the delivered HTML, including all frames and actions (SHA256
`e290a0c4efd8bbc55c947d7a86292efbb76bbd187e245aabfe04889745d46d6c`).
