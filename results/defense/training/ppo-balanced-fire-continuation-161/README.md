# Balanced-fire continuation after matched learning gain

The fresh balanced-control trial 160 beat its otherwise matched no-offset
control in all 128 paired untouched complete games, but none of the 256
games reached stage two. Its last 100 own training games still improved to
a 9,448.6 mean at 8,380,416 actions. This continuation tests the limited
hypothesis that the newly effective score-only learner has not converged.
It does not assume that more actions will clear the repeated obstacle.

Resume the immutable [trial-160 terminal checkpoint](../ppo-balanced-fire-from-scratch-160/terminal-checkpoint/state.json)
with its optimizer and policy RNG, using the same original game, four raw
video-memory frame inputs, 12 grouped physical commands, 100,000-T-state
action period, 16 environments, PPO settings, original displayed-score
reward and visible-life boundaries. Emulator episodes restart at original
boot on resume; no hidden game state, route script, demonstrations, searched
beam actions, intrinsic reward, or game patches enter learning. The new
absolute target is **16,777,216** actions: exactly **8,388,608** additional
own actions. Fixed checks use ten complete original-boot games on seeds
10000–10009 every 1,048,576 additional actions. Preserve every fixed
evaluation and model/optimizer/RNG state until final checkpoint selection;
then archive selected and terminal full states, metrics and verified replays
before pruning unselected intermediate snapshots. An exact-run disk guard
signals a clean stop below **5.1 GiB** free.

Selection and stage proof are predeclared: choose the frozen checkpoint
with successful mission first, then highest stage, then fixed ten-game
mean (earliest step on a tie). Any stage-two or successful-mission claim
requires an independent original-boot native replay verification. Compare
the selected model against the unchanged trial-160 selected parent on
**128 fresh matched complete games**, seeds 612000–612127, before claiming
a score improvement. A higher stage supersedes score; a same-stage score
gain alone does not complete the goal or replace the protected global-best
replay without the existing promotion rules. If the run remains in stage
one, do not automatically extend the same setup again without new evidence.

Launch command:

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-balanced-fire-continuation-161 \
  --artifacts runs/defense-ppo-balanced-fire-continuation-161-artifacts \
  --resume results/defense/training/ppo-balanced-fire-from-scratch-160/terminal-checkpoint \
  --steps 16777216
```

The trainer restores inherited settings from the checkpoint and records
the resolved configuration in its own run directory. The full Mac-native
537-test suite passed after the preceding trainer/diagnostic changes;
this continuation changes no code or architecture.

The [`rl.defense_continuation_compare`](../../../../rl/defense_continuation_compare.py)
watcher waits for the exact trainer PID and target. It rejects an early
stop, incompatible source/configuration, missing optimizer or incomplete
fixed checks. If a selected fixed check reports stage two or mission
success, it independently replays the corresponding original-boot seed
before making a claim. It then evaluates the selected frozen continuation
and frozen parent on the predeclared 128 matched fresh games, verifies a
local replay for each, and writes a comparison report without promotion.
Its current state appears in the live run's `comparison-status.json`.
Four focused fail-closed tests pass. The watcher and disk guard are separate
processes; neither changes the learner's action, reward or checkpoint.

## First fixed checkpoint

At **9,437,184** total actions (**1,048,576** new), the first frozen
[ten-game evaluation](milestone-000009437184/evaluation.json) averaged
**9,904**, median **9,900**, best **9,980**. All ten original-boot games
ended in stage one; none completed a mission. The seed-10003
[best-effort replay](milestone-replay-000009437184/replay.html) scored
9,980 over 2,537 learned decisions and independently verified against
model SHA-256
`dcc6f58919ac686215b41c0d4b10eeb1696b7f79a46451825468779e19d9ca6c`.
The full model/optimizer/RNG state, fixed evaluation and resolved replay
bundle were copied byte-for-byte into this archive before the run
continues. This fixed-seed score improvement does not establish a fresh
generalization gain, a stage-two passage or a new protected global best.

## Third fixed checkpoint: strongest so far

The second fixed check at **10,485,760** actions averaged **9,595**,
median **10,130**, best **10,380**; all ten complete games stayed in stage
one. Its higher single-game replay was natively verified but did not beat
the first checkpoint's fixed mean, so its optimizer state remains in the
live run pending final selection rather than another remote duplicate.

At **11,534,336** total actions (**3,145,728** new), the third frozen
[ten-game evaluation](milestone-000011534336/evaluation.json) averaged
**10,220**, median **10,210**, best **10,390**. All ten original-boot
games still ended in stage one, with no mission. Its seed-10006
[10,390-point replay](milestone-replay-000011534336/replay.html) was
independently reexecuted for all **2,474** learned actions with
`verified: true`; model SHA-256 is
`488ac706272025a72080107dc4abd64c2cb5549bafa659111c296610f89ef28b`.
The complete model/optimizer/RNG checkpoint and resolved replay bundle
were copied byte-for-byte into this archive. This is the provisional
stage-then-fixed-mean selection, not a new global-best replay or evidence
of passage. The run continues to its predeclared target.

## Seventh fixed checkpoint: new provisional selection

The fourth, fifth and sixth fixed ten-game means at **12,582,912 / 13,631,488 /
14,680,064** actions were **10,006 / 10,043 / 10,156**; all 30 games
remained in stage one. None surpassed the third checkpoint's fixed mean.

At **15,728,640** total actions (**7,340,032** new), the seventh frozen
[ten-game evaluation](milestone-000015728640/evaluation.json) averaged
**10,258**, median **10,280**, best **10,370**, all complete and stage one.
This narrowly exceeds the third checkpoint's **10,220** fixed mean. Because
the ordinary artifact publisher ranks best *single* games and did not
publish a replay for this higher-*mean* selection, an independent frozen
ten-game evaluation was run on the same seeds and reproduced every game
record exactly. Its seed-10001 [10,370-point replay](milestone-replay-000015728640/replay.html)
then independently reproduced all **2,431** learned actions from original
boot with `verified: true`. Its four life scores were **2,600 / 2,550 /
2,600 / 2,620**: one life again reached the known stage-one score ceiling
without advancing. Model SHA-256 is
`51945f630ef3c0ad3b92033321c03f22c8b08e11d31efbe9615a57de85fadc28`.
The complete model/optimizer/RNG state, both fixed evaluations and resolved
replay bundle were copied byte-for-byte into this archive. This is the new
provisional stage-then-mean selection, not a mission claim or replacement
for the protected global-best replay. The final planned fixed check remains.

## Target and fresh comparison complete

The trainer stopped normally at its exact **16,777,216**-action target.
The final fixed ten-game evaluation averaged **9,844**, median **10,275**,
best **10,410**, all stage one. The predeclared selector therefore retained
the seventh checkpoint as the best frozen model; all eight [fixed checks](fixed-evaluations/)
are archived. The complete [terminal model/optimizer/RNG state](terminal-checkpoint/state.json),
resolved config, terminal status and [full compressed metric log](metrics.jsonl.gz)
were copied from the finished run, with byte comparisons and a
decompression comparison against the source log. Terminal model SHA-256 is
`b9f6a17e2501f793e0ba1f824d2ec16bedeb0a7415797883845a24876ef82b3d`.

The unattended [fresh comparison](fresh-comparison/report.json) evaluated
the selected seventh model and unchanged trial-160 terminal parent on **128
matched complete original-boot games**, seeds 612000–612127. The selected
continuation averaged **9,641.72** (median **10,315**, best **10,480**)
versus parent **9,314.06** (median **9,890**, best **10,170**), winning
**102** paired seeds to **26**, with no ties. Both fresh local replay bundles
were independently native-verified against their respective frozen model
hashes. The continuation's [10,480-point replay](fresh-comparison/continuation-612000-replay/replay.html)
reproduced seed 612059 for **2,525** learned actions but remained in stage
one. All **256** fresh games stayed in stage one with zero missions.

This validates a score-training gain from continuing the balanced prior,
but it does **not** complete the original game or exceed the protected
global-best replay's stage or score. The selected seventh checkpoint is an
eligible score-training parent for a *different* next experiment. Merely
repeating this same continuation again is not justified by these outcomes.
After this archive is committed, retain the selected and terminal full
states, all evaluations, logs and verified replays, and prune unselected
intermediate live optimizer snapshots under the documented retention rule.
