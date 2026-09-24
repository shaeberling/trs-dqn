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
