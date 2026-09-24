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
