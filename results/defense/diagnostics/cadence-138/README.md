# Frozen-policy action-cadence diagnostic

The latest learned key-duration policy still loses at original stage-one
stream rows 33–34. Prior random and phase-changing exploration did not
establish passage. A separate visible-screen probe shows two openings
arriving close together, raising the narrower question whether the
default **100,000-T-state base-action cadence** hides useful control
opportunities between screens.

This is a **predeclared evaluation-only diagnostic**, not training. Freeze
the stronger option-credit checkpoint selected by run 136 at action
1,179,648. Evaluate it from original boot on the same **64 untouched
complete-game seeds 608000–608063** at **100,000 / 50,000 / 25,000**
T states per policy decision, respectively. The 100,000 arm is native
checkpoint timing; the other two are explicit temporal overrides. No
weights, reward, action set, score parser, game bytes, or evaluation
temperature change. The policy still sees only four rendered screen
frames; no hidden course pointer or geometry enters play. Record all
three full-game score distributions, incomplete games, stage reach and
paired differences. Any later-stage observation needs independent native
reexecution at that same cadence before it can inform a new training
experiment. An override result cannot promote the protected replay and
does not prove a 100,000-cadence policy can clear the game. If no arm
reaches stage two, report whether this simple cadence change improves
or regresses complete-game play; do not infer that all finer-cadence
training is impossible.

```sh
venv/bin/python -u -m rl.defense_evaluate \
  results/defense/training/ppo-duration-credit-136/run/step-000001179648/model.safetensors \
  --output runs/defense-cadence-138/native-64.json \
  --games 64 --seed 608000 --envs 10
```

Repeat with `--tstates 50000` and `--tstates 25000`, writing to separate
`half-64.json` and `quarter-64.json` results. Neither override supports
the normal replay bundle, so save its complete evaluation record only.

## Result

All **192** evaluations completed; none reached stage two or a mission.
The same frozen weights produced:

| T states / decision | Mean | Median | Best | Scores below 9,000 |
| ---: | ---: | ---: | ---: | ---: |
| [100,000](native-64.json) | 10,475.16 | 10,480 | 10,480 | 0 / 64 |
| [50,000](half-64.json) | 7,649.84 | 8,075 | 9,690 | 61 / 64 |
| [25,000](quarter-64.json) | 7,418.13 | 7,460 | 9,710 | 63 / 64 |

Both finer-cadence arms lost **all 64 paired games** against native timing.
The frozen policy plainly does not transfer by just calling it more often:
its observation history, hold lengths and learned temporal dynamics all
change in real game time. This negative override test does **not** show
that a separately trained finer-cadence policy would fail, nor does it
identify the physical collision cause. It gives no basis to change the
protected best replay or claim a later-stage result.
