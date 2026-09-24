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
