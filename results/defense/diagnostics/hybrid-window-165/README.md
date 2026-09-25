# Frozen-policy movement-window diagnostic (run 165)

This is a bounded *reachability diagnostic*, not a trained hybrid controller.
The source is the independently verified, original-boot first life of the
balanced-fire continuation at seed 612059. It visibly loses a ship after
action 412, with 2,620 displayed points. The intervention samples actions
from the frozen movement-only PPO checkpoint for a fixed window, then returns
to the original frozen balanced-fire PPO policy. The fixed switching schedule
is scripted and cannot be promoted as a learned-policy replay or used as a
demonstration. No game internals, route, oracle, reward change, or model update
enters either neural policy.

The complete grid is declared before running it: starts at actions
**300, 320, 340, 360**; windows of **16, 32, 48, 64** actions; **16**
independent movement-policy draws per cell; first visible ship loss, stage
advance, or action **520** ends a branch. That is **256** branches from one
exact own-policy first-life source. Before branching, the probe must replay
the entire source action/screen/reward trace from original boot and verify
that each restored start reproduces the source's exact frozen-policy suffix.
Only displayed score and visible life/stage events rank outcomes. No branch
action trace is saved as training data.

Decision gate: any branch reaching stage two or lasting at least **20 actions
beyond** the source's first visible loss at action 412 merits a separate
original-boot confirmation and consideration of a *learned* screen-conditioned
switching mechanism. If neither happens, do not start a long two-expert-gate
trainer on this evidence. Either outcome is specific to these two checkpoints,
one source life and this grid; failure cannot prove that the course is
impassable. Action count includes animations and is not exact course depth.

The two-branch [integration smoke](smoke-report.json) at start 320/window 32
reexecuted the source exactly. Both switches died earlier (actions 362 and
365), so the smoke is not positive reachability evidence.

## Completed result

Both focused native tests and the full **573-test** native repository suite
passed. The [full report](full-report.json) contains
all **256** predeclared branches and confirms exact source-prefix and restored
source-suffix parity. The latest first visible loss was action **413**, just
one action beyond the source's **412**; only **7/256** branches reached action
412 or later, and **5/256** reached 413. No branch lasted 20 additional
actions, reached stage two, survived to the 520-action cap, or displayed a
mission. The maximum first-life score was the same **2,620** points, reached
by 64 branches. Earlier movement windows commonly lost before collecting
that score; windows starting at 360 often kept the score but still lost at
the familiar boundary.

This fails the predeclared gate for training a screen-conditioned switch
between these *two frozen experts*. It does not rule out a newly trained
policy, a different route, or a different intervention start. In particular,
the movement-only expert was trained without fire and itself fails much
earlier in the course. The source's original game, protected global best
weights, and verified replay remain unchanged. The next useful work is a
separate control/observability audit of the two-opening approach, not an
unbounded extension of this hybrid.

Reproduce the full diagnostic with:

```sh
venv/bin/python -m rl.defense_hybrid_window_probe \
  --output runs/defense-hybrid-window-165-full.json
```
