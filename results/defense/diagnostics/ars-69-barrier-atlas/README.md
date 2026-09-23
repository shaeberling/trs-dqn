# Visible-screen atlas of the recurring stage-one loss

These are the unaltered recorded screen bytes from the original strong
learned policy's 12 own complete training games in
[run 69](../../training/ars-score-gated-69/README.md). Each row is one
life; columns are 64, 32, 8 and 1 neural decisions before the **visible**
life-loss marker. Separate sheets show [life 1](life-1-screens.png),
[life 2](life-2-screens.png), [life 3](life-3-screens.png) and
[life 4](life-4-screens.png). The [report](report.json) checks all 48 source
archive hashes and records per-life displayed points and actions.

In 45 of 48 lives, the displayed score gained before visible loss was
between 2,500 and 2,650. The sheets repeatedly show the same broad
barrier sequence, not proof of the same exact collision location. Across
the last 64 recorded actions of every life, the model chose 894 pure
rightward commands, 503 pure leftward commands, and 1,277 firing/side-fire
commands. In stage 1, Space suppresses arrow movement. This makes
movement interruption a plausible contributor, but the action window can
contain post-collision animation and the display does not reveal collision
timing or distinguish a wall hit from a projectile. The counts do show that
the problem is subtler than simply never choosing Right.

This diagnostic is never read by the trainer, changes no rewards or policy
inputs, and did not update any model. It is a way to inspect the user's
shared-failure observation using the agent's own visible traces.

```bash
venv/bin/python -m rl.defense_barrier_atlas \
  results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output results/defense/diagnostics/ars-69-barrier-atlas
```
