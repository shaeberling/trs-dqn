# Screen/action comparison of trial 163 and the protected best

This post-checkpoint, read-only comparison uses the independently verified
[9,930-point grouped-duration replay](../../training/ppo-grouped-duration-fresh-163/milestone-000006291456/verified-replay/replay.html)
and the protected [10,480-point replay](../../learned/best/replay.html).
The [report](report.json) checks source hashes and shows four stage-one life
scores of **2,470 / 2,520 / 2,470 / 2,470** for the new model versus
**2,620 / 2,620 / 2,620 / 2,620** for the protected best. The unaltered
[new-policy screen sheet](policy-1-losses.png) and
[protected-model sheet](policy-2-losses.png) show both policies approaching
the same two-offset-opening region with the ship still in the left part of
the screen while a later right-side opening approaches. They are selected
best replays, not matched-seed population estimates.

In the last 64 chosen actions before each visible white-flash alignment
marker (or visible loss if no marker was sampled), the new model chose
ordinary movement on **50.0% / 75.0% / 50.0% / 75.0%** of decisions,
versus **56.3% / 71.9% / 62.5% / 51.6%** for the protected model. These
fractions are commands, not measured displacement. The white flash can
lag collision; the panels do not establish exact collision geometry.

The separate [native course audit](../grouped-duration-163-course-progress-6m/README.md)
found row 34 at all four visible losses. This screen comparison reads no
hidden RAM, and neither diagnostic supplies observations, actions, rewards,
demonstrations, checkpoint ranking or parameter updates to the live learner.
The protected model/replay and the original game remain unchanged.

Reproduce from the two archived bundles with:

```sh
venv/bin/python -m rl.defense_loss_probe \
  results/defense/training/ppo-grouped-duration-fresh-163/milestone-000006291456/verified-replay \
  results/defense/learned/best \
  --output runs/defense-grouped-duration-163-loss-comparison-reproduction
```
