# The frozen score critic predicts the familiar reward plateau

This [read-only report](report.json) validates the confirmed run-121
[own-policy replay](../../training/ppo-continue-121/fresh-confirm-selected-replay/replay.html),
reconstructs its four visible input frames at each inspected action, and
queries the frozen PPO value head. Predictions are divided by the unchanged
0.01 score-unit scale for comparison with the subsequent **displayed-score
increments within that life**. The diagnostic makes no parameter update,
action choice, reward change or replay promotion.

At **64 decisions before each visible loss**, the critic predicts roughly
**1,433–1,481** more displayed points; the recorded suffixes earn
**1,520–1,540**. By **32 decisions before loss**, all four predictions
have fallen to **6–44** points, while the recorded suffixes earn only
**0–20**. Each life ultimately ends at 2,620 points. The critic is not
uniformly exact at every offset, but this repeated drop follows the
late 1,500-point score jump and shows that the learned score value largely
expects little additional *within-life* reward at the obstacle.

This is one selected replay, not a held-out calibration study or a causal
proof. A passing trajectory could receive later score and change the value
target, but no such stage-two trajectory has been observed. The experiment
clarifies why repeated score-only PPO updates can become uninformative near
the shared loss once the final score jump has been collected. It does not
authorize a collision bonus, scripted route or hidden-state input; future
training must still learn from its own screen observations and original
score/visible boundaries.

```bash
venv/bin/python -m rl.defense_value_probe \
  results/defense/training/ppo-continue-121/fresh-confirm-selected-replay \
  --output runs/defense-score-value-probe-reproduction.json
```
