# Original-game winning policy

This frozen, screen-only PPO policy has beaten all eight original Breakdown
levels in a complete reused validation game. The game itself is unchanged.

Model SHA-256:
`b74d8f62e3414afef7e74fba504d35e5cb085c5dc3080e712562fafabe5b7d2c`

The [585-point winning replay](../../results/level10/best/replay.html) contains
49,159 verified neural actions, zero mismatches, seed10001, and one reserve
ball remaining at the final GAME OVER. The game uses that message for both
victory and defeat; the reserve proves this is the final-level victory path.

Selected before fresh testing from three predeclared historical checkpoint
comparisons. Reused validation:70/70 complete, mean298.94, median280, best585,
eleven reaches of Level8, one verified win and ten unverified last-ball Level8
endings. These are selection results, not an unbiased success-rate estimate.

The one-time 100-game fresh test finished: mean **278.75**, median **277.5**,
best **583**, highest **Level 8**, **zero verified wins**, and **three
unverified last-ball final-level endings**. All seeds40000–40099 completed
naturally, without a step limit or policy overrides. These seeds are now used;
do not treat them as fresh in future experiments. This model can win, but
reliable winning has not been demonstrated. See the
[complete test records](../../results/level10/all-eight-fresh-test.json) and
[completion audit](../../results/level10/all-eight-completion-audit.json).

Files retain the exact frozen weights, optimizer and original checkpoint
configuration. `selection.json` includes the outcome-labelled70-game audit.
The original `evaluation.json` predates win labels; use the selection record
for current validation outcomes. No teacher model is needed for inference.
Continuing training would require its own archived reference checkpoint.

From the repository root, with its existing emulator and dependencies:

```bash
venv/bin/python -m rl.evaluate models/breakdown-all-eight/model.safetensors \
  --games 10 --envs 10 --seed 50000 --max-steps 0
```

This is a sampled learned policy, not an argmax override:50,000 T-states per
action, observation stride2, four raw video-memory frames, six learned key
combinations. Rewards remain visible score differences only. Last-ball Level8
endings are conservatively unverified, so verified win rates are lower bounds.
See [TRAINING.md](../../TRAINING.md) for the method and outcome proof.
