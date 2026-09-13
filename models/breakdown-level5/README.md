# Breakdown: frozen level-5 PPO checkpoint

This screen-only learned policy reached displayed **level 5** (cleared levels
1–4) in a complete validation game, scoring **278**. Every action, including
serves, comes from the network; training used only its own experience and the
visible score reward. There are no gameplay overrides or demonstrations.

| Suite | Complete | Mean | Median | Best | Highest level | Reached level 5 |
|---|---:|---:|---:|---:|---:|---:|
| Selection validation, seeds 10000–10019 | 20/20 | 88.00 | 64 | 278 | 5 | 1/20 |
| Frozen final test, seeds 30000–30099 | 100/100 | 83.32 | 72.5 | 167 | 3 | 0/100 |
| Matched uniform-random test | 100/100 | 1.54 | 1 | 4 | 1 | 0/100 |

**This is a first-reach milestone, not reliable level-5 play.** In the fresh
100-game test, 50 games reached level 2, one reached level 3, and none reached
levels 4 or 5. The model was selected and frozen before that test; its results
were not used for further training or checkpoint selection.

The [278-point level-5 replay](../../results/level5/replay.html) is the selected
**validation** game, seed 10016, not a held-out test game. Its 15,346 actions,
15,347 stored frames, score, level, and model hash reproduce the validation
record exactly. The original level-2 model and public replay are untouched.

## Files and provenance

- `model.safetensors`: fixed trained weights.
- `state.json`: PPO configuration, counters, RNG and selection history; keep
  it beside the weights so evaluation uses the learned categorical policy.
- `optimizer.npz`: Adam state for optional future continuation.
- `validation.json`: the complete 20-game selection suite.
- `evaluation.json`: the fresh, unbounded 100-game final test.
- `SHA256SUMS`: integrity checks for these five files.

Source: `runs/level5-entropy003-extended/step-017502208`. Its checkpoint lineage
contains **17,502,208 PPO actions plus 500,000 earlier DQN actions**; separate
comparison branches consumed additional training. See [experiment notes and
archived logs](../../LEVEL5.md). Checkpoints were ranked by deeper-level reach
counts, then score. `state.json`'s historical `best_mean` is not this model's
evaluation score; use the two evaluation files above.

Model SHA256:

```text
f84d6346798330742a1890e9ee6308fbc380c155bac9d24086dde10bf62d3b87
```

Environment: `normalized-game-over-v2`; Apple M4, Python 3.12, MLX 0.32.2.
The policy sees four video-memory frames and chooses one of six key combinations
every 100,000 Z80 T-states. Final learning rate: 0.0001; entropy coefficient:
0.003. Selection used a 40,000-action diagnostic guard, with every game ending
naturally before it. The final test and replay had **no action limit**.

## Use from the repository root

After the [setup and from-scratch training instructions](../../TRAINING.md):

```sh
venv/bin/python -m rl.evaluate models/breakdown-level5/model.safetensors \
  --games 100 --seed 30000 --max-steps 0 --output results/level5/re-evaluation.json

venv/bin/python -m rl.evaluate models/breakdown-level5/model.safetensors \
  --watch --seed 10016

venv/bin/python -m rl.record models/breakdown-level5/model.safetensors \
  --seed 10016 --max-steps 0 --output results/level5/re-recorded.html

venv/bin/python -m rl.ppo --run runs/future-level5 --resume models/breakdown-level5 \
  --additional-steps 5000000 --target-level 5 --target-clears 3

(cd models/breakdown-level5 && shasum -a 256 -c SHA256SUMS)
```

Resume starts fresh emulator episodes. Exact replay assumes the recorded
software and environment version. The test seeds are now public results, not
fresh data for any future model-selection experiment.
