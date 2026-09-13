# Breakdown: trained screen-only PPO policy

This is the frozen level-clearing checkpoint selected **before** final evaluation.
All gameplay actions, including serves, come from the neural network. Inputs are
four consecutive video-memory screens; training used only its own experience
and the on-screen score reward.

Final test: 50 complete games, seeds 20000–20049. Mean **44.3**, median **45.5**,
best **88**, highest level **2**, level-1 clears **3/50**. This is a first-clear
milestone, not a consistently winning policy. See [full records](../../results/trained.json).

## Files and provenance

- `model.safetensors`: trained network weights.
- `state.json`: algorithm, environment settings, counters, and RNG state.
  Keep it beside the weights; it selects categorical PPO evaluation.
- `optimizer.npz`: Adam state for continued training.
- `evaluation.json`: the five-game validation result used to select this model.

Original checkpoint: `runs/ppo-settled/step-006402048`.
Lineage: 500,000 DQN actions plus 6,402,048 PPO actions. Validation had one
level-1 clear in five complete games (mean 50.6, best 66). The model was not
selected by final test performance.

Model SHA-256:

```text
a54161326684ca4e63dec62e68053f9363a44640b75cab69f06bf59b85f7da69
```

## Use

After the [repository setup](../../TRAINING.md), run from the repository root:

```sh
venv/bin/python -m rl.evaluate models/breakdown/model.safetensors \
  --games 50 --seed 20000 --max-steps 0 --output results/re-evaluation.json

venv/bin/python -m rl.evaluate models/breakdown/model.safetensors --watch --seed 20005

venv/bin/python -m rl.ppo --run runs/continue --resume models/breakdown
```

The learned policy is sampled with fixed per-game seeds. There is no random
action override. Resume restores parameters and optimizer but starts fresh
emulator trajectories. Exact reproducibility assumes the pinned dependencies,
current wrapper, and Apple Silicon MLX backend used here.

The [portable 88-point replay](../../results/replay.html) requires only a browser.
