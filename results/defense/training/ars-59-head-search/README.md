# Real-game categorical-head search, trial 59

This completed six-generation calibration searched the strong Defense PPO
policy's 20-action head with ARS V1 style paired perturbations. The screen
encoder and value head stayed fixed. Each perturbation played complete games
in the real emulator; displayed score was the sole fitness. The PPO optimizer
is preserved but was not updated. There were no scripted actions, demonstrations,
hidden-game-state policy inputs, or evaluation-game gradients.

The parent is `ppo-09-curriculum-life/step-000008342272`. Each generation used
16 paired Gaussian directions, two paired training seeds per direction,
sigma 0.005, and normalized step size 0.002. Training seeds start at 70000;
the 10 fixed boot-validation seeds are 10000–10009. Full plans and returns
are in `run/population-*`.

The 384 training games used **969,966 new actions**. No game reached stage 2
or completed a mission. Initial ten-game mean was 9,981; final mean was
**10,090**, median **10,420**, best **10,460**, all stage 1. Best intermediate
mean was 10,412 at generation 4, also all stage 1. These reused validation
seeds are model-selection diagnostics, not a fresh generalization estimate.
The global best remains the 10,480-point PPO effort.

`run/` preserves every full generation checkpoint and search RNG, the parent
PPO optimizer, configuration, raw returns, evaluations, collector artifacts,
and metrics. The final frozen model was independently rerun in
`../../diagnostics/ars-59-final-replay`; all **2,589 neural actions, screens,
and rewards** reproduced exactly. A read-only screen comparison is in
`../../diagnostics/shared-loss-ars-59/report.json`. The new replay's four
lives scored 2,620 / 2,620 / 2,620 / 2,600 near the recurring right-opening
barrier. The panels cannot establish an exact collision time or direct cause.

`regression-tests.txt` records all **419 tests passing**. The loader and
native population tests reproduce actions from saved weights. The original
[ARS paper](https://arxiv.org/html/1803.07055v1) used linear continuous
policies; this trial adapts its return-difference update to a learned
categorical head.

Reproduce from the parent checkpoint:

```bash
venv/bin/python -u -m rl.defense_ars_fit \
  --initialize results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --output runs/defense-ars-59-reproduction \
  --generations 6 --directions 16 --repetitions 2 \
  --sigma .005 --step-size .002 --envs 16 --seed 71 \
  --first-training-seed 70000 --eval-every 1
```
