# Trainable fixed grouped-fire PPO policy

The frozen [canonical-fire diagnostic](../../diagnostics/action-group-110/README.md)
retained the original model's fire-versus-movement probability and
matched all 64 complete-game scores. This trial now makes the **same
fixed twelve-group action distribution trainable**: the screen-only
network still emits twenty logits, but PPO samples and learns the sum
of fire-alias probabilities as one group. That group always emits Space;
commands 0–8 and 18–19 stay distinct. There is no screen-dependent
action rule, route, extra reward or game modification. Later-stage
equivalence of the grouped keys remains unverified.

The parent is the independently stronger run-107 first checkpoint,
including its exact optimizer and policy RNG at 8,407,808 actions. The
own-loss training curriculum stays unchanged: 128-action lookback,
first-life-loss restored cuts, twelve eligible practice workers and four
reserved complete-boot workers. Entropy stays at the parent's 0.002.
The bounded test adds 524,288 actions and four ten-game complete-boot
evaluations. Its same-parent ungrouped control is run 113. Only later
stage reach or independently fresh full-game results can establish
progress; the original global verified replay remains preserved.

The live output will be `runs/defense-ppo-canonical-112/`. Exact
configuration/source and all full model/optimizer/RNG milestones will be
archived here.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-canonical-112 \
  --artifacts runs/defense-ppo-canonical-112/artifacts \
  --resume results/defense/training/ppo-own-loss-107/checkpoint-000008407808 \
  --steps 8932096 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --canonical-fire --curriculum-probability 1 --curriculum-share \
  --curriculum-boot-envs 4 --curriculum-trigger life-loss \
  --curriculum-lookback 128 --curriculum-restored-life-only --life-terminal
```
