# Longer, exploratory own-loss PPO continuation

This continuation resumes the **first** run-107 model, optimizer and policy
RNG at 8,407,808 actions. That checkpoint beat its original parent by
521.25 displayed points per game in an independent 64-game paired check,
but every game still ended in stage one. The later run-107 optimizer was
weaker on the same fresh seeds and is not the parent here.

The new test keeps 128-action own-visible-loss snapshot resets, first-loss
restored-segment cuts, a 16-worker split with four full-boot workers, and
the same screen-only policy/score-only reward. It raises PPO's *training*
entropy coefficient from 0.002 to 0.01 to sample a broader action mix.
No action is prescribed and normal complete-game evaluation has no state
restoration. The bounded target is 524,288 new actions, with complete
ten-game evaluations every 131,072. The previous 64 fresh comparison
seeds 600000–600063 are not used to select this run's improvement; any
subsequent independent comparison needs new seeds.

The live output is `runs/defense-ppo-own-loss-108/`. The parent and
source/configuration are preserved here; model/optimizer/RNG milestones
and the full log will be archived as the run progresses. The global
10,480-point verified best replay is preserved independently.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-own-loss-108 \
  --artifacts runs/defense-ppo-own-loss-108/artifacts \
  --resume results/defense/training/ppo-own-loss-107/checkpoint-000008407808 \
  --steps 8932096 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --entropy .01 --curriculum-probability 1 --curriculum-share \
  --curriculum-boot-envs 4 --curriculum-trigger life-loss \
  --curriculum-lookback 128 --curriculum-restored-life-only --life-terminal
```
