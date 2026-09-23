# PPO practice from its own visible life-loss approaches

This bounded continuation starts from the independently verified strong
8,342,272-action PPO model **and optimizer** (`ppo-09-curriculum-life`), not
from ARS's unconfirmed short-sample candidates. It keeps the original
four-screen categorical policy and displayed-score reward. Only the
training-state distribution changes: after this learner visibly loses a
life, eligible workers archive their exact own native state from 128
actions earlier. A restored training segment ends at its first visible
life loss; four of sixteen workers always play full games from boot.
Normal complete-game evaluation cannot access the archive. The snapshot
is never a policy input, route, collision label, action target or bonus.

The run finished cleanly after **131,072 new actions**. It logged **24 new
complete boot games and 508 restored practice segments**; no recorded game
reached stage two. Two ten-game evaluations on the reused fixed seeds
10000–10009 averaged **10,269** at 8,407,808 actions and **10,137** at
8,473,344, against the preserved parent's **9,981**. All thirty games
stayed in stage one. Both full model/optimizer/RNG checkpoints, the exact
parent, complete metrics, source/configuration and an independently native-
verified local 10,480-point replay are archived here. The older global
10,480-point replay remains unchanged.

A separate **64 fresh complete-game paired check** used training-only seeds
600000–600063, which neither PPO update nor ARS proposal selection used.
The parent averaged **9,673.59**, the first checkpoint **10,194.84**
(**+521.25**), and the final checkpoint **9,977.66** (**+304.06** versus
parent). The first checkpoint improved 42 seeds, worsened 17 and tied five;
its below-9,000-point games fell from the parent's 22 to seven. All **192
fresh games** remained in stage one. Thus targeted practice reduced
unusual early failures, but did not move the shared late failure. The
first checkpoint is the stronger independently checked optimizer for a
longer exploratory continuation; these paired seeds are not reused to
judge that next run.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-own-loss-107 \
  --artifacts runs/defense-ppo-own-loss-107/artifacts \
  --resume results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --steps 8473344 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 65536 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 128 \
  --curriculum-restored-life-only --life-terminal
```
