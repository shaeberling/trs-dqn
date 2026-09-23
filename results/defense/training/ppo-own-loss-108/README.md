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

The run finished cleanly after all **524,288 new actions**, adding **47
complete boot games and 1,859 restored practice segments**. Fixed ten-game
means at the four successive checkpoints were **9,464 / 8,903 / 10,178 /
10,457**. Every training and validation record stayed in stage one;
the final attractive fixed-seed mean is not a stage clear. The full run,
including each model/optimizer/RNG checkpoint, complete log and locally
verified replay, is preserved in [run/](run/). The global verified
10,480-point best replay remains unchanged.

A new matched **64-game full-boot check** on seeds 600100–600163 found the
confirmed run-107 parent at **10,183.28** mean and this run's final model
at **10,037.19**, a **146.09-point regression** despite the fixed-seed
rebound. The final policy improved 42 paired scores, worsened 18 and tied
four; below-9,000-point games increased from seven to eleven. All **128
fresh games** remained in stage one. Raising entropy and practicing the
same own-loss region longer is therefore not a confirmed improvement, and
this optimizer is not promoted as the next parent. A different action-
representation/exploration test is warranted.

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
