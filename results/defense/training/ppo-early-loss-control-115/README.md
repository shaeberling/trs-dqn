# Matched 128-action own-loss control

This arm continues from the same full model, optimizer and policy RNG as
[run 114](../ppo-early-loss-114/README.md), with the existing 128-action
training-only rewind. All other settings, evaluation seeds, duration and
selection rules match. It distinguishes a longer-lookback effect from ordinary
continuation and training variance. Both arms use learned screen-only actions,
displayed-score reward, and complete games from boot for evaluation.

The first [full model/optimizer/RNG milestone](milestone-000008669952/state.json)
is preserved after 131,072 new actions. Ten complete fixed-seed games
averaged **10,430**, median **10,440**, best **10,480**; all stayed in stage
one. The longer-lookback arm averaged 10,385. These early small checks are
not independent confirmation or evidence of barrier passage.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-early-loss-control-115 \
  --artifacts runs/defense-ppo-early-loss-control-115/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 128 \
  --curriculum-restored-life-only --life-terminal
```
