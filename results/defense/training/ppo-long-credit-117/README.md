# Longer credit from the same visible-loss practice

The recurring barrier appears roughly 64 decisions before a verified
replay's visible loss. The ordinary PPO continuation uses GAE lambda 0.95,
which gives the residual-return kernel `(0.997 × 0.95)^k` a roughly
13-decision half-life. This trial changes **only training GAE lambda to
0.995**, giving a roughly 87-decision half-life before rollout and life
boundaries. It tests whether score consequences can influence earlier actions
without changing input, keyboard choices, actual reward, environment timing,
rollout length, or the 128-decision own-loss practice window. Longer returns
also have higher variance; the effect is not presumed beneficial.

Run 117 resumes the same full model, optimizer and policy RNG as
[run 116](../ppo-near-loss-116/README.md) and the completed
[128-decision control](../ppo-early-loss-control-115/README.md). The latter
is the matched lambda-0.95 comparison. Four fixed ten-game checks select a
checkpoint before 128 fresh complete games on seeds 600800–600927. The
confirmed common parent and near-loss arm will be tested on those seeds too.
All are screen-only learned policies, with displayed-score reward and no
demonstrations, collision oracle, scripted steering or evaluation-time resets.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-long-credit-117 \
  --artifacts runs/defense-ppo-long-credit-117/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 128 \
  --curriculum-restored-life-only --life-terminal --gae-lambda 0.995
```
