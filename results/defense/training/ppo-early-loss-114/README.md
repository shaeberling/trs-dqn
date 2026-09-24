# Earlier own-loss practice: 256-action lead-in

The repeated stage-one losses in native-verified replays occur near the same
visible barrier sequence, usually after roughly 2,550–2,620 points per life.
Score is not an exact collision locator, and equal scores need not mean equal
positions. This trial tests whether a learner needs more lead-in time to move
before the barrier arrives. It changes only the **training-only** rewind from
128 to 256 decisions before this policy's own *visible* life loss. The model
still receives four screen frames; the reward is only displayed score change.
No collision memory, route, extra reward, scripted steering or demonstration
enters training or evaluation.

Run 114 and [same-parent run 115](../ppo-early-loss-control-115/README.md)
resume the exact same full model, optimizer and policy RNG from
[control 113's confirmed score checkpoint](../ppo-canonical-control-113/milestone-000008538880/state.json).
Their only intended setting difference is the 256-versus-128-action own-loss
lookback. Both start new emulator episodes, retain four boot-only workers, and
continue for 524,288 actions to 9,063,168. Four ten-game fixed-seed checks
are scheduled every 131,072 actions. A checkpoint will be selected using only
those checks, then compared with its parent and control on new, matched,
complete-game seeds 600600–600727. A stage clear requires a screen-observed
transition; score alone is not enough.

The sole independent collector watches this run's isolated `artifacts/` and
preserves the globally verified best replay if no better candidate appears.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-early-loss-114 \
  --artifacts runs/defense-ppo-early-loss-114/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 256 \
  --curriculum-restored-life-only --life-terminal
```
