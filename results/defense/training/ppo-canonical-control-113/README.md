# Same-parent ordinary-action PPO control

This control resumes the **same full model, optimizer and policy RNG** as
run 112 and uses the same 16 workers, own-loss curriculum, action count,
score reward, training duration and evaluation settings. It leaves the
original twenty-command categorical distribution intact. The sole
intended learning/acting difference is run 112's fixed probability-
preserving fire-key grouping. Shared training random streams can diverge
after the two policies select different keyboard commands; matched
complete-game evaluations and new independent seeds are needed to judge
the outcome. No result is inferred from a small score sample or a
stage-one score ceiling.

The live output will be `runs/defense-ppo-canonical-control-113/`.
Both runs keep isolated artifact roots, watched by the one independent
collector. The global verified best replay is never overwritten by a
lower-ranked result.

The first [full optimizer/RNG milestone](milestone-000008538880/state.json)
is preserved after **131,072 new actions**. Its ten complete fixed-seed
games averaged **10,438**, median **10,435**, best **10,480**; all stayed
in stage one. This is an early score check, not a barrier clear or an
independent result. The grouped arm's first mean was 9,457, and both
learners continue to the predeclared endpoint.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-canonical-control-113 \
  --artifacts runs/defense-ppo-canonical-control-113/artifacts \
  --resume results/defense/training/ppo-own-loss-107/checkpoint-000008407808 \
  --steps 8932096 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 128 \
  --curriculum-restored-life-only --life-terminal
```
