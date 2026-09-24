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

The completed live output was `runs/defense-ppo-canonical-control-113/`;
its entire model/optimizer/RNG, log and verified local replay history
is archived in [run/](run/).

Both runs keep isolated artifact roots, watched by the one independent
collector. The global verified best replay is never overwritten by a
lower-ranked result.

The first [full optimizer/RNG milestone](milestone-000008538880/state.json)
is preserved after **131,072 new actions**. Its ten complete fixed-seed
games averaged **10,438**, median **10,435**, best **10,480**; all stayed
in stage one. This is an early score check, not a barrier clear or an
independent result. The grouped arm's first mean was 9,457; both learners
subsequently completed the predeclared endpoint.

Both learners completed their planned **524,288 new actions**. This
control added **64 complete boot games and 2,095 restored segments**.
Its four fixed ten-game means were **10,438 / 10,428 / 10,139 / 9,314**,
all stage one. The first [8,538,880-action checkpoint](milestone-000008538880/state.json)
was selected by those fixed ten games before fresh checking. On [128
new matched complete games](fresh-selected-128.json), it averaged
**10,370.78**, versus the common parent's [10,173.75](../ppo-canonical-112/fresh-parent-128.json):
**+197.03**. It improved 91 paired seeds, worsened 23 and tied 14;
below-9,000-point games fell from 15 to five. The grouped arm averaged
9,825 on the same seeds. All 384 fresh games across the three frozen
policies stayed in stage one. This checkpoint is a confirmed **score**
improvement and retains its full optimizer/RNG for further experiments,
but it does not solve the recurring course barrier. Its native-verified
10,480-point local replay ties, rather than displaces, the global best.

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
