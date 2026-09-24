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

The first [full model/optimizer/RNG milestone](milestone-000008669952/state.json)
is preserved after 131,072 new actions. Its ten fixed complete games
averaged **7,888**, median **7,900**, best **10,280**; all stayed in stage
one. This is an early regression, not evidence of improved barrier credit;
the predeclared continuation remains active.

The full 524,288-action run finished cleanly, with **64 boot games and
2,182 restored practice segments**. Its four fixed ten-game means were
**7,888 / 9,445 / 10,152 / 10,402**, all stage one. The final
[full optimizer/RNG checkpoint](run/step-000009063168/state.json) was
selected by those fixed games before fresh testing. The entire model,
optimizer, RNG, metrics and native-verified local replay history is
preserved in [run/](run/).

On [128 new matched complete games](fresh-selected-128.json), seeds
600800–600927, the selected model averaged **10,121.80**, median
**10,400**, best **10,460**; none reached stage two. The common parent
averaged **10,398.91** and the ordinary lambda-0.95 control **10,377.27**
on those seeds. Longer credit improved only **15** paired games over the
parent, worsened **111**, and tied **two**; **14** scores were below 9,000
versus four for the parent. Its **-277.11** mean difference and no stage
passage reject this setting as a confirmed successor. The selected policy's
[fresh best replay](fresh-selected-replay/replay.html) independently verifies
**2,553 neural actions**, but is not a mission win. The global best replay
remains unchanged.

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
