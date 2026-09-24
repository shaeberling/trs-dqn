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

This arm finished its planned 524,288 new actions with **61 boot games and
1,958 restored segments**, without a stage-two training or evaluation game.
Its four fixed ten-game means were **10,430 / 9,708 / 10,229 / 10,081**,
all stage one. The first checkpoint above was selected by those fixed games
before the new test. The complete model, optimizer, RNG, log, and verified
local replay history is preserved in [run/](run/).

On [128 new matched complete games](fresh-selected-128.json), seeds
600600–600727, the selected control averaged **10,291.80**, median
**10,460**, best **10,480**, all stage one. The shared parent averaged
10,416.48 and the longer-lookback arm 10,399.69 on the same games. This
control improved **38** paired seeds over the parent, worsened **67**, and
tied **23**; nine games scored below 9,000 versus the parent's two. It is
not a confirmed successor. Its [fresh best replay](fresh-selected-replay/replay.html)
re-executes **2,559 learned actions** with independent verification and
does not complete the mission.

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
