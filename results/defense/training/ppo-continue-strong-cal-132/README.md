# Stronger learned-continuation calibration

The confirmed run-121 policy almost never selected its learned
`CONTINUE_PREVIOUS` action near the repeated first-stage loss. Run 122
initialized that extra actor row with a direction-neutral +7 bias; it was
used overall but only 0–2 times in each 64-action pre-flash window after
training, and did not pass stage one. The three subsequent unbiased
frontier searches changed no model weights and also failed to discover a
higher-score path. This run returns to an actual learned-policy test.

The exact same ordinary-action run-119 parent, score-only PPO, four raw
screen frames, per-life training-only actor-bias noise, own-visible-loss
rewind, four boot-only workers, and complete-game evaluation settings are
used as in run 122. The only intended algorithmic change is the new
`CONTINUE_PREVIOUS` row's neutral initial bias **+9 instead of +7**;
all twenty existing action rows, encoder and value head transfer exactly.
No direction or route is favored, and continuation merely repeats the
policy's own last physical key. No hand-controlled demonstrations,
hidden-state input, action override or reward shaping are used.

A read-only calculation on the confirmed run-121 replay's four true
64-action pre-flash windows predicts 9.37–13.16 continuation choices per
window for +9, versus 3.35–5.00 at +7 **before training**. These are
probability sums on another policy's recorded visible screens, not actual
new-game samples or evidence of better play. The calculation used the
unchanged run-119 model plus the documented mean-row initializer. It did
not select movement actions or enter training data.

First, save an exact no-learning initializer and evaluate 32 complete
fresh games at seeds 604000–604031, preserving a separately verified
best-effort replay. Then train **1,048,576 new base actions**, with eight
fixed ten-game checks every 131,072 actions. Select the highest fixed
ten-game mean, earliest exact tie. Any stage-two screen or mission from
training or validation must be independently re-executed from the saved
neural policy. Otherwise, if selected fixed mean is at least **10,450**,
compare the frozen selected model on 128 new matched complete games with
the run-121 score parent and run-122 +7 predecessor. A score tie or
unverified exploratory branch cannot replace the protected global best.
Independently count sampled continuation choices in the selected model's
native-verified best-effort replay, especially the pre-flash 64-action
windows. If continuation remains sparse there or the fresh score/stage
gate fails, archive and stop this calibration rather than simply run it
longer unchanged. The collector and global best remain independently
available throughout.

The no-learning [initializer](initializer/) was saved with complete
model/optimizer/RNG state. Its 32 fresh complete games averaged **10,454.38**
points (median 10,460; best 10,480), all stage one. The separately
[verified replay](initializer-replay/replay.html) reproduces **2,539**
physical actions, screens and rewards from the frozen 21-way policy. A
read-only [choice-use check](initializer/continuation-use.json) reexecuted
that same game exactly and counted **691** sampled `CONTINUE` choices,
including **18 / 10 / 9 / 7** in the four pre-flash 64-action windows.
It still lost each life at 2,620 displayed points. The calibration is
therefore effective at making the option available without destroying
baseline score, but is not gameplay progress. No policy update occurred
before the bounded training run.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-continue-strong-cal-132-init \
  --artifacts runs/defense-ppo-continue-strong-cal-132-init/artifacts \
  --initialize-repeat-policy results/defense/training/ppo-persistent-noise-long-119/milestone-000009718528 \
  --repeat-previous-action --continue-initial-bias-offset 9 \
  --initialize-only --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --learning-rate .00025 --entropy .002 --policy-bias-noise 1 \
  --gamma .997 --gae-lambda .95 --life-terminal \
  --curriculum-probability 1 --curriculum-lookback 128 \
  --curriculum-trigger life-loss --curriculum-restored-life-only \
  --curriculum-share --curriculum-boot-envs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 \
  --eval-seed 10000 --mlx-cache-mb 512

venv/bin/python -m rl.defense_evaluate \
  runs/defense-ppo-continue-strong-cal-132-init/latest/model.safetensors \
  --games 32 --seed 604000 --envs 8 \
  --output runs/defense-ppo-continue-strong-cal-132-init/fresh-evaluation.json \
  --replay-output runs/defense-ppo-continue-strong-cal-132-init/fresh-replay

venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-continue-strong-cal-132 \
  --artifacts runs/defense-ppo-continue-strong-cal-132/artifacts \
  --resume runs/defense-ppo-continue-strong-cal-132-init/latest \
  --steps 1048576
```

The bounded [full run](run/) completed all **1,048,576** score-only
training actions, with **115** complete boot games, **3,984** restored
segments, eight fixed ten-game validations and no observed stage-two or
mission event. Fixed means in order were **10,291 / 10,446 / 10,160 /
9,908 / 10,156 / 9,924 / 10,456 / 10,158**. The predeclared rule selected
the seventh full model/optimizer/RNG checkpoint at 917,504 actions.
All eight checkpoints, final state, log, local artifacts, three fresh
evaluation records and a natively verified selected replay are preserved.

The [128-game matched fresh comparison](comparison.json) on seeds
604100–604227 found selected / confirmed run-121 parent / +7 run-122
predecessor means **10,319.69 / 10,476.56 / 10,472.81**. The selected
model was **−156.88** versus its score parent, with 2 paired wins,
118 losses and 8 ties; it was **−153.13** versus +7. All **384** fresh
games ended in stage one. Only 9/128 selected games reached the usual
10,480 ceiling, versus 108/128 and 106/128 for the comparators. The
selected model is therefore a clear score regression, not a promoted
successor. The protected global best remains unchanged.

The selected model's [fresh replay](run/fresh-selected-replay/replay.html)
independently reproduces **2,586** neural physical actions. A separate
[exact choice-use check](selected-continuation-use.json) counted **766**
sampled `CONTINUE` choices overall, yet only **4 / 3 / 1 / 5** in the four
pre-flash 64-action loss approaches; the four lives each scored 2,620.
Thus a stronger neutral initial bias genuinely increased continuation
use, but PPO again reduced it near the recurring obstacle and did not
escape. More unchanged training from this model is not justified by the
predeclared gate. The result does not rule out a different learned
temporal-control architecture or score-only exploration method.
