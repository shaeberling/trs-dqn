# Calibrated exploration of the learned continue action

Run 121 confirmed a +26.48-point **score** gain over its ordinary-action
parent on 256 matched fresh games per policy, but all games remained stage
one. A read-only re-execution of its verified best replay found just one
`CONTINUE_PREVIOUS` choice in 2,551 decisions, none near the recurring
losses. The added action had effectively not been explored.

The initialization row was the mean of the twenty original actor rows.
On the new model's own verified stage-one screens, its frozen pre-training
probability of selecting `CONTINUE_PREVIOUS` averaged only **0.0717%**;
the trained checkpoint averaged **0.0116%** on the same screens. A
direction-neutral **+7 logit-bias offset** to only the extra row of a new
own-policy initialization predicts about **12.76%** choice probability over
those screens and **6.80%** in the four 64-action pre-loss windows. This
setting was chosen from screen-only diagnostics to make the option usable,
not to select a movement direction or route. It is an initialization/
exploration hyperparameter, never an action override or extra reward.

The no-learning calibrated checkpoint was checked on ten complete native
games at seeds 602900–602909: mean **10,220**, median **10,480**, best
**10,480**, all stage one. In its native-verified best replay, re-execution
matched all screens, physical actions and rewards while counting **320**
sampled `CONTINUE_PREVIOUS` choices in 2,531 actions (**12.64%**), across
all four lives. Thus this calibration actually tests the proposed learned
action, although its initial score is lower. The smoke checkpoint and
evaluation stay separate from production selection and global best.

Run 122 starts **again from the unchanged run-119 ordinary-action parent**,
not from run 121's largely unused extra row. It copies the learned screen
encoder, value head and twenty existing actor rows exactly. Only the neutral
mean-row extra choice receives offset +7; optimizer, action/noise RNGs,
episodes and training experience start fresh. All other settings match
run 121: score-only PPO, raw-screen observations, per-life training-only
actor-bias exploration, own-visible-loss rewind and four boot-only workers.
Every base step is selected by the 21-way learned policy; `CONTINUE` merely
reuses its own prior key. No direction, obstacle, hidden game state, route
or stage target enters the action rule.

The first gate is **1,048,576 new base actions**, with eight complete
ten-game unperturbed validations at 131,072-action intervals. Select the
highest fixed mean checkpoint, earliest exact tie. Any stage-two or native
mission game must be independently reproduced. Otherwise, if a selected
fixed mean reaches at least **10,450**, compare it on 128 untouched matched
complete games with both the run-119 ordinary-action parent and the
run-121 score parent; only independent gains justify continuation or a
score-parent claim. If it fails that score gate, archive the full negative
run, including action-use diagnostics. An equal 10,480-point stage-one replay
cannot replace the protected global best.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-continue-calibrated-122 \
  --artifacts runs/defense-ppo-continue-calibrated-122/artifacts \
  --initialize-repeat-policy results/defense/training/ppo-persistent-noise-long-119/milestone-000009718528 \
  --repeat-previous-action --continue-initial-bias-offset 7 \
  --steps 1048576 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --learning-rate .00025 --entropy .002 --policy-bias-noise 1 \
  --gamma .997 --gae-lambda .95 --life-terminal \
  --curriculum-probability 1 --curriculum-lookback 128 \
  --curriculum-trigger life-loss --curriculum-restored-life-only \
  --curriculum-share --curriculum-boot-envs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 \
  --eval-seed 10000 --mlx-cache-mb 512
```

The full bounded run completed **1,048,576** own-screen/score actions,
**116** complete boot training games and **3,915** own-restored segments.
Its eight fixed ten-game means were **10,126 / 10,406 / 10,406 / 10,416 /
10,166 / 10,407 / 10,243 / 10,456**. The final checkpoint was selected
under the predeclared rule and passed the 10,450 gate, but no training or
fixed-validation game entered stage two. Its full model/optimizer/RNG,
every checkpoint, metrics, source snapshots and local artifacts are
preserved in [run/](run/).

The planned [128-game matched fresh check](comparison.json), seeds
603000–603127, averaged **10,450.94** for the calibrated selected model,
**10,456.80** for the stronger run-121 score parent, and **10,453.13** for
the run-119 ordinary-action parent. The selected model is −5.86 against
the score parent and −2.19 against the ordinary parent; it won/lost/tied
**20/23/85** and **55/14/59** paired games respectively. All **384** fresh
games remained in stage one, with no native mission. The selected model's
[fresh 10,480-point replay](fresh-selected-replay/replay.html) was independently
verified for 2,576 learned decisions, but it cannot outrank the protected
global best.

The [visible-loss and action-use diagnostic](../../diagnostics/continue-calibrated-122-losses/README.md)
finds four more 2,620-point lives at the right-opening barrier. Crucially,
the trained policy really did choose `CONTINUE_PREVIOUS` **132 times** in
that verified game, but only **2, 2, 2 and 0** times in the four last-64-
action pre-loss windows. This is a genuine negative test of the calibrated
learned-continuation hypothesis: the option was available and used, yet it
did not change the repeated outcome. The run-121 score parent remains the
stronger confirmed model; more identical training is not justified by this
result. The full **468-test** suite passed before the production run.
