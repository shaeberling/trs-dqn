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
