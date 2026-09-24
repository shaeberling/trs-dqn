# Direction-neutral duration exploration without changing evaluation play

The [frozen duration audit](../ppo-duration-long-option-147/frozen-parent-duration-mass.json)
shows the confirmed fine-cadence parent assigns about 98–99% of its
joint-action mass to one-step holds on a selected pre-loss life.
Simply appending a 128-step option sampled it only 1–4 times per
16,384 training actions, and a stronger new-row prior harmed complete
games. This trial tests a different mechanism: make longer options
genuinely available *during training* without changing the network's
unperturbed evaluation distribution or favoring a physical direction.

Start both arms from the same [run-146 score parent](../ppo-duration-fine-cadence-146/full/step-000001441792/)
with five durations **1 / 4 / 16 / 64 / 128**, the verified transfer
that copies every old parameter and appends the 64-hold physical-key
rows with the cautious **−2** logit offset. Both use the same fresh
Adam, policy RNG seed, original screen-only 50,000-T-state/stride-2
environment, displayed-score reward, option-start semi-Markov credit,
16 workers (four boot-only), training-only factorized key/duration
noise and own visible-life-loss archive. No diagnostic intervention
trajectory enters training.

The treatment samples each actual option start from
`(1−ε) π(key,duration | screen) + ε π(key | screen)/5`, with **ε = 0.08**.
The physical-key marginal is exactly the existing actor's (including
its documented training-only noise); only duration gets a uniform
8% mixture component. Its actual mixed action log-probability is
stored for PPO, and the *same mixed distribution* is used in the
clipped actor ratio and entropy, avoiding an unrecorded action
override. Forced option-continuation steps still have no actor loss.
The control sets ε = 0, otherwise matching the treatment. Native
complete-game evaluation and replay use the ordinary unperturbed
saved neural actor for **both** arms: the exploration floor never
acts in evaluation or the protected global replay.

First run separate **16,384-action integration smokes**. Require
finite updates, full model/optimizer/RNG save and resume, ten complete
fixed games on seeds 10000–10009, and independently verified native
learned replay. The treatment must start at least **50** 128-step
options in the smoke; otherwise the proposed exposure mechanism has
not occurred and the production trial is cancelled. Smoke scores
cannot select a policy. Once the integration gate passes, run both
arms from the frozen common source for **524,288** new base actions,
with ten complete fixed games at each 131,072-action checkpoint.
Stop an arm early only after two consecutive fixed means below 5,000,
unless a later stage is independently verified. Select each arm's
earliest highest-stage checkpoint, breaking ties by fixed-game mean.
Any stage-two or mission report requires original-boot replay and
fresh complete-game confirmation before global promotion.

If all remain stage one, compare frozen selected treatment, control
and run-146 parent on **64 new matched complete games**, seeds
**608700–608763**, at the same 50,000/2 timing. A score-only parent
requires treatment mean above both alternatives, no worse sub-9,000
tail, and a separately predeclared fresh confirmation; it never
replaces the protected best replay. A negative outcome is archived
with the same full-state and replay fidelity. No hidden course row,
wall parser, inferred collision, demonstration, non-score reward or
hand-coded direction is used by the learner.

## Passed integration gate

Both [treatment](treatment-smoke/) and [control](control-smoke/)
16,384-action smokes finished with finite PPO updates, full model/
optimizer/RNG checkpoints, ten complete fixed games and originally
verified learned-policy replays (5,094 / 5,047 neural actions).
The treatment started **52** 128-step options and executed **6,241**
base actions within them, meeting the predeclared ≥50 exposure gate;
the control started only **one**. Their smoke means were **9,972 /
10,470**, both stage one. These are integration checks, not policy
selection. The control's final model and optimizer bytes exactly
match the earlier no-mixture −2 smoke from run 147, establishing
default-path parity under the new implementation.

Separate [treatment](treatment-resume-smoke/) and
[control](control-resume-smoke/) resume checks restored each full
checkpoint and trained another 16,384 actions after boot-only emulator
restart. Their fixed means were **10,198 / 10,470**, both stage one;
each local 10,480 replay independently verified (5,082 / 5,031
actions). Resume smokes are also excluded from production selection.
The planned matched production stage gate is now authorized by its
specified integration conditions. Its two arms must start afresh from
the frozen run-146 checkpoint, not resume either smoke.
The full repository regression suite passed **507 tests** with this
implementation before production.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-mixture-148-full \
  --artifacts runs/defense-duration-mixture-148-full/artifacts \
  --initialize-duration-checkpoint results/defense/training/ppo-duration-fine-cadence-146/full/step-000001441792 \
  --extend-longest-duration --appended-longest-logit-offset -2 \
  --learned-durations 1 4 16 64 128 --duration-explore-mix 0.08 \
  --steps 524288 --seed 41 --envs 16 --rollout 512 --batch-size 512 \
  --epochs 4 --learning-rate 2.5e-5 --entropy 0.002 \
  --gamma 0.9984988733093293 --gae-lambda 0.9746794344808963 \
  --reward-scale 0.01 --life-terminal --option-actor-gae \
  --duration-initial-logit-spacing 4 --tstates 50000 --observation-stride 2 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --policy-key-noise 2 --policy-duration-noise 4 \
  --curriculum-probability 1 --curriculum-cells screen \
  --curriculum-bins 128 --curriculum-per-bin 1 --curriculum-share \
  --curriculum-boot-envs 4 --curriculum-trigger life-loss \
  --curriculum-lookback 256 --curriculum-restored-life-only
```

For the production control, use a distinct `...-control-full` run and
artifacts path with `--duration-explore-mix 0`; keep every other flag
and the frozen initializer identical.

## Treatment stage gate: stopped on predeclared collapse rule

The [production treatment](treatment-full/) did expose long options
in real training (**1,154** 128-step starts across 319,488 actions),
but its first and second ten-game fixed-check means were **3,616** and
**2,305**, both entirely stage one. Two consecutive means below
5,000 triggered the planned graceful stop. The learner exited normally
after finishing its in-flight rollout at **319,488** total actions,
with full model/optimizer/policy-RNG state, both fixed checkpoints,
metrics and any verified local replay retained. No mission or stage-two
result was observed. The control is evaluated separately from the
same initial policy. The stopped treatment remains in the predeclared
fresh-score comparison, but cannot be promoted without outperforming
both alternatives on score and low-tail safety; fixed results alone
do not make it a candidate.

The [matched control](control-full/) completed all **524,288** actions
normally. Its four fixed ten-game means were **10,472 / 9,956 /
10,416 / 10,446**, every game stage one. By the frozen selection rule,
the earliest highest fixed mean is checkpoint **131,072** for both
treatment (3,616) and control (10,472). No checkpoint was selected
from the new-seed comparison.

## Frozen fresh comparison and disposition

All three frozen policies played **64 complete games on the same new
seeds 608700–608763** at 50,000 T states and stride 2:

| Arm | Mean / median / best | Below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Mixture treatment](fresh-treatment-64.json) | **5,177.97 / 5,390 / 10,410** | **61** | 0 |
| [No-mixture control](fresh-control-64.json) | **10,067.50 / 10,480 / 10,480** | **10** | 0 |
| [Unchanged run-146 parent](fresh-parent-64.json) | **10,420.63 / 10,480 / 10,480** | **1** | 0 |

Treatment versus control had **2 paired wins / 62 losses / 0 ties**;
versus its parent it had **0 wins / 64 losses / 0 ties**. The control
also trailed the parent (6 wins / 24 losses / 34 ties). No arm met the
predeclared score-parent criterion, so a second fresh confirmation
would not change the decision and is not run. The treatment's
[5,117-action best fresh replay](fresh-treatment-replay/replay.html)
and control's [5,013-action replay](fresh-control-replay/replay.html)
were independently reproduced from original boot. A strictly
post-hoc [course-pointer audit](../../diagnostics/duration-mixture-148-course-progress/README.md)
finds their visible losses around original stage-one stream rows
**32–34 of 126**, not later passage. Neither this hidden pointer nor
any diagnostic trajectory entered learning.

Conclusion: the mixture achieved actual, correctly credited
long-duration exposure but severely damaged screen-only score play,
and even the no-mixture fresh-optimizer control underperformed the
confirmed parent. Retain all full states and replays as negative
evidence; do not promote either checkpoint or repeat this unchanged
mixture. The original game, run-146 score parent and protected global
best model/replay remain unchanged. The active gameplay goal remains
unmet.
