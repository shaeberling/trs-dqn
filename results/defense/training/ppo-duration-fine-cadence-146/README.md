# Finer control timing with matched visible history

The selected strong key-duration policy, its spatial residual and the
matched control still place the visible ship around character columns
18–28 shortly before the repeated loss, rather than near the later
right-side opening around column 51. The [selected-screen comparison](../../diagnostics/spatial-143-loss-sheets/README.md)
is diagnostic, not a route. Exact-prefix trials show holding RIGHT too
early loses at the preceding opening while starting later remains short.
This raises a control-rate hypothesis: the standard 100,000-T-state
base decision may be too coarse to switch options between the openings.

An earlier [frozen cadence trial](../../diagnostics/cadence-138/README.md)
changed 100,000 to 50,000 T states **without** restoring the four-frame
history span and regressed sharply. A much older ordinary PPO checkpoint
recovered most of its frozen score with 50,000-T-state decisions and
observation stride 2, and separate matched-history training still did
not pass stage one. Neither test combined the current strong learned
key-duration controller with fine cadence and matched history.

First freeze the [option-credit checkpoint](../ppo-duration-credit-136/run/step-000001179648/)
and run three 32-game **evaluation-only** arms on identical untouched
complete-game seeds 608200–608231:

| Arm | T states per base decision | Four-frame observation stride |
| --- | ---: | ---: |
| Native source | 100,000 | 1 |
| Fine, matched history | 50,000 | 2 |
| Fine, narrow history | 50,000 | 1 |

The 50,000/2 arm retains the same nominal **300,000-T-state** temporal
span across the four observed frames as the source. All arms use the
same frozen learned weights and original game; no override is eligible
for the protected best replay. Record complete-game mean, median, best,
stage and paired differences. A later-stage probe would require exact
same-timing native replay before it can inform training. No stage-one
score alone counts as passage.

If the 50,000/2 frozen arm retains at least **9,000** mean points and
does not enter a new stage, proceed to a bounded score-only PPO
adaptation from the source's **full model/Adam/RNG state**, not a
diagnostic trajectory. Keep its four raw visible frames, 20 learned
physical keys × holds 1/4/16/64, semi-Markov option-start actor credit,
16 workers (four boot-only), own-screen archive and displayed-score
reward. At half base cadence set observation stride 2, rollout
256→512 and own-life-loss lookback 128→256 to preserve nominal
physical horizons. Set gamma to sqrt(.997) and GAE lambda to sqrt(.95)
to preserve nominal per-physical-time discount and trace decay; this
does not make the changed policy MDP identical. Initial action holds
last half as long in physical time by design, providing finer control.
Run **131,072** new actions for an integration gate with full
checkpoint and ten complete fixed games, then consider a separate
524,288-action stage gate only if that short continuation avoids
collapse (fixed mean at least 5,000) or independently verifies stage
two. A lower score with no stage passage is archived rather than
silently extended. Any stage-two/mission outcome must be reproduced
from original boot using the frozen learned policy at its saved timing,
then confirmed on fresh complete games. No diagnostic action,
wall coordinate, hidden pointer, demonstration or extra reward is
provided to the policy or trainer.

This is a test of **retrained finer control**, not a claim that a frozen
timing override or higher stage-one score solves the obstacle.

The predeclared frozen 32-game probes have completed:

| Frozen arm | Mean / median / best | Scores below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Native 100,000/1](frozen-native-32.json) | **10,480 / 10,480 / 10,480** | 0 | 0 |
| [Fine 50,000/2](frozen-fine-matched-32.json) | **10,477.19 / 10,480 / 10,480** | 0 | 0 |
| [Fine 50,000/1](frozen-fine-narrow-32.json) | **7,942.81 / 8,035 / 10,440** | 30 | 0 |

The matched-history fine arm beats the narrow-history fine arm on all
**32 paired seeds**. It trails native timing on just three seeds (29
ties), by 2.81 mean points; its minimum is 10,430. All **96 games**
remain stage one. This is strong evidence that history compression,
not merely more frequent action calls, caused the earlier frozen-score
regression. It does **not** prove fine cadence will learn passage.
The predeclared ≥9,000 condition is met, so the bounded score-only
training adaptation is warranted. These diagnostic override records
are evaluation-only and cannot promote a best replay.

## Trained short gate and full-stage continuation

The [bounded 131,072-action adaptation](gate/) resumed the source's
full model/Adam/policy RNG at absolute counter 1,179,648 with the planned
50,000/2 timing, doubled 512-action rollout and 256-action own-loss
rewind, and square-root-adjusted score discount/GAE lambda. It stopped
normally at **1,310,720** actions. Its ten noise-free fixed complete
games averaged **10,476**, median 10,480 and best 10,480, all stage one.
The local [10,480-point replay](gate/artifacts/best/replay.html) was
independently reproduced from original boot for all **5,069** selected
neural actions at 50,000 T states and observation stride 2. Full weights,
optimizer, RNG and training log are archived; neither this replay nor
the frozen timing overrides replace the protected global best.

The ≥5,000 no-collapse gate is met. Before extending, freeze this exact
checkpoint and continue for **393,216** more base actions to absolute
counter **1,703,936**, making **524,288 new fine-cadence actions** in
all. Preserve full model/optimizer/RNG milestones at **1,441,792 /
1,572,864 / 1,703,936**, with ten complete fixed games on seeds
10000–10009 every 131,072 actions. A standard resume restarts emulator
episodes from boot and refills the own-state archive; it restores learner
weights, optimizer and policy RNG. Stop early only if two consecutive
fixed means fall below 5,000; archive any collapse. Select the earliest
highest-stage checkpoint, breaking ties by fixed-game mean across the
short gate and full continuation. Stage two or a mission requires exact
native reexecution from boot at 50,000/2, then fresh complete-game
confirmation before global promotion.

If all remain stage one, only a selected fixed mean at least **10,450**
triggers **64 new matched complete games** on seeds 608400–608463.
Compare the frozen selected trained checkpoint against the unmodified
option-credit source **at the same 50,000/2 timing**, and report the
source's native 100,000/1 result as context if measured. A score-only
improvement must have no worse sub-9,000 tail and survive a separate
fresh confirmation before becoming a *score-training* parent. It still
cannot count as stage passage or replace the protected best replay.
Keep all source/selection hashes and full-game records. No wall/gap
parser, hidden course index, hand-coded steering, demonstration or
reward other than displayed score enters learning.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-fine-cadence-146-full \
  --artifacts runs/defense-duration-fine-cadence-146-full/artifacts \
  --resume results/defense/training/ppo-duration-fine-cadence-146/gate/step-000001310720 \
  --steps 1703936 --eval-every 131072 --eval-games 10 --eval-envs 10
```

## Completed full gate and fresh comparison

The [full continuation](full/) stopped normally at absolute action
**1,703,936**. Its three ten-game fixed-check means at 1,441,792,
1,572,864 and 1,703,936 were **10,480 / 10,470 / 10,458**;
all 30 complete games remained stage one. Including the short gate,
all **40** fixed games remain stage one. The earliest best fixed
checkpoint is [1,441,792](full/step-000001441792/), selected before
looking at fresh games.

On fresh complete-game seeds **608400–608463** at the same 50,000/2
timing, [selected trained policy](fresh-selected-64.json) versus
[frozen option-credit parent](fresh-source-fine-64.json) averaged
**10,413.44 versus 10,391.56**. Paired seeds: **9 wins / 6 losses /
49 ties**; games below 9,000: **1 versus 2**; both arms remained
stage one for all 64 games. The selected [learned replay](fresh-selected-replay/replay.html)
was independently reproduced from boot for all **5,014** neural
actions. A forensic-only original-course pointer audit places its four
visible losses at **33 / 33 / 33 / 34 of 126** stream rows. This hidden
pointer is not an input or training signal and does not identify exact
collision positions. The result is a small score-only gain, not a
solution to the recurring obstacle or a global-best promotion.

The predeclared no-worse-below-9,000 condition is met, so freeze both
checkpoints and run a **separate confirmation** on untouched complete
seeds **608500–608563**, again 64 matched games at 50,000/2 with the
same learned action sampling. Count score wins/losses/ties, means,
sub-9,000 tails and reached stages. A score-parent promotion requires
the trained arm to have a higher confirmation mean and no worse
sub-9,000 tail. If either fails, archive the result as inconclusive or
negative and retain the old score parent. Stage-one score alone never
changes the protected best replay or establishes passage.

The separate confirmation is complete: [selected trained](confirm-selected-64.json)
mean **10,462.03**, [frozen parent](confirm-source-fine-64.json) mean
**10,436.56**, a **+25.47** mean difference. Paired seeds: **4 wins /
5 losses / 55 ties**; below 9,000: **0 versus 1**; all **128**
confirmation games remain stage one. Across both untouched sets (128
games per arm), means are **10,437.73 versus 10,414.06**, with
**1 versus 3** sub-9,000 outcomes. The predeclared *score-training-parent*
criterion is met, so the fine-cadence checkpoint at 1,441,792 can
seed a later score-only experiment at its saved timing. The improvement
is modest and dominated by ties; it provides no evidence of passage
through the row-33/34 obstacle. The older global-best model and replay
remain protected and unchanged.
