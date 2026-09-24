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
