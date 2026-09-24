# Restore a long physical hold at fine decision cadence

The trained 50,000-T-state, stride-2 key-duration policy in
[run 146](../ppo-duration-fine-cadence-146/README.md) improves score
slightly but still loses all four ships at original stage-one stream rows
33–34. Its longest learned hold is **64 × 50,000 = 3.2 million T states**,
half the source policy's former 64 × 100,000 = 6.4 million. The verified
fine-cadence replay still shows the ship around character columns 24–27
late in the approach, while the later visible opening begins around 51.
This does not prove a held direction solves the obstacle; it motivates
testing whether the *learned* actor benefits from a longer physical
commitment while retaining fine short decisions.

Freeze [run 146's independently confirmed score-training parent](../ppo-duration-fine-cadence-146/full/step-000001441792/).
Transfer every screen encoder, critic and existing actor parameter
unchanged to a new five-duration actor with holds **1 / 4 / 16 / 64 /
128**. Initialize each new 128-hold physical-key row by copying the
corresponding old 64-hold row with a uniform **+2-logit** offset in
production, conditional on the revised integration gate below.
The extension has no preferred key, gap, wall,
route or steering rule. The old 80 logits and all non-actor outputs
remain bit-identical at initialization; normal softmax renormalization
means the new actor is not exactly policy-identical. Use fresh Adam
and policy RNG because the actor shape changed. A matched control
copies the same 80-row policy with the same fresh optimizer/RNG and
retains holds 1/4/16/64. Both keep the source's 50,000/2 screen timing,
score-only reward, option-start semi-Markov credit, 16 workers (four
full-boot), screen-diverse own-life-loss curriculum, training-only
direction-neutral key/duration perturbations and fixed complete-game
evaluation. No diagnostic branch is a training example.

First run a 16,384-action integration smoke for each arm, checking
finite updates, complete-game evaluation, original-boot learned replay
verification, full model/optimizer/RNG save and exact resume. These
smokes cannot select a policy. Then start separate production runs
from the same frozen source and fixed random seed, evaluating ten
complete games on seeds 10000–10009 every **131,072** new base actions
through **524,288** actions. Stop either arm only if two consecutive
fixed means fall below 5,000. Select each arm's earliest highest-stage
checkpoint, breaking ties by fixed-game mean. Any stage-two or mission
event requires native reexecution from boot and fresh complete-game
confirmation before global promotion.

If both arms remain stage one, compare their frozen selections and the
unchanged run-146 source on **64 new matched complete games**, seeds
**608600–608663**, all at 50,000/2. A stage-one score gain is eligible
only as a score-training parent if it has a higher mean than both
control and source with no worse sub-9,000 tail, then survives a second
independent 64-seed confirmation fixed before evaluation. It cannot
replace the protected best replay or count as obstacle passage.
Record actual 128-hold starts/executed actions to check that the new
option was truly explored. Keep full checkpoints, logs and verified
replays. Do not extend the same variant merely because it again attains
the 10,480-point stage-one ceiling.

## Integration finding and prospective prior adjustment

The first 16,384-action long-option smoke used the initially specified
−2-logit prior. Its ten fixed complete games averaged **10,470** and its
5,047-action learned replay passed independent native verification, but
only **one** of its 12,760 option starts selected a 128 hold. That is
not enough exposure for a meaningful long-hold training test. Before
starting any production arm, set the appended bias offset to **zero**:
the new 128-row logits initially equal the matching 64-row logits for
every physical key. This changes no old row and favors no direction.
Keep the −2 pilot as a separate, excluded integration record. The
zero-offset repeat also finished with finite updates, a **10,214**
ten-game fixed mean, verified 5,089-action replay, but only **two**
128-hold starts out of 12,374. Thus the intended ≥5 exposure gate
failed; neither smoke qualifies as evidence about useful long holds.
Before production, redesign the direction-neutral prior to a **+2**
offset (about 7.4 times the matching 64-row unnormalized mass). Run
one new 16,384-action smoke and require at least **five** actual
128-hold starts, finite updates and a verified learned replay. If it
fails, do not run the planned production comparison. The score from
all smokes is explicitly excluded from model selection and offset
choice; the offset revision responds only to the observed exposure
count. The matched production control remains unchanged.

## Stopped integration result

The +2 smoke also finished normally with finite PPO updates and a
native-verified 5,057-action learned replay. Its ten fixed complete
games averaged **9,220**, all stage one, but it started only **four**
128-step holds in 16,384 training actions (508 actually executed base
actions). The predeclared ≥5 exposure gate therefore failed. There is
no matched production control, 524,288-action continuation, fresh-score
claim or replay promotion. The earlier −2 and zero-offset smokes
started only **one** and **two** 128-step holds, respectively. All
[full smoke states](smoke-minus2/), [zero-offset states](smoke-zero/)
and [+2 states](smoke-plus2/) retain weights, optimizer, policy RNG,
logs, complete-game evaluations and independently verified local
replays. None is a replacement for the protected best.

A separate [+2 optimizer-resume integration check](resume-smoke/)
restored the full 16,384-action checkpoint, restarted episodes from
boot and completed another 16,384 actions without nonfinite updates.
It sampled eight 128-step holds in that resumed segment, but its
fixed ten-game mean fell further to **6,012**, with all games still
stage one. That later result cannot rescue the failed smoke gate; it
reinforces the decision not to continue this treatment. Its own
10,480-point local replay also passed original-boot verification.

The reproducible [frozen duration-mass audit](frozen-parent-duration-mass.json)
on the selected source's verified first-life screen frames finds mean
probability on 64-step options of about **0.0012%** before the white
alignment marker and **0.0010%** in its final 64-action window; the
one-step option holds roughly **98.35% / 99.25%** in those windows.
This uses hypothetical logits every fourth visible frame, including
forced-continuation frames, not actual option-start draws or an unbiased
population estimate. It explains why simply adding a longer row to
this strongly one-step-preferring actor barely exposes the treatment.
Future work should address long-horizon, screen-conditioned exploration
without letting a direction-neutral long-hold prior destroy existing
score. No hidden course pointer, diagnostic branch or non-score reward
entered training.
The full repository regression suite passed **505 tests** after this
implementation; the five focused transfer/probe tests also passed
against the final source files.

Reproduce the diagnostic in a new output path:

```sh
venv/bin/python -m rl.defense_duration_mass_probe \
  results/defense/training/ppo-duration-fine-cadence-146/full/step-000001441792/model.safetensors \
  results/defense/training/ppo-duration-fine-cadence-146/fresh-selected-replay \
  --output runs/defense-duration-mass-147-reproduction.json
```
