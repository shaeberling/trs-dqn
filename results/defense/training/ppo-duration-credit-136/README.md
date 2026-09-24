# Credit the complete learned hold, not just its early steps

The joint key-duration policy repeatedly reaches a high stage-one score
but loses at decoded stream row 33–34. Its [screen-diverse continuation](../ppo-duration-screen-135/README.md)
filled all 128 screen cells per worker and still did not pass the obstacle;
perfect reused-seed checks regressed on 64 fresh games. A specific
learning-rule limitation remains: the current PPO actor advantage is
ordinary **base-step** GAE sampled only at a learned option's start.
With gamma .997 and lambda .95, a TD event 63 actions later receives
roughly `(.997 * .95) ** 63`, about **3.2%**, of its original weight at
the start of a 64-action hold. Thus a late collision or delayed score
may barely update the option that caused it.

Add an **opt-in semi-Markov actor GAE**. At each actual option start, sum
every displayed-score reward during the complete hold with ordinary
per-base-action gamma, then bootstrap from the next option start (or
zero at a visible learning boundary). Recursively carry advantage with
one lambda factor **per completed option**, not per forced base action.
If a hold extends beyond a rollout, omit that unfinished start from actor
updates rather than pretending it finished; all its real base transitions
still train the same critic. The actor still samples from four raw screen
frames and the only reward is the displayed score. The executor, action
choices, evaluation and native replay semantics do not change. This is
an attribution change, not an obstacle hint, extra reward or demonstration.

Use the **same frozen full 917,504-action joint checkpoint** as the
screen-diverse [control run 135](../ppo-duration-screen-135/README.md),
the same 128-cell visible-screen life-loss archive, 16 workers (four
boot-only), symmetric key noise 2 and duration noise 4, learning rate
2.5e-5, gamma .997 and GAE lambda .95. The sole planned training change
is `--option-actor-gae`. A short optimizer smoke may check finite losses
and exact replay but cannot select or tune the production model.

Train **1,048,576 additional base actions** to absolute counter
**1,966,080**, with eight complete ten-game validations every 131,072
actions on seeds 10000–10009. Retain all full optimizer/RNG checkpoints.
Select earliest highest stage rank, breaking ties by fixed-game mean.
Any stage-two result requires independent native replay and fresh
confirmation. If all remain stage one, compare the frozen selection,
the control's selected 1,310,720-action checkpoint, and the shared input
joint checkpoint on **64 new matched complete games, seeds 607200–607263**.
Include the established ordinary-action score parent on those same seeds
as an informational fourth arm, without changing checkpoint selection or
the stage-pass criterion; a win over a weakened duration control alone
would otherwise be misleading.
Score-only improvement cannot replace the protected best replay. If two
consecutive fixed checks average below 5,000, archive and stop early.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-credit-136 \
  --artifacts runs/defense-duration-credit-136/artifacts \
  --resume results/defense/training/ppo-duration-joint-134/run/step-000000917504 \
  --curriculum-cells screen --curriculum-bins 128 --curriculum-per-bin 1 \
  --option-actor-gae --steps 1966080 --eval-every 131072 \
  --eval-games 10 --eval-envs 10
```

A separate 16,384-action optimizer smoke from the same frozen source
finished with finite updates, four complete fixed-seed games at 10,480
mean (all stage one), and a native-verified **2,553-action** local replay.
It only establishes that option-credit updates, exact replay and resume
semantics run end to end; the smoke checkpoint is excluded from production
selection and from any claim of obstacle passage.
The full **494-test** repository regression suite passed before production.

The full planned run finished without stage-two play. On its first 64
untouched games, the frozen option-credit selection averaged **10,412.03**
versus **10,399.38** for the screen-diverse control, **10,277.19** for
their shared joint input and **10,376.56** for the ordinary-action score
parent. All **256** games stayed stage one. Only three paired games improved
over the control, seven worsened and 54 tied: the +12.66 mean is weak
evidence for the credit change. Against the ordinary parent, however, it
improved 29, worsened seven and tied 28, with a +35.47 mean. This is
another small one-set score edge like the one that failed confirmation
in run 134.

Before any further evaluation or model updates, freeze the selected
option-credit checkpoint, frozen control checkpoint and ordinary parent;
compare all three on **128 additional untouched matched complete games,
seeds 607400–607527**. Do not reselect checkpoints from the result.
Only a positive combined **192-game** mean against the ordinary parent
with no worse sub-9,000 tail can qualify the option-credit model as a
*score-training* parent. A stage-one-only score gain still cannot replace
the protected best replay or count as barrier passage. Archive a negative
confirmation equally.

The completed run's eight fixed ten-game means were **10,478 / 10,480 /
10,214 / 10,460 / 10,470 / 10,468 / 10,480 / 10,478**, all stage one.
The **earliest** 10,480-point checkpoint at absolute counter **1,179,648**
was selected, not the final model. All [eight full optimizer/RNG milestones](run/)
and the exact training log are retained. At its last progress report, the
learner had started **2,555 sixteen-step** and **565 sixty-four-step**
options; the option-aware actor usually used about 3,000 complete starts
per rollout and excluded only a few rollout-end unfinished holds. The
screen archive admitted **230 distinct fingerprints** over time, and all
16 workers reported 128 stored cells at their last episode.

The second [128-game selected evaluation](run/fresh-confirm-selected-128.json)
averaged **10,403.59**, versus **10,398.59** for the
[screen-diverse control](run/fresh-confirm-control-128.json) and
**10,409.14** for the [ordinary parent](run/fresh-confirm-ordinary-parent-128.json).
All **384** games stayed stage one. Against the control the option-credit
model improved 14 paired games, worsened 19 and tied 95; across both
untouched sets (**192 games per policy**), its mean edge over that control
is only **7.55** with **17 improved / 26 worse / 149 tied**. This does not
establish that the semi-Markov actor objective itself improved play.

Against the established ordinary-action parent, however, the two fresh
sets combined give option-credit / ordinary means of **10,406.41 /
10,398.28**, with **four sub-9,000 games each**. Option-credit improved
**86** paired games, worsened **20**, tied **86**, and narrowly meets the
predeclared *score-training-parent eligibility* rule. It is reasonable
to use its full selected state as a provisional training parent for new
experiments, while keeping the ordinary parent and the protected global
best replay intact. The +8.13 mean is small, not a level clear, and no
mission has been verified.

The selection's [fresh native-verified replay](run/fresh-selected-replay/replay.html)
exactly reproduces **2,537** neural decisions and 10,480 points. Its
[forensic course audit](../../diagnostics/option-credit-136-course-progress/README.md)
still records visible life losses at original stream rows **33, 34, 34,
33** of 126. Neither that hidden pointer nor any counterfactual action
was supplied to training or replay action choice. The repeated obstacle
remains unsolved.
