# Joint symmetric key-and-duration exploration

The [previous duration learner](../ppo-duration-133/README.md) exercised
hundreds of 64-action holds, but both its frozen replay and its conservative
pilot still lost every life around decoded course row 33–34 of 126. A
[screen-only timing probe](../../diagnostics/gate-timing-127/README.md)
found two offset openings: simply holding RIGHT too early loses at the first,
while starting late falls short of the second. Direction-neutral duration
noise alone does not greatly increase the chance of *sustained directional*
movement in the narrow transition window. This is a hypothesis, not an
oracle route or a measured collision-time requirement.

This opt-in trial starts from the same frozen own PPO key-duration
[initializer](../ppo-duration-133/initializer-spacing4/latest/state.json).
At each training-life start, it draws independent zero-mean physical-key
factors (standard deviation **2**) and hold-length factors (standard
deviation **4**). A physical factor is shared across all four durations and
a duration factor across all 20 physical commands. Left and right have the
same distribution. The network still samples every option from four raw
screen frames; execution only counts down its own selected hold. The
perturbations are included in PPO action likelihoods and removed from
evaluation. Reward remains displayed score difference. There is no
obstacle-coordinate input, stage reward, hand-picked direction, scripted
expert or demonstration.

Run **1,048,576** base training actions with the inherited own-state
life-loss curriculum, 16 workers (four boot-only), learning rate 2.5e-5,
GAE lambda .95, and eight fixed ten-game complete validations every
131,072 actions on seeds 10000–10009. Archive every full optimizer/RNG
milestone. Select the earliest highest-stage checkpoint, breaking stage
ties by fixed-game mean. At the endpoint compare that frozen selection,
the previous factor-duration selection at 393,216, and the untouched
initializer on **64 new matched complete games**, seeds **606400–606463**.
Also include the established ordinary-action score parent at
`results/defense/training/ppo-persistent-noise-long-119/milestone-000009718528`
on those same seeds; otherwise a gain over the weakened duration initializer
could be misleading. This comparator does not affect checkpoint selection.
No score-only gain replaces the protected best replay or counts as a stage
pass. Any stage-two candidate requires independent native-verified replay
and fresh confirmation. If two consecutive fixed checks have means below
5,000, end and archive the collapsed trial early.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-joint-134 \
  --artifacts runs/defense-duration-joint-134/artifacts \
  --resume results/defense/training/ppo-duration-133/initializer-spacing4/latest \
  --steps 1048576 --learning-rate 0.000025 --gae-lambda 0.95 \
  --policy-bias-noise 0 --policy-key-noise 2 --policy-duration-noise 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10
```

The predeclared run finished without a stage-two game. The seventh
917,504-action checkpoint won fixed-game selection with **10,474** mean.
On the first 64 fresh matched games its mean was **10,382.50**, versus
**10,357.34** for the established ordinary-action score parent, **9,832.81**
for the earlier duration-noise selection, and **9,535.00** for the frozen
duration initializer. All **256** games stayed stage one. Against the
ordinary parent it improved 32 games, worsened nine and tied 23. A
25.16-point mean edge on one fresh set is too small to declare a new score
parent, although the paired distribution merits a second check.

Before running that check, freeze both the selected joint checkpoint and
the ordinary parent and compare them on **128 further untouched matched
complete games, seeds 606600–606727**. Do not reselect checkpoints or tune
temperature. Only if the joint checkpoint retains a positive combined
192-game mean edge with no worse sub-9,000 tail can it be considered a
new *score-training* parent. Even then, stage-one score gain is not a
barrier pass and never displaces the protected best replay. Preserve any
negative confirmation too.

The complete run's eight fixed ten-game means were **9,384 / 10,271 /
10,420 / 10,219 / 10,468 / 10,217 / 10,474 / 10,466**. No training or
validation game reached stage two. At the last progress report, 1,036,288
training actions, the learner had sampled **3,764 sixteen-step** and **473
sixty-four-step** options. The [selected full checkpoint](run/step-000000917504/state.json)
keeps its optimizer and policy/noise RNG; the [complete run archive](run/)
retains all eight milestones and local verified replay history.

The second [128-game joint confirmation](run/fresh-confirm-selected-128.json)
averaged **10,375.78**, while the matched
[ordinary parent](run/fresh-confirm-ordinary-parent-128.json) averaged
**10,408.59**, all stage one. The joint policy improved 60 paired games,
worsened 20 and tied 48, but five sub-9,000 outliers versus two outweighed
its many small gains. Across both untouched sets (**192 games each**), the
joint/ordinary means were **10,378.02 / 10,391.51**, with **seven / four**
sub-9,000 games. Joint improved 92 paired games, worsened 29 and tied 71,
yet failed both predeclared mean and lower-tail requirements. It is **not**
a new score parent or a stage-pass claim.

The selected checkpoint's separate [fresh 10,480-point replay](run/fresh-selected-replay/replay.html)
independently replays **2,546** learned neural decisions. A
[read-only original-course audit](../../diagnostics/joint-134-course-progress/README.md)
finds visible life losses at decoded row **34, 33, 33, 33** of 126. No
hidden pointer was used to train, evaluate, select or play this model.
The global protected best replay remains unchanged.
The full **491-test** repository regression suite passed after adding the
joint exploration mode.
