# Grouped durations do not show a reliable first-life survival gain

The [evaluation-only visible-life probe](../../../../rl/defense_visible_life_probe.py)
compared the frozen [seed-41 balanced-fire continuation](../../training/ppo-balanced-fire-continuation-161/README.md)
with the frozen [fresh grouped-duration PPO selection](../../training/ppo-grouped-duration-fresh-163/README.md).
Both were run on the same **64 previously unused complete original-boot
games**, seeds 621200–621263, at each checkpoint's original action timing.
The probe logs only screen-derived ship-loss steps and displayed scores;
it does not decode game memory, alter actions, select a checkpoint, or
change the score reward. All **128** games ended in stage one.

| Frozen policy | Mean / median / best displayed score | Mean / median actions to first visible loss |
| --- | --- | --- |
| [Balanced-fire parent](parent-64.json) | 9,918.59 / 10,300 / 10,460 | 387.92 / 408.5 |
| [Grouped-duration selection](grouped-64.json) | 9,273.75 / 9,890 / 10,010 | 391.64 / 407 |

The grouped policy's first loss was later on **20** paired seeds and
earlier on **40**, with **4** ties. Its mean first-life length was **3.72
actions higher** because of tail outcomes, while its mean displayed score
was **644.84 lower**. This is not a reliable passage or survival advantage
over the score parent. It is consistent with the separate post-freeze
[course-pointer audit](../grouped-duration-163-course-progress-14m/README.md)
of an earlier grouped-duration checkpoint, whose replay still lost around
rows 33–34 of 126.
The private pointer was not used in this timing probe, any policy input,
reward, or model selection. Action counts include animations and are only
a rough survival proxy, not exact course-row or collision measurements.

The linked JSON files retain every complete game, visible loss event,
checkpoint/source/game hash and full evaluation result. The probe passed
native replay parity and the full 571-test suite before these evaluations.
