# Movement-only extension has not progressed through the course

This is a post-freeze diagnostic of the independently verified
[2,097,152-action extension replay](../../training/ppo-movement-only-164/extension/milestone-000002097152/verified-replay/replay.html).
The selected fixed ten-game mean improved from the pilot's 368 to 374, but
the exact original-boot replay's four visible losses occurred at decoded
stage-one stream row **16 / 16 / 16 / 16 of 126**. Its displayed cumulative
scores were 80 / 180 / 280 / 380. The pilot's separate fresh replay lost at
rows 15 / 15 / 16 / 16. These are different replay seeds and not a paired
course-progress estimate; together they provide no evidence that more
movement-only PPO has escaped the early barrier.

The separately preserved [400-point training best effort](../../training/ppo-movement-only-164/extension/training-best-efforts/versions/step-000003407872-b67a70a980b5-seed-10005/replay.html)
is a useful caveat: its [forensic report](training-best-report.json) found
one life at row **21**, with the other three at 15 / 15 / 15. Its
[screen sheet](training-best-losses.png) shows that isolated effort. This
is some within-stage movement beyond the usual no-fire row-16 loss, not a
selected policy improvement, a repeatable course-depth estimate, or passage
of the stronger firing policies' row-33/34 bottleneck.

The [course report](report.json) pins the original game, replay trace, model,
and source hashes. The private stream pointer was read only after visible
loss bookkeeping while reexecuting every one of the verified replay's 1,695
actions and screen/reward observations. It was never a policy input, reward,
curriculum target, action source, or checkpoint selector. Its value is not
an exact collision timestamp or a count of rows safely navigated.

A separate [screen/action report](loss-report.json) compares this movement
replay with an already frozen grouped-duration firing replay. Its two
unaltered screen sheets are [movement](movement-losses.png) and
[firing](firing-losses.png). The sheets illustrate different action and
failure patterns, but are selected best efforts, not representative matched
games. The firing replay's [independent course audit](../grouped-duration-163-course-progress-14m/README.md)
found losses around rows 33–34; neither policy reached stage two.

The source code copied beside the course report is provenance for this
forensic replay only. No model updates or replay promotion came from it.
