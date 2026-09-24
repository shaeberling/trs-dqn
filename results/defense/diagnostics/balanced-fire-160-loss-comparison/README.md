# Balanced-fire treatment still fails at the early stage-one barrier

This is a **post-training, read-only diagnostic** of two independently
verified learned-policy replays: the treatment's
[10,220-point seventh-check replay](../../training/ppo-balanced-fire-from-scratch-160/milestone-replay-000007344128/replay.html)
and the [protected 10,480-point global best](../../learned/best/replay.html).
It neither selects a checkpoint nor changes training observations, rewards,
actions, resets or model weights. The game and both original replays remain
unchanged.

The [screen/action loss audit](screen-losses/report.json) confirms the
treatment's four life scores were **2,550 / 2,570 / 2,570 / 2,530**, against
the protected best's **2,620** on each life. In the 64-action windows before
the visible white-flash marker or final loss, the treatment chose ordinary
movement commands on **87.5% / 76.6% / 85.9% / 62.5%** of decisions,
versus **56.3% / 71.9% / 62.5% / 51.6%** for the protected model. The
[unaltered treatment screen sheet](screen-losses/policy-1-losses.png) and
[protected-model sheet](screen-losses/policy-2-losses.png) show both
approaching the same two-offset-opening region. These windows do not measure
actual movement or collision time; a firing command can suppress movement
in stage one, and the sprite can be temporarily obscured.

The separate [forensic course audit](course-progress/report.json) reexecutes
every recorded action, score increment and visible screen from original boot
before reading the game's hidden stream pointer **only at visible life-loss
bookkeeping**. It finds treatment stream rows **33 / 34 / 34 / 31** of 126,
versus protected-best rows **33 / 34 / 34 / 34**. The hidden pointer is
diagnostic-only: it must never enter the policy, training reward, reset
selection, checkpoint ranking or replay promotion. Stream rows are not exact
collision locations or proof of a passable path.

Thus the new balanced-control prior learned substantial stage-one scoring
and a more movement-heavy policy, but this verified replay supplies **no
evidence of passing the recurring barrier or entering stage two**. A new
training strategy must be judged by independently replayed later-stage
behavior, not by stage-one action counts or a near-tied score. These are two
selected replays, not matched-seed population estimates; the control and
fresh comparison continue separately.

Reproduce with the unchanged tools:

```sh
venv/bin/python -m rl.defense_loss_probe \
  results/defense/training/ppo-balanced-fire-from-scratch-160/milestone-replay-000007344128 \
  results/defense/learned/best \
  --output runs/defense-ppo-balanced-fire-160/loss-audit-7-reproduce

venv/bin/python -m rl.defense_course_progress_probe \
  results/defense/training/ppo-balanced-fire-from-scratch-160/milestone-replay-000007344128 \
  results/defense/learned/best \
  --output runs/defense-ppo-balanced-fire-160/course-audit-7-reproduce
```
