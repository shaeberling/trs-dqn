# Visible-screen novelty at the stage-one passage barrier

Previous exact-branch probes and learned key/memory changes still lose at
original first-stage course rows 33–34, despite the usual 2,620 points per
life. This experiment tests a different credit signal: during training only,
PPO receives **0.05 optimizer reward** for the first visit to each coarse,
HUD-free, visible gameplay-screen cell *within the worker's current life*.
The screen is the same native-rendered video memory already observed by the
policy. Cells reset at visible life loss or actual environment reset. There
is no course pointer, game RAM, wall/ship parser, target action, expert,
scripted route or evaluation bonus. The original displayed-score reward is
still used (multiplied by 0.01); adding novelty is a **real reward-shaping
change**, not a claim of score-only training. From-boot evaluation and the
protected best replay remain score/stage-only.

The [ordinary score parent](../ppo-persistent-noise-long-119/run/step-000009718528/)
is the exact common full optimizer, model and policy/noise RNG starting
point at counter **9,718,528**. Treatment resumes it with novelty beta 0.05;
control resumes it with beta 0. Both inherit 16 native workers (four boot
only), 256-step rollouts, 512-action minibatches, four epochs, 0.00025 Adam,
0.002 entropy, gamma 0.997, lambda 0.95, 100,000-T-state actions, per-life
symmetric logit-bias noise and 128-action own-visible-loss curriculum. Use
separate run and artifact directories, never the protected global-best
directory.

First require unit, native smoke/resume and whole-suite regression tests.
Run a **16,384-action integration smoke** for each arm; require finite PPO
updates, saved full model/optimizer/RNG state and no obvious gameplay
collapse. If both pass, restart each from the common frozen parent and run
**262,144 actions**, checking ten complete fixed from-boot seeds at every
65,536 actions. Abort an arm after two consecutive means below 5,000. Select
the earliest checkpoint with greatest stage/mission then highest fixed mean.
Any stage-two or mission claim requires an independently reexecuted native
policy replay, plus fresh-seed complete-game confirmation. Otherwise compare
the frozen selections and original parent on 64 untouched matched complete
games, seeds **610100–610163**; only if the novelty arm improves mean and
does not worsen the sub-9,000 tail against both does it earn a second fresh
64-game confirmation, seeds **610300–610363**. Score alone cannot promote
the protected best effort or prove course passage.

Archive full optimizer/RNG checkpoints, metrics, evaluations, independent
best-effort replays and source hashes. Honor the 5 GiB disk safety guard.

## Progress

The reward implementation and three focused tests (including original
emulator PPO training/resume) pass. The full repository suite passes **517
tests**. Both 16,384-action native integration smokes completed with finite
updates, full saved model/optimizer/RNG states and independently verified
original-boot local best-effort replays. Treatment's ten fixed complete
games averaged **10,426** (best 10,480; all stage one); control averaged
**10,408** (best 10,480; all stage one). Neither failed the 5,000-point
gate. In the treatment smoke, **7,035** first-time within-life visible
cells generated **351.75** auxiliary optimizer-reward units across the
16,384 training actions. This is substantial exploration pressure, not a
negligible logging feature. Both production arms restart from the original
frozen parent, not these smoke weights.

## Full matched gate and fresh result

Both arms completed **262,144** new training actions at counter **9,980,672**.
The treatment finished **36 full boot games and 832 own-reset segments**;
the control finished **36 / 845**. None reported stage two or a mission.
Treatment registered **124,560** first visits (**6,228** auxiliary reward
units) by its final progress event. That is nearly half the training actions:
ordinary moving-screen changes were often novel, which is a likely reason
this coarse bonus did not specifically identify a navigable passage.

| Arm | Fixed ten-game means at +65,536 / +131,072 / +196,608 / +262,144 |
| --- | --- |
| Visible novelty | 10,464 / 10,446 / 10,406 / **10,476** |
| Score-only control | 10,456 / 10,436 / 10,270 / **10,466** |

All **80** fixed complete games were stage one. Both frozen selections are
the final checkpoint by the predeclared stage/mean rule. On the first
untouched **64 matched original-boot games**, seeds **610100–610163**:

| Frozen policy | Mean / median / best score | Below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Novelty treatment](fresh-treatment-64.json) | **10,348.44 / 10,480 / 10,480** | 3 | 0 |
| [Score-only control](fresh-control-64.json) | **10,388.12 / 10,480 / 10,480** | 2 | 0 |
| [Unchanged parent](fresh-parent-64.json) | **10,434.69 / 10,460 / 10,480** | 0 | 0 |

Treatment beat the control in 14 paired games, lost 13 and tied 37;
against its own unchanged parent it won 31, lost 10 and tied 23, but
three low-score failures erased those wins in the mean. It fails the
frozen score-parent gate on **both** mean and low tail, so no second fresh
confirmation or score-parent designation is warranted. All **192** fresh
games were stage one. The best-effort [treatment](fresh-treatment-replay/replay.html),
[control](fresh-control-replay/replay.html) and
[parent](fresh-parent-replay/replay.html) replays were independently
reexecuted from original boot; their `verification.json`
files report verified native actions and no later stage. Nothing here
promotes over the protected global best replay.

The four smoke and full run directories plus their four isolated replay
artifact directories are archived here with every file and symlink
byte-checked against the original run directories. Complete policy,
optimizer and RNG states, validation tables, training metrics and original
screen-only source hashes are retained. The unchanged game executable and
5 GiB storage safeguard were respected throughout.
