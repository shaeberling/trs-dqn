# Fresh score-only PPO with balanced physical-control initialization

The protected best policy and many continuations score about 2,620 on each
life but fail to pass stage one's two offset openings. The prior trainable
grouped-fire trial resumed a mature, strongly fire-biased PPO checkpoint;
its negative result does not test a **fresh** physical-control prior. In
stage one, nine raw fire+arrow logits all mean forward fire. Starting their
twenty logits near zero gives forward fire roughly 9/20 initial mass,
while each distinct movement or side-fire command gets only 1/20. The
treatment subtracts `log(9)` from the nine alias biases at initialization,
so the fixed twelve-choice grouped distribution starts approximately
uniform over NOOP, eight movements, forward fire and two side fires.
All twenty neural output rows remain trainable. The matched grouped-control
starts from the same random seed and architecture without this offset.

This is a direction-neutral **initialization**, not a route rule, obstacle
cue, demonstration, extra reward or evaluation-time steering. Both arms
sample the same twelve grouped physical controls on every screen and use
only four original raw video-memory frames as neural input. PPO reward is
only original displayed score difference times 0.01, with visible ship-loss
learning boundaries. Evaluation uses complete original-boot games and the
ordinary sampled frozen neural policy. Any apparent later-stage result
requires independent native replay verification before best promotion.

Predeclared first comparison: seed 41, 16 emulator workers, 256-action
rollouts, batch 512, four PPO epochs, learning rate 0.00025, entropy 0.01,
gamma 0.997, lambda 0.95, 100,000-T-state actions, no reset curriculum,
no intrinsic reward or noise. Start each arm from scratch and run
**8,388,608** own actions, with ten complete fixed validation games at
each 1,048,576 actions. Preserve model/optimizer/RNG at every check.
Choose the earliest highest-stage checkpoint per arm, then the highest
fixed-game mean. Stage-two/mission training or validation events require
original-boot replay confirmation; a tied 10,480 stage-one score cannot
replace the protected best. If both remain stage one, compare each frozen
selection on 64 new matched complete games, seeds 611000–611063, and use
a second untouched set, 611200–611263, before claiming a score-training
parent. A stage claim needs original-boot learned-policy evidence, not this
diagnostic's searched branches. The 5 GiB disk safety threshold applies;
if approached, stop cleanly and preserve resumable states before extending.

Before production, require the complete repository test suite and separate
16,384-action integration smokes from scratch for both arms. Each must save
finite optimizer state, finish ten complete fixed validation games, and
produce an independently native-verified local replay. These smoke weights
are excluded from production; restart both full arms from seed 41.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-balanced-fire-160/treatment \
  --artifacts runs/defense-ppo-balanced-fire-160/treatment-artifacts \
  --canonical-fire --balanced-canonical-init --life-terminal \
  --steps 8388608 --seed 41 --envs 16 --rollout 256 --batch-size 512 \
  --epochs 4 --learning-rate .00025 --entropy .01 \
  --eval-every 1048576 --eval-games 10 --eval-envs 10 \
  --mlx-cache-mb 512

venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-balanced-fire-160/control \
  --artifacts runs/defense-ppo-balanced-fire-160/control-artifacts \
  --canonical-fire --life-terminal \
  --steps 8388608 --seed 41 --envs 16 --rollout 256 --batch-size 512 \
  --epochs 4 --learning-rate .00025 --entropy .01 \
  --eval-every 1048576 --eval-games 10 --eval-envs 10 \
  --mlx-cache-mb 512
```

## Integration passed

The full Mac-native repository suite passed **527 tests**. A zero-update
matched initializer test confirmed every checkpoint weight is identical
between arms except the nine specified actor biases. Separate 16,384-action
smokes completed finite PPO updates and ten full fixed-seed games each.
The [treatment smoke](smoke-treatment/step-000000016384/evaluation.json)
averaged **276**, best **320**; the [control smoke](smoke-control/step-000000016384/evaluation.json)
averaged **290**, best **300**. All twenty games were stage one, as expected
at fresh initialization. Local best-effort replays independently reproduced
**1,568** and **1,575** learned actions from original boot; both
`verification.json` files report `verified: true`. Full smoke model,
optimizer, RNG, metrics and replay bundles were **moved**, not copied,
into this archive. Neither smoke checkpoint enters production selection.

The full treatment started from the unchanged seed-41 fresh initializer in
`runs/defense-ppo-balanced-fire-160/treatment/`. A separate
`rl.defense_disk_watch` process checks the exact trainer PID and signals a
graceful stop if free space falls below **5.1 GiB**; it neither reads screens
nor changes learning. The trainer's `status.json`, optimizer checkpoint and
local verified replay are the authoritative progress records. A disk stop
is a safety pause, not an evaluation result or goal completion.

## First production checkpoint

At **1,048,576** new actions, the treatment's [fixed ten-game evaluation](milestone-000001048576/evaluation.json)
averaged **322**, median **320**, best **340**, all complete original-boot
games in stage one. Its [local best-effort replay](milestone-replay-000001048576/replay.html)
independently reproduced **1,580** neural actions and has `verified: true`.
The complete immutable model/optimizer/RNG checkpoint, fixed evaluation,
replay and frozen config were copied into this archive and compared
byte-for-byte with the active run. Training continues toward the remaining
predeclared checkpoints. This early score is not a barrier improvement or
global-best promotion; the protected replay remains unchanged.

## Why the repeated loss matters (post-launch audit)

The user's observation is supported by the independently original-boot-
verified [protected best replay](../../learned/best/replay.html), not yet by
this fresh learner. Its four visible ship losses occur at replay decisions
410 / 1133 / 1863 / 2580; each life earns exactly **2,620** displayed
points. Its last positive displayed-score increments occur at decisions
388 / 1114 / 1832 / 2550, leaving **22 / 19 / 31 / 30** consecutive
zero-reward decisions before those *visible* losses. These are not known
physical collision times. This audit uses only the replay's recorded score
increments, actions and visible life events; the protected trace SHA-256 is
`d83408861feaa23f75b56b8fad55faeaeedc25bf6717cde42307612beedd4a4d`.

The [verified-screen timing audit](../../diagnostics/gate-timing-127/README.md)
shows two offset openings: one around columns 21–30, then another beginning
at column 51. A learned ship stays near the middle as the latter approaches.
The [frozen-value audit](../../diagnostics/score-value-barrier-01/README.md)
also finds that the score critic expects almost no remaining within-life
points near the loss. Together these support a *navigation and sparse-credit
hypothesis*, not proof of a wall collision or an exact route. Earlier
constant-key, multi-phase and screen-conditioned searches failed within
their specified bounds; their actions are not demonstrations or learner
inputs.

The new treatment's first verified best-effort replay instead scores
**60 / 120 / 80 / 80** on its four lives (340 total, 1,580 reproduced
actions; trace SHA-256
`79723d75dce004ee015fd318be023c46ff5b68c51cbe9cb49155a0bada324849`).
It has **not** reached the 2,620-point bottleneck. Thus this experiment
tests whether a less fire-biased fresh action prior can first learn the
early route and eventually behave differently at the shared barrier;
its current low scores cannot be interpreted as a new barrier outcome.
No training reward, policy input, action, checkpoint selection rule or
predeclared run length changed because of this read-only audit.

## Second checkpoint, disk pause, and resume

At **2,097,152** actions, the next [fixed ten-game evaluation](milestone-000002097152/evaluation.json)
averaged **348**, median **350**, best **360**; all ten complete games
remained in stage one. The [360-point replay](milestone-replay-000002097152/replay.html)
was independently reexecuted for all **1,701** learned actions with
`verified: true`. Its model SHA-256 is
`a79d4b8d3096bd8e2c4c1a9c82c3cee93b153fa1283815a17869fd8d0d255cf1`.
The full checkpoint and resolved replay version were copied to this
archive and compared byte-for-byte with the active run. Neither result
approaches the protected 10,480-point stage-one best.

The disk watchdog then signalled a clean stop at **2,101,248** actions
when free space fell to **5.087 GiB**, below its 5.1 GiB floor. The
trainer saved its full `latest` model, optimizer and policy RNG. No
validation or model promotion was inferred from this safety stop.
After auditing stopped historical runs, we removed **399** superseded
intermediate checkpoint directories from a completed Breakdown run and
**299** from four retired Defense runs. Each run's selected/best and final
checkpoint, metrics, replay bundles and the separately preserved winning
or protected-best packages remain. Free space rose from about **5.07**
to **13.69 GiB**. Deleted intermediate optimizer states are not
recoverable unless separately archived; their validation history remains
in the run logs. This cleanup did not modify the game or live learner.

The same treatment resumed from its exact 2,101,248-action `latest`
optimizer/RNG state, with original episodes freshly booted as documented
by the trainer. The resumed process uses the same configuration and
8,388,608-action target, guarded by the exact-run 5.1 GiB watchdog.
Because the trainer schedules its next evaluation relative to the
resume action counter, the later fixed checks shift by **4,096** actions
from the original multiples; the first two checkpoint counts above are
unchanged. Keep all active treatment/control milestones until their
predeclared comparison is complete, then retain selected and terminal
resumable states, metrics, verified replays and evaluation records while
pruning unselected intermediate optimizer snapshots. This retention
policy prevents retired runs from again accumulating hundreds of
unneeded full checkpoints.

The resulting in-process checks are at 3,149,824 / 4,198,400 /
5,246,976 / 6,295,552 / 7,344,128; its next automatic check would be
8,392,704, **4,096 beyond the 8,388,608-action stop**. A one-shot
[`rl.defense_balanced_followup`](../../../../rl/defense_balanced_followup.py)
monitor therefore evaluates the frozen terminal treatment checkpoint on
the same ten complete original-boot seeds 10000–10009 and independently
verifies a local replay. It then launches the predeclared fresh matched
control only if the treatment ended at its target, its original and final
configurations still match, the full optimizer checkpoint is present,
the terminal evaluation is complete and verified, no control exists, and
free disk is at least **6 GiB**. The control gets its own exact-run
5.1 GiB disk watchdog. Any early stop, incompatible run, incomplete
evaluation or existing control fails closed; no policy selection or
promotion happens automatically. Focused supervisor/watchdog tests pass.
The one-shot monitor was launched while the treatment was live at about
4.0 million actions; it records its own current state in
`runs/defense-ppo-balanced-fire-160/followup-status.json`. Its command is
`venv/bin/python -u -m rl.defense_balanced_followup`. The trainer and
original exact-run disk watchdog remain separate live processes.

## Seventh checkpoint: fresh score learning finally emerges

The fifth and sixth fixed ten-game checks at **5,246,976 / 6,295,552**
actions averaged **562 / 512**, with all games in stage one. Between the
sixth and seventh checkpoints, the last 100 complete own-training games
rose from roughly 500 points to over 8,000 mean. These episodes can span
weight updates, so their scores are learning signals, not frozen-policy
validation or replay candidates.

The immutable [seventh checkpoint](milestone-000007344128/state.json) at
**7,344,128** actions then averaged **8,434**, median **9,170**, best
**10,220** over the ten complete original-boot fixed seeds 10000–10009.
All ten remained in stage one; no successful mission was observed. The
[10,220-point replay](milestone-replay-000007344128/replay.html) for seed
10002 was independently reexecuted for all **2,487** frozen-neural actions
with `verified: true`. Its checkpoint SHA-256 is
`5fd070f2372089901c935b4baa560bc99b4713c95a4e4c08c902bf57d1abc377`.
The complete model/optimizer/RNG/evaluation and dereferenced verified
replay bundle were copied into this archive and compared byte-for-byte
with the live originals. Training continues to its target; this result
does not replace the protected 10,480-point global best or imply passage
of the stage-one barrier.

## Treatment complete; matched control started

The treatment stopped normally at its exact **8,388,608**-action target,
with a complete [terminal model/optimizer/RNG state](terminal-checkpoint/state.json).
The one-shot follow-up evaluated that frozen model on the same ten
original-boot seeds 10000–10009 because the resumed run's next automatic
check fell beyond the target. The [terminal evaluation](terminal-evaluation.json)
averaged **8,502**, median **8,750**, best **9,910**; all ten games remained
in stage one. Its [9,910-point terminal replay](terminal-replay/replay.html)
was independently reexecuted for all **2,586** frozen-policy actions with
`verified: true`; model SHA-256 is
`c0f84420424b25ef129803e7f3144764ebc15bb9754e40704da9a7808d4bc241`.
The terminal fixed mean narrowly exceeds the seventh check's 8,434, so it
is the treatment's provisional validation selection under the predeclared
stage-then-mean rule. The state, evaluation and replay were copied into
this archive and compared byte-for-byte against their finished-run sources.
The [full compressed training log](treatment-metrics.jsonl.gz), including
every scored episode, validation and clean stop event, was decompression-
compared byte-for-byte against the finished run. The
[terminal status](treatment-status.json) is also preserved.
No game reached stage two or the successful mission sequence, and neither
model is promoted over the protected global best.

After verifying the terminal state and replay, the follow-up launched the
fresh seed-41 grouped-action **control without balanced initialization**,
with the same 8,388,608-action target and its own exact-run disk guard.
The control's live state remains under
`runs/defense-ppo-balanced-fire-160/control/`; the treatment/control
comparison and untouched fresh-seed gates have not yet run.

A separate one-shot [`rl.defense_balanced_compare`](../../../../rl/defense_balanced_compare.py)
monitor waits for the exact control process to stop normally at its target.
It fails closed on incompatible or incomplete states, requires all eight
ten-game fixed checks in each arm, and selects each frozen checkpoint by
successful mission, highest stage, then fixed mean (earliest on a tie).
It then runs both selected models on **64 matched fresh complete games** at
seeds 611000–611063 and a second untouched matched set 611200–611263.
Each set includes an independently native-verified local replay. The
monitor records paired outcomes under
`runs/defense-ppo-balanced-fire-160/fresh-comparison/` and never promotes
either model or modifies the protected global best. It waits for at least
6 GiB free before starting each frozen evaluation.

The separate [read-only seventh-replay loss comparison](../../diagnostics/balanced-fire-160-loss-comparison/README.md)
checks whether the large score jump represents a different passage through
the shared obstacle. Its verified treatment life scores were 2,530–2,570,
with forensic visible-loss stream rows 31–34 of 126. Despite more movement
choices than the protected best near those losses, it has not passed the
stage-one barrier. The diagnostic never enters learning or promotion.

## Matched control and untouched-seed comparison complete

The no-offset control stopped normally at its exact **8,388,608**-action
target. Its eight fixed ten-game means were **384 / 558 / 578 / 588 / 580 /
588 / 574 / 586**, all stage one. The predeclared stage-then-mean rule
selected the earliest tied **588** checkpoint at **4,194,304** actions;
the balanced arm selected its **8,388,608**-action terminal checkpoint
(fixed mean **8,502**). The selected model SHA-256 values are respectively
`f349809cd8607a8ad14bffebe5031d350e87dd05aa461fad2c8af62686e50659`
and `c0f84420424b25ef129803e7f3144764ebc15bb9754e40704da9a7808d4bc241`.

The [fresh matched report](fresh-comparison/report.json) used frozen models,
64 complete original-boot games per arm at seeds 611000–611063, then a
second untouched 64 per arm at 611200–611263. Balanced/control means were
**9,382.97 / 579.69** and **9,424.38 / 581.88**. The balanced arm won all
**128** paired seed comparisons. All **256** complete games remained in
stage one, with zero successful missions. Each of the four [fresh replay
bundles](fresh-comparison/) contains an independently native-verified local
replay (`verified: true`) tied to the selected model hash. Thus the
balanced physical-action prior substantially improved *learning from
scratch* under the same displayed-score reward, but did not solve the
observed stage-one obstacle or justify replacing the protected global-best
replay.

The archive preserves the control's [selected](control-selected/state.json)
and [terminal](control-terminal/state.json) complete model/optimizer/RNG
states, all eight [fixed evaluations](control-fixed/), its
[full compressed training log](control-metrics.jsonl.gz), resolved config
and terminal status, and the complete fresh comparison. These copies were
byte-compared against the finished run; the compressed log was
decompression-compared against the original. All seven treatment fixed
evaluations were additionally copied byte-for-byte into
[treatment-fixed](treatment-fixed/) before cleanup. After the comparison
archive was committed and pushed, **14** non-selected live checkpoint
directories (about **130 MiB**) were deleted. The treatment terminal and
control selected/terminal full resumable states remain locally and in the
archive; all fixed evaluations, logs and verified replays remain archived.
The pruned intermediate optimizer states are not recoverable unless one was
separately archived (the treatment's first, second and seventh were).
