# Fresh recurrent balanced-control Defense test

## Why this test, and what it cannot prove

The protected learned replay scores 10,480 but loses four 2,620-point lives
in stage one, near original stream rows 33–34 of 126. Repeated high-score
PPO continuations and open-loop suffix searches have not crossed that
barrier. A trained 50,000-T-state, history-matched control also lost near
the same rows, so simply increasing the action rate is not an untested fix.
Neither the static original-binary audit nor any current replay proves that
this emulator has a passable stage-one trajectory at the policy cadence.
Do not describe a stage-one score gain as passage or solvability evidence.

This bounded experiment combines two already separate learner mechanisms:
fixed grouping into twelve **physical** keyboard choices, initialized with
equal mass, and a GRU that remembers only its own prior rendered screens and
chosen physical key. The original game, raw four-frame video-memory input,
100,000-T-state action duration, visible-score-difference reward, visible
life boundary and original-boot evaluation remain unchanged. The grouping
is fixed for all screens and all stages; it does not read stage or choose a
direction. No source replay, search branch, internal pointer, scripted
controller, hidden-state feature, intrinsic reward or demonstration enters
training. A zero-memory-scale control has the same architecture, random
initial weights and physical action prior; only the recurrent residual is
disabled. This isolates whether usable own-history memory changes learning.

The trial is not a substitute for an independent emulator reachability
proof. It is one distinct score-only learner test while that engineering
question remains open, not permission for repeated longer PPO tuning if
both arms end at the same stage-one barrier.

This is a **stage-one discovery test, not yet a full-game-eligible policy**.
The fixed grouping emits canonical Space for all nine forward-fire aliases.
That preserves their stage-one physical equivalence, but the original
stage-two/three routines allow movement while firing, so the grouping
removes potentially necessary later-stage commands. Also, the treatment
GRU explicitly receives its own previous physical key. This is not hidden
game state or an oracle, but it is not literally a screen-only model input
under the strict wording of `GOALS.md`. Until that interpretation is
resolved, any passage is a diagnostic discovery, not proof that the
specified screen-only learner has completed the game. A full-game
successor should retain all twenty original controls, avoiding this
unproven restriction, and by default receive only screen observations
and its learned recurrent state.

## Frozen protocol, before looking at outcomes

Use fresh seed **41**, 16 original-emulator workers, 256-action rollouts,
512-action minibatches, four epochs, learning rate 0.00025, entropy 0.01,
gamma 0.997, GAE lambda 0.95, reward units 0.01, 128 GRU units and
32-action recurrent sequences. The only arm difference is
`--memory-scale 1` versus `--memory-scale 0`. Do not initialize either arm
from an older model. Keep all four physical-control modes available at
every stage, including original side fire. Use the frozen sampled neural
policy for complete original-boot evaluation.

1. Require full native regressions, a saved zero-update weight-parity check,
   and an exact 16,384-action integration smoke **per arm**. Each smoke must
   stop normally, retain finite model/optimizer/RNG state, evaluate ten
   complete fixed games on seeds 10000–10009 with mean score at least 200,
   and produce an independently verified original-boot replay. Smoke
   weights never initialize production.
2. On both smoke gates passing, restart **each arm from fresh seed 41** for
   8,388,608 own actions. Evaluate ten complete original-boot games every
   1,048,576 actions on fixed seeds 10000–10009. Preserve all fixed
   evaluations and full checkpoint state at each check until selection.
   Select the earliest checkpoint by successful-mission count, then highest
   stage, then fixed-game mean score. Any later-stage claim requires a
   reloaded-model, independently verified original-boot trace. Do not stop
   an arm solely for low early score: the prior balanced learner first made
   a large score jump after five million actions.
3. If neither arm shows a verified later stage, compare the frozen selected
   checkpoints on **two untouched matched 64-game sets**, seeds
   622000–622063 and 622200–622263. Report score mean/median/best,
   paired wins/losses, low-score tail, stage, mission and complete-game
   counts separately. Verify one complete replay per arm/set. A score gain
   is only evidence about score; it does not promote the protected best or
   justify repeating this mechanism indefinitely. If either arm reaches a
   verified later stage, prioritize independent fresh-game confirmation
   and preservation of that learned model/replay.

The protected best model and replay are immutable unless a genuinely
better **independently verified learned-policy** result is found. Use an
exact-PID 5.1-GiB free-space guard for each live trainer. After normal
completion and fresh comparison, archive and hash-check the selected and
terminal full optimizer states, full compact metrics, every fixed and fresh
evaluation, and verified replays before pruning unselected local snapshots.
Never delete an active run, unique checkpoint or the protected replay.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-recurrent-balanced-169-treatment-smoke \
  --artifacts runs/defense-ppo-recurrent-balanced-169-treatment-smoke-artifacts \
  --canonical-fire --balanced-canonical-init --recurrent-hidden 128 \
  --recurrent-own-action --sequence-length 32 --memory-scale 1 --life-terminal \
  --steps 16384 --seed 41 --envs 16 --rollout 256 --batch-size 512 \
  --epochs 4 --learning-rate .00025 --entropy .01 \
  --gamma .997 --gae-lambda .95 --reward-scale .01 --tstates 100000 \
  --eval-every 16384 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512
```

For the control, change `treatment` to `control` in the run and artifacts
paths and set `--memory-scale 0`. For production, use fresh run/artifact
paths ending in `-full`, `--steps 8388608`, and `--eval-every 1048576`;
keep all other arguments exactly identical.

## Integration state

The complete repository suite passed **582** native tests with this
opt-in learner path. Three focused MLX/native tests passed after an extra
fail-closed resume-profile check. At seed 41, the separately saved
[treatment initializer](initial-parity/treatment-checkpoint/) and
[control state](initial-parity/control-state.json) have byte-identical
model weights **and optimizer file** before any update; both model SHA-256
values are `17e37b992b224d5bf244e34bd1d88ef75dd124a74778c61b63b3a8c8b7f94d69`.
The single archived model/optimizer copy avoids storing duplicate binary
data; both original run configs and status records remain in
[initial-parity](initial-parity/).

Both exact 16,384-action integration smokes stopped normally with finite
optimizer arrays. The treatment's ten complete fixed games averaged
**284** (best 320); the control's averaged **294** (best 320). All games
remained in stage one, as expected from this short fresh start. Reloaded
models independently reproduced every action, reward and screen of the
[treatment's 1,576-action replay](treatment-smoke/replay-versions/step-000000016384-a9d0793ffefd-seed-10001/replay.html)
and [control's 1,580-action replay](control-smoke/replay-versions/step-000000016384-74c581c3e959-seed-10001/replay.html)
from original boot. The complete smoke model/optimizer/RNG checkpoints,
fixed evaluations, compact metrics, statuses and replay versions were
copied into the respective arm directories and byte-compared with their
live sources. Smoke model SHA-256 values are
`a9d0793ffefd02fd855e3715e30cf22ec2c0b90bd7148eb6aa7e3cea4d4fe9a5`
and `74c581c3e959f08b8a763bc7175ee118ee9fa4e6338cb59ed3288ba552d7e513`.
These results pass the predeclared plumbing gate, not a gameplay gate.
Production must restart fresh; neither smoke checkpoint is eligible for
selection or promotion. No production model has yet been selected.

The learner and both smoke bundles were committed and pushed before the
long run. After comparing the archived checkpoints, configurations,
evaluations, metrics, statuses and verified replay directories against
the stopped live sources, the redundant local initializer/smoke run
directories were deleted. The archived smoke files are recoverable from
the pushed branch; the omitted control initializer binary was byte-
identical to the retained treatment initializer and is reproducible from
the saved control config and seed. No active or unique model was removed.

The fresh treatment is live at
`runs/defense-ppo-recurrent-balanced-169-treatment-full`, with its own
exact-PID 5.1-GiB disk guard. The tested
[`rl.defense_recurrent_balanced_followup`](../../../../rl/defense_recurrent_balanced_followup.py)
monitor verifies a normal treatment stop and all eight fixed checkpoints
before launching the fresh matched control and its own disk guard. After
both exact targets it independently checks later-stage claims, runs the
two predeclared fresh matched sets, and writes a report. It fails closed
on an early/incompatible stop, does not train on any diagnostic branch,
and never automatically promotes the protected best. Its live status is
`runs/defense-ppo-recurrent-balanced-169-followup/status.json`.

## First fixed checkpoint and post-launch regression correction

At **1,048,576** treatment actions, ten complete original-boot games
averaged **346** points (median 340, best 380); all ended in stage one.
The full checkpoint, evaluation and run-specific verified best-effort
replay remain in the active run/artifacts. This is an early recovery point,
not a stage-one barrier improvement or a new global best.

A post-launch full-suite run found that a new resume guard accidentally
rejected the **historically supported feedforward** ordinary-to-grouped
optimizer continuation. It did not affect fresh training or this live
treatment, but it was a regression. The guard was narrowed to recurrent
grouped-policy resumes; both historical canonical-fire and new recurrent
focused tests then passed. The complete corrected repository suite passed
**587 tests**. The already-running treatment had loaded trainer source
SHA-256 `56cfc7061c5bc8abe99159188fbd9bd3253c90392597c06a0eacf59858acf1c2`;
the later matched fresh control will load corrected trainer source SHA-256
`a6016adf5e7d72cdbd93a040adb73755b57104e3f8e75e5dad59fdfab791ca39`.
The only code difference between those two trainer hashes is the scoped
resume-validation guard. The follow-up explicitly permits **only** this
known source-hash difference, still requiring every other config field
except arm paths and memory scale to match. Its original process was
replaced after verifying its exact PID; the trainer and disk guard were
not interrupted. The restarted follow-up recognized the live treatment
at 1,118,208 actions and continues to monitor it.

## Strategy audit while production is live

At the second fixed check (**2,097,152** treatment actions), the ten
complete original-boot games averaged **322** points (median 320, best
340); all ended in stage one. The first checkpoint's independently
verified 1,611-action replay scored 380. A read-only ablation on the
*same recorded screens and preceding actions*, using that frozen model,
found mean total-variation distance **0.261** between grouped action
distributions with and without its recurrent residual; the greedy choice
changed on **57.2%** of screens. Memory therefore affects action
probabilities, but this off-policy ablation does not establish a gameplay
gain. The matched zero-memory control remains the causal comparison.

That replay chose a firing command in **186/1,611 (11.5%)** decisions;
the protected 10,480-point stage-one best chose one in **952/2,580
(36.9%)**. This is an early training snapshot, not a matched policy
comparison or evidence that firing alone solves passage. Earlier fresh
balanced learning improved sharply only after five million actions, so
the predeclared bounded run continues rather than stopping on the low
early score. No stage-two claim follows from a memory-effect or score
increase alone.

The broader evidence still points to route discovery and long-range
credit as the bottleneck: the protected replay loses four 2,620-point
lives around original course rows 33–34 of 126, and earlier bounded
searches and score-only continuations did not cross. With this trial's
gamma 0.997 and GAE lambda 0.95, direct advantage credit is multiplied
by about **0.947 per decision** (roughly a 13-decision half-life), while
the relevant maneuver can start many decisions before a delayed visible
loss. This is a diagnostic explanation, not proof of impossibility;
learned value bootstrapping can propagate farther if successful
trajectories are ever experienced.
Merely raising gamma/lambda is not an untested fix: the historical
[run-14 longer-return PPO](../../../../DEFENSE.md#longer-return-ppo-run-14)
used a roughly 346-decision direct-credit half-life from a strong score
parent, completed 2,064,384 further actions, and still never reached
stage two. Any successor needs a genuinely different exploration/credit
mechanism, not this parameter change alone.

After the frozen two-arm comparison, do not extend this mechanism merely
for stage-one score. Independently audit stage-one physical reachability
at the current and finer action cadence, using original-emulator mechanics
only for **forensics**, never as demonstrations, policy inputs or reward.
If a passable route is established, prioritize a full-twenty-command,
strictly screen-observed learner with a longer-horizon credit mechanism
and original-boot stage/mission gates. If action cadence makes passage
unreachable, correct that environment interface and restart controlled
learning; never relabel a diagnostic intervention as a learned win.

The third fixed check at **3,145,728** actions averaged **320** (best
320), all stage one. A separate [native forensic reexecution](../../diagnostics/recurrent-169-first-course-progress/README.md)
of the first checkpoint's verified 380-point replay matched every
recorded action, score increment and screen, then found visible losses
at original stream rows **13 / 13 / 16 / 16 of 126**. The lower score
therefore was not accompanied by deeper course progress in that selected
game. This is not a fresh-sample result or an early-stop gate.

## Completed treatment and corrected control handoff

The fresh treatment stopped normally at exactly **8,388,608** own actions
with all eight planned fixed checkpoints. Their ten-game mean scores were
**346 / 322 / 320 / 318 / 412 / 408 / 516 / 554**; every fixed game
remained in stage one and no mission occurred. The terminal checkpoint
is the frozen selector's treatment choice, with best individual score
**600** and model SHA-256
`03c5a616a7b5cf26dc095a359bff462d7e5adacf5b12048872caa168f2f3be20`.
This is below the protected 10,480-point learned best and does not
justify promotion. Its full optimizer/RNG state, evaluations and local
replay remain in the treatment run pending the matched comparison.
An independent [course-depth audit](../../diagnostics/recurrent-169-course-depth-initial/README.md)
of the archived terminal replay found all four visible losses at decoded
stream row **21 of 126**, up from rows 13–16 in the treatment's first
replay, but still before the protected score winner's row-33/34 barrier.
This is selected-replay progress, not a fresh-game or causal memory result.

The unattended follow-up initially failed closed before starting control:
it incorrectly required `checkpoint_sha256` inside the trainer's fixed
`evaluation.json`, whereas the trainer writes that file *without* a hash.
No result was lost and no treatment process was restarted. The corrected
follow-up now reloads **each of all eight** saved models and reruns the
same ten complete original-boot games, requiring exact game and summary
equality and a model-hashed recheck file. All eight treatment rechecks
matched exactly, including an independently run terminal recheck before
the monitor correction. The focused follow-up tests passed. The matched
zero-memory control has now launched from fresh seed 41 with its own
disk guard; its fixed and two untouched fresh comparisons remain pending.
No final causal memory conclusion is drawn from the treatment alone.

## Control disk interruption and fresh restart

The first fresh control launch advanced to **249,856** own actions, then its
exact-process disk guard observed **3.41 GiB** free (below the 5.1-GiB
floor) and signaled a clean stop. No fixed checkpoint had yet been due;
the latest full 249,856-action model/optimizer/RNG state, compact metrics,
configuration, trainer log, disk-watch event and follow-up failure status
are retained under [control-interrupted-disk-stop](control-interrupted-disk-stop/latest/state.json).
Its model and optimizer SHA-256 values are
`0aa6809e843534691e81122e8dad2cfb1d8f35bd801e241deb911bd0ef5a0a04`
and `28050d7991bda1eddfa8054dea4330394596a08e8e6bd756423467ee113a7562`.
This partial run is **excluded** from the matched comparison; resuming it
would restart emulator episodes and would not match the treatment's fresh,
uninterrupted protocol.

The free-space drop came from two roughly 8-GiB Git packs. An object-index
comparison proved the older pack had **zero unique objects**; Git's own
multi-pack-index writer was directed to prefer the superset pack, then
`git multi-pack-index expire` removed only the unreferenced duplicate.
The multi-pack index and repository connectivity verified afterward, and
free space recovered to about **17 GiB**. Automatic Git maintenance was
disabled in *this local clone only* (`maintenance.auto=false`, `gc.auto=0`)
to avoid another surprise repack during disk-guarded training. The first
post-fix full-suite run had one failure solely because its own disk guard
executed during the 3.4-GiB interval. A clean-space rerun passed **all
594 native tests**. No original game, learned weights or protected best
replay was changed by this storage repair.

The interrupted run and its logs were moved aside so the same fail-closed
follow-up can start a **new seed-41 control from zero** with the original
frozen configuration and a new exact-run disk guard. Reuse all eight
hash-bound treatment rechecks; do not incorporate any interrupted-control
weights, metrics or games in selection or fresh comparison. Only after
both planned arms and untouched sets finish should a causal memory
conclusion be reported.

At the restarted control's first two fixed checkpoints, **1,048,576** and
**2,097,152** own actions, the ten complete original-boot games averaged
**384** and **468** points (best **420** and **540**). All twenty games still
ended in stage one with zero missions. The trainer, exact-PID disk guard and
fail-closed follow-up remain live; neither early check is a matched final
memory result. The completed treatment's corresponding fixed means were
**346** and **322**, but no causal conclusion is drawn before the full
predeclared budget and untouched paired sets finish.
