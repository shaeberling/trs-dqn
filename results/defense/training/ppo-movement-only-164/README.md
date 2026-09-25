# Fresh stage-agnostic movement-only PPO: remove the firing-score shortcut

The previous [grouped key-duration trial](../ppo-grouped-duration-fresh-163/README.md)
trained for 16,777,216 own actions, but all 160 fixed and 256 fresh new-policy
games lost in stage one. Its verified best fresh replay earns 2,520 displayed
points on each of four lives; the visible reward trace includes large
750/1,500-point firing gains shortly before the recurring loss. The
protected score parent earns 2,620 per life and also loses there. The
visible-screen diagnostic suggests the score objective may reward firing
more quickly than repositioning for the later opening. It does **not**
establish a safe route or prove that suppressing fire will work.

This trial changes the **fixed action profile**, not the reward: the fresh
policy has nine logits for the original keyboard IDs 0–8 (NOOP and eight
movement directions). Those same nine choices apply at **every stage**.
There is no stage-aware controller, action override, programmed direction,
demonstration, original-game patch, hidden memory input or auxiliary reward.
The actor still sees only four raw 16×64 video-memory frames; it learns from
the displayed score difference and visible episode/life boundaries. It can
never fire, including after a stage transition, so this may impair later
stages; any passage must be observed in the original game before expanding
the action profile. This is an exploration/action-competition test, not a
replacement for the globally protected best policy.

Predeclared sequence:

1. Require the full native regression suite and focused action-profile tests.
   Freeze a seed-44 **untrained initializer** with the exact production model
   and evaluate 64 complete original-boot games, seeds 620000–620063. Save
   the full state, evaluation and independently verified best replay. This is
   a no-learning control, not a candidate for promotion.
2. Run a separate **16,384-action** seed-44 smoke with 16 original-emulator
   workers, 256-action rollouts, batch 512, four epochs, learning rate
   0.00025, entropy 0.01, gamma 0.997, GAE lambda 0.95, visible-score scale
   0.01, 100,000 T-states per action and life-terminal targets. Require a
   finite optimizer update, exact stop/full optimizer and RNG state, all
   chosen keyboard IDs below 9, ten complete fixed games on seeds
   10000–10009 with mean at least 100, and a native-verified original-boot
   replay. Smoke weights are never production weights.
3. On passing smoke, restart **fresh** from seed 44 for a bounded
   **1,048,576 own-action** pilot with the same settings. Evaluate ten
   complete fixed original-boot games every 262,144 actions. Select mission
   first, then highest stage, then fixed-game mean, earliest on a tie. Any
   stage-two or mission claim requires independent original-boot neural
   replay verification.
4. Compare the frozen selected pilot checkpoint and the frozen untrained
   initializer on **128 untouched matched complete games**, seeds
   620200–620327. Report score, game length, stage and mission separately;
   verify one best-effort replay per arm. If still stage one, extend the
   selected learner toward 4,194,304 total actions only if its fresh mean
   displayed score exceeds the initializer by at least **20** points and its
   mean complete-game action count exceeds it by at least **20** actions.
   These are compute-allocation gates, not a new reward or checkpoint rank.
   If the gate fails, archive the negative result and change mechanism.

The exact-run disk guard stops new training gracefully below 5.1 GiB free.
Selected and terminal full states, all fixed evaluations, compressed metrics
and verified replays must be archived and pushed before deleting unselected
local checkpoints. The protected 10,480-point global replay is never
overwritten by a lower-score or unverified outcome.

## Frozen no-learning control

The full **563-test** native regression suite passed with the opt-in action
profile. The seed-44 initializer stopped at **zero** training actions with
its complete [model/optimizer/RNG state](baseline/checkpoint) preserved.
Across the predeclared **64** complete original-boot games at seeds
620000–620063, its [evaluation](baseline/baseline-64.json) averaged
**290.94** displayed points (median 290, best 340) and **1,510.59**
neural actions per game; all games ended in stage one. Its
[best-effort replay](baseline/verified-replay/replay.html) independently
reproduced **1,611** actions, rewards and screens. Model SHA-256:
`e22114747b5cd18611a13ea052ab5476ec1b308738ea9d3208e21b5407e43bf1`.
This untrained control's weights will not initialize the production run;
that run restarts independently from the same fresh seed and settings.

## Integration smoke

The separate seed-44 smoke stopped normally at exactly **16,384** own
actions. Its [checkpoint](smoke/checkpoint) retains model, finite optimizer
and RNG state; all ten fixed complete original-boot games averaged **316**
points (median 320, best 340), still stage one. The independently
[verified replay](smoke/verified-replay/replay.html) reproduced **1,588**
neural actions, rewards and screens, and every chosen keyboard ID was in
the nine-action set. Configuration, metrics and status are preserved under
`smoke/`. All predeclared integration gates passed. Neither the smoke
weights nor its score will initialize or select production checkpoints.

After byte-comparing the initializer and smoke state, replay bundles,
configurations, metrics and statuses with this pushed archive, their three
stopped local run/artifact directories were deleted (about **36 MiB** by
`du`). The archived baseline and smoke remain recoverable from this branch.

The fresh seed-44 production pilot is active at
`runs/defense-ppo-movement-only-164-pilot`; its settings match the archived
smoke field-for-field except the run/artifact paths, target action count
and fixed-evaluation interval. An exact-run 5.1-GiB disk guard is active.
The fail-closed [`rl.defense_movement_compare`](../../../../rl/defense_movement_compare.py)
watcher checks the normal exact-target stop and all four full fixed
checkpoints, independently verifies a claimed later stage, then evaluates
the frozen selected model and no-learning control on the predeclared 128
fresh matched games with original-boot replay checks. It neither alters
training nor promotes or prunes a model. Its live status is in the run's
`comparison-status.json`.

## Bounded pilot result and extension gate

The pilot stopped normally at exactly **1,048,576** own actions and logged
**640** complete training games. No training game reached stage two. Its
four fixed ten-game means were **322 / 346 / 364 / 368**, best scores
**340 / 360 / 380 / 380**; all **40** fixed games ended in stage one. The
terminal [full checkpoint](pilot/checkpoint) wins the predeclared selector.
A separate original-boot reload reproduced all ten terminal fixed games
exactly, and its [replay](pilot/verified-replay/replay.html) verified
**1,629** neural actions, rewards and screens. Model SHA-256:
`664ded8ace39fafcc8b3e462bb8d5c93b8283207af1be67388685bd58167e34b`.
The [pilot archive](pilot/) also retains all four fixed evaluations,
complete compressed metrics, normal stop and disk-guard statuses.

The fail-closed [fresh report](pilot/fresh-comparison/report.json) compared
that frozen model with the separately frozen no-learning control on the
predeclared **128** matched complete original-boot games, seeds
620200–620327:

| Policy | Mean / median / best displayed score | Mean actions | Highest stage |
| --- | --- | ---: | ---: |
| [Movement learner](pilot/fresh-comparison/movement-620200-replay/replay.html) | **361.72 / 360 / 380** | **1,641.63** | 1 |
| [No-learning control](pilot/fresh-comparison/baseline-620200-replay/replay.html) | 286.72 / 280 / 340 | 1,505.91 | 1 |

The movement learner won **127** paired scores, lost **zero**, and tied
one. Both best-effort fresh replay bundles passed independent original-boot
verification. Its +75-point score and +135.71-action length gains exceed
the predeclared +20/+20 stage-one extension gate, but **no stage passage**
was observed. The learner therefore earns *more score-only exploration*,
not promotion over the protected 10,480-point policy.
A separate [post-freeze course audit](../../diagnostics/movement-only-164-pilot-course-progress/README.md)
located that verified 380-point replay's four visible losses at original
stream rows **15 / 15 / 16 / 16 of 126**. It has not yet approached the
firing policies' row-33/34 barrier. The private pointer was diagnostic-only
and did not enter learning or selection.

The extension resumes the archived pilot's exact model, optimizer and
policy RNG at 1,048,576 actions, with fresh original-game episodes, toward
**4,194,304 total own actions**. All action, screen, reward and optimizer
settings remain unchanged; ten complete fixed games are checked every
262,144 new actions. Freeze the earliest stage/mission-then-mean selected
checkpoint. Independently verify any later-stage claim from original boot.
If still stage one, compare the frozen extension selection against the
frozen pilot selection on **two untouched matched sets of 128 complete
games**, seeds **620400–620527** and **620600–620727**, with one verified
best-effort replay per arm/set. This checks whether longer no-fire training
actually improves survival beyond the pilot, rather than trusting a fixed
seed or transient training curve. The disk guard and archive-before-prune
rules remain in force.

After verifying and pushing the pilot archive and confirming the trainer,
disk guard and comparison watcher had exited, its two stopped local
run/artifact directories were deleted (about **71 MiB** by `du`). The
selected/terminal full state, all four fixed evaluations, full metrics and
both fresh replay bundles remain on this branch. The three unselected
intermediate optimizer snapshots were intentionally pruned and are not
recoverable; their complete fixed evaluation records remain. The extension
resumes the archived selected full state, not a local leftover.

The extension is now active at `runs/defense-ppo-movement-only-164-extension`.
Its resume configuration differs from the archived pilot only in the new
run/artifact paths, the exact archived resume checkpoint and 4,194,304-action
target. An exact-PID 5.1-GiB disk guard is active. The tested, fail-closed
[`rl.defense_movement_extension_compare`](../../../../rl/defense_movement_extension_compare.py)
watcher waits for a normal exact-target stop, checks all **twelve** extension
fixed checkpoints and any later-stage claim, then runs both predeclared
128-game matched sets and four original-boot replay checks. It never changes
training, promotes weights or prunes data. Its live state is
`runs/defense-ppo-movement-only-164-extension/comparison-status.json`.

At **2,097,152 total actions**, the fourth extension fixed check averaged
**374** over ten complete original-boot games (best **380**), surpassing
the pilot's 368 fixed mean. The first four extension means were
**354 / 362 / 362 / 374**; all forty games stayed in stage one. The new
current selection's [full model/optimizer/RNG checkpoint](extension/milestone-000002097152/checkpoint)
and [verified replay](extension/milestone-000002097152/verified-replay/replay.html)
were byte-compared with the live files. A fresh reload reproduced all ten
fixed game records and all **1,695** replay actions, rewards and visible
screens. Model SHA-256:
`347ba5c5662c8df8cec5b1a3a9ce4d457d60c0c885444ad98e7fd833e61c9353`.
This is a recoverable survival-score milestone, not a later-stage result;
training and the frozen fresh-comparison plan continue.

A [post-freeze forensic audit](../../diagnostics/movement-only-164-extension-course-progress/README.md)
of that selected verified replay found all four visible losses at original
stage-one stream row **16 of 126**, the same early region as the pilot.
The pointer was diagnostic-only and did not enter the learner, reward, or
checkpoint selector. The small fixed-score gain has not yet demonstrated
course progress; this interim audit did not alter the bounded extension or
its predeclared fresh comparison. Their final results follow.

## Final extension result and disposition

The extension stopped normally at exactly **4,194,304 total own actions**.
Its log contains **1,920 new complete training games**; the resumed status
counter is **2,560 cumulative games** including the pilot's 640. None
reached stage two.
The twelve fixed ten-game means were **354 / 362 / 362 / 374 / 358 / 364 /
362 / 360 / 368 / 342 / 342 / 344**; all 120 original-boot games remained
in stage one. The predeclared selector retained the already archived
**2,097,152-action** checkpoint (ten-game mean **374**, best **380**).
The [terminal full model/optimizer/RNG checkpoint](extension/terminal-checkpoint)
averaged only **344**, best **360**, at the final fixed check. The
[complete run record](extension/final-run-record) preserves configuration,
normal-stop and disk-guard records, every fixed evaluation and losslessly
compressed full metrics. The two versioned, verified
[training best efforts](extension/training-best-efforts/versions) are also
preserved; the later one scored **400**, still stage one.

The fail-closed [fresh comparison](extension/fresh-comparison/report.json)
froze that selected extension checkpoint and the pilot checkpoint, then
played two untouched 128-game matched sets per arm:

| Seeds | Extension mean / median / best | Pilot mean / median / best | Extension wins / pilot wins / ties | Later-stage games |
| --- | --- | --- | --- | --- |
| 620400–620527 | **375.47 / 380 / 380** | 364.06 / 360 / 380 | 65 / 11 / 52 | 0 |
| 620600–620727 | **376.25 / 380 / 380** | 362.34 / 360 / 380 | 74 / 7 / 47 | 0 |

All **512** fresh complete games across both arms ended in stage one. Each
arm/set has a separately original-boot-verified best-effort replay in the
comparison archive. The extension selection modestly improved score and
game length over the pilot on these timing-jittered original-game resets,
but this is not a mission result or a broad randomized-layout test.
The [course-progress audit](../../diagnostics/movement-only-164-extension-course-progress/README.md)
found the selected replay losing at row **16 / 16 / 16 / 16 of 126**. An
isolated 400-point training-best replay had one life reach row **21**;
its other three lost at row 15. This does not approach the firing policies'
row-33/34 failure, let alone finish stage one. The course pointer was
diagnostic-only, never a policy input, reward or selector.

Longer movement-only training is **not justified as the next mission
experiment** by these results. Removing fire also excludes a control
available in later stages. The protected [10,480-point global-best learned replay](../../learned/best/replay.html)
and original game remain unchanged. Future work should target the observed
phase-sensitive fire/movement bottleneck with a different learning mechanism,
not simply extend this score-improving but stage-one-only action profile.
