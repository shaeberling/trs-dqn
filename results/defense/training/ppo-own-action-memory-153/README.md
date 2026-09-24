# Learned own-action memory at the repeated two-gap barrier

The [native screen diagnosis](../../diagnostics/preposition-window-151/README.md)
shows a fixed late steering correction cannot preserve both nearby
passages. The strongest ordinary categorical PPO still observes four
rendered video-memory frames but not which physical key it just chose.
This experiment tests whether giving a recurrent residual learner its
*own previous key* helps it sustain and time early screen-conditioned
movement. The input is only displayed pixels and the learner's own
prior command. There is no game RAM, wall or ship parser, route target,
demonstration, counterfactual action, shaped reward or automatic
steering in the trainer or evaluation policy.

Freeze the same [ordinary score parent](../ppo-persistent-noise-long-119/run/step-000009718528/)
for both arms. Copy its complete 20-action CNN/actor/critic exactly into
a zero-output residual GRU, freeze that base, and start fresh Adam and
policy RNG with seed **41**. Treatment adds a one-hot previous chosen
physical key (plus a reset sentinel) to the GRU input. Control uses
the established screen-only residual GRU. Both carry neural memory
across visible ship losses, resetting it and the own-key sentinel only
on an actual environment/game reset. PPO trains on contiguous 32-step
within-worker sequences and stores the exact previous-key input seen
at each step. Neither arm changes the physical action set, screen
timing (**100,000 T states**, stride 1), original score-only reward,
life-terminal learning target or from-boot evaluation protocol.

First run **16,384-action integration smokes** separately, with 16
workers, 256-step rollouts, 512-base-action minibatches, four epochs,
learning rate **1.25e-4**, entropy **0.002**, gamma **0.997**, GAE
lambda **0.95**, four protected from-boot workers and the same
128-action own-visible-life-loss rewind curriculum. Require finite
updates, full model/optimizer/RNG save, independently verified local
replay and ten complete fixed-seed games for each arm. A zero-update
initializer parity test and focused/whole-repository tests must pass.
Smoke scores are integration evidence only; if either arm falls below
a 5,000-point ten-game mean, do not scale the pair.

If integrated, restart both arms from the frozen source for **262,144
new base actions**, evaluating ten complete fixed-seed games at
131,072 and 262,144 actions. Stop an arm after two consecutive means
below 5,000, otherwise finish both checks. Select its earliest
highest-stage checkpoint, breaking stage ties by fixed mean. Any
stage-two or mission report needs original-boot replay verification
and fresh complete-game confirmation before promotion.

If all checks remain stage one, compare frozen selected treatment,
control and unchanged parent on **64 new matched complete-game seeds
609100–609163**. A score-training parent must beat both alternatives
on fresh mean with no worse sub-9,000 tail, then pass a separate
predeclared fresh confirmation; it never automatically replaces the
protected 10,480-point replay. Preserve all full checkpoints, optimizer
and RNG states, evaluation records, verified local replays and source
hashes even if negative. Honor the unchanged 5 GiB disk safeguard.

## Zero-update integration parity

The [saved treatment initializer](treatment-initial/latest/state.json)
and [control initializer](control-initial/latest/state.json) both copy
every base model parameter exactly from the frozen run-119 source.
On two [recorded parity seeds](initial-parity.json), **610000** and
**610001**, all three policies produced identical full original-boot
screens, chosen actions, displayed-score rewards and final outcomes
(2,518 / 2,496 actions). The action-memory feature therefore changes
neither initial play nor the original-game protocol. The complete
initial model/optimizer/RNG states are retained separately from the
planned training smokes.

The full **514-test** repository regression suite passed with the new
architecture, including saved-policy native replay and training resume.

## Integration gate passed

Separate [own-action](treatment-smoke/) and [screen-only](control-smoke/)
16,384-action smokes resumed their corresponding zero-update initializers.
Both saved complete model/optimizer/RNG states, made finite GRU/residual
updates while retaining every frozen base parameter exactly, and
published independently original-boot-verified learned replays
(**2,528 / 2,502** actions). Their ten complete fixed-game means were
**10,444 / 10,434**, all stage one and safely above the 5,000-point
integration threshold. All four initializer/smoke archives matched
their run files by hash after copying. The 262,144-action production
arms are therefore authorized, restarting from the two saved
zero-update initializers rather than either smoke model.

## Frozen first fresh comparison and confirmation rule

The production treatment completed both checks at **10,430 / 10,466**;
control completed them at **10,466 / 10,184**. Every one of the **40**
fixed games remained in stage one. Frozen selection therefore takes
treatment [262,144](treatment-full/step-000000262144/) and control
[131,072](control-full/step-000000131072/), both with 10,466 fixed mean.
Both full run archives were copied with matching file hashes.

The predeclared first independent 64-game comparison on seeds
**609100–609163** is complete:

| Policy | Mean / median / best | Below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Own-action treatment](fresh-treatment-64.json) | **10,412.97 / 10,480 / 10,480** | 1 | 0 |
| [Screen-only control](fresh-control-64.json) | **10,452.50 / 10,460 / 10,480** | 0 | 0 |
| [Unchanged parent](fresh-parent-64.json) | **10,414.38 / 10,460 / 10,480** | 1 | 0 |

Treatment failed the first score-parent gate: its mean is below both
the control and parent. The control provisionally passes it by mean
and low-tail count, though it won only 21 versus 19 paired parent
games, with 24 ties. The [treatment](fresh-treatment-replay/replay.html)
and [control](fresh-control-replay/replay.html) best-effort fresh
replays are separately original-boot-verified; neither shows a later
stage. Before any score-parent designation, freeze a **second**
previously unused 64-game set, seeds **609300–609363**, and evaluate
the same three unchanged checkpoints. Require the control again to
beat both treatment and original parent on mean with no larger number
of sub-9,000 games than either. A failure ends this arm as a negative
or inconclusive score result. A pass permits only a score-training
parent designation, never global-best replay promotion or a stage
claim. Preserve all three complete-game tables either way.

## Second fresh confirmation and disposition

All three frozen policies completed the independent **609300–609363**
set without truncation:

| Policy | Mean / median / best | Below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Own-action treatment](confirm-treatment-64.json) | **10,457.19 / 10,480 / 10,480** | 0 | 0 |
| [Screen-only control](confirm-control-64.json) | **10,424.06 / 10,460 / 10,480** | 1 | 0 |
| [Unchanged parent](confirm-parent-64.json) | **10,412.81 / 10,460 / 10,480** | 1 | 0 |

The screen-only control failed the frozen confirmation rule: it lost
to the own-action treatment on mean and low-score tail. In paired
games it beat treatment **15** times, lost **36** and tied **13**;
against the parent it won **18**, lost **33** and tied **13**. The
own-action treatment improved on this second set but had already
failed the *first* score-parent gate by narrowly trailing the parent
and more clearly trailing the control. The reversed set ordering is
an inconclusive score result, not grounds to rewrite checkpoint
selection or run another unchanged confirmation. None of the **384**
fresh games across both sets entered stage two.

Both [treatment](fresh-treatment-replay/replay.html) and
[control](fresh-control-replay/replay.html) first-set best-effort
replays independently reproduced all **2,574 / 2,505** learned
actions from original boot. Each scores exactly **2,620 on all four
lives** at the recurring first-stage barrier. The new own-action
feature worked technically, preserved initial parent behavior and
trained without hidden-state information, but did not yield a
verified passage or confirmed score parent at this bounded gate.
Keep all full states and records; do not promote a new global replay.
