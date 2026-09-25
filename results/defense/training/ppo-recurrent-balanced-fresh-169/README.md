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
