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
