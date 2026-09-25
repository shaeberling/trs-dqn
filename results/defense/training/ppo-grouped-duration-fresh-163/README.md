# Fresh grouped key-duration PPO: independent temporal exploration

The independent balanced-action seed-42 run finished all 16,777,216 actions
without reaching stage two and lost all 256 fresh matched games to the frozen
seed-41 score parent. Simply continuing seed 42 is not justified. Earlier
learned-duration trials copied an already trained actor that strongly favored
one-step choices; a fresh option actor with balanced physical controls has
not been tested. This trial asks whether learning *both* which original key
combination to press and how long to hold it can explore a different route.

The actor sees only four raw 16×64 video-memory frames. It has twenty raw
logits for each hold duration, but fixed log-sum-exp grouping makes the nine
stage-one forward-fire aliases one physical `SPACE` choice, as in balanced
trial 160. The twelve emitted commands are original keyboard combinations.
The initial prior assigns equal mass to every distinct physical command
*within each duration*, with direction-neutral duration masses
**0.88 / 0.08 / 0.03 / 0.01** for holds **1 / 4 / 16 / 64** base actions.
During training only, a **5%** key-marginal-preserving uniform-duration
mixture keeps long holds exposed. Its exact likelihood enters PPO. The
semi-Markov actor credits each completed option for its actual displayed-
score return; the critic sees every base action. The only reward is the
original visible score difference times 0.01. Holds cancel at visible life
loss or game end. There is no demonstration, hand-coded route, hidden-state
input, auxiliary reward, native-state curriculum, or diagnostic search
trajectory in learning. All evaluation starts from the unchanged original
game boot and uses the unperturbed learned policy.

First run a **16,384-base-action** integration smoke from fresh seed **43**,
with 16 original-emulator workers, 256-action rollouts, batch 512, four
epochs, learning rate 0.00025, entropy 0.01, gamma 0.997, lambda 0.95 and
100,000 T-states per base action. Require a finite optimizer update, exact
target stop and full model/optimizer/RNG state; ten complete fixed-seed games
on 10000–10009 with mean at least **200**; an independently native-verified
original-boot best-effort replay; and at least **16 actual 64-action option
starts** recorded in training. If any gate fails, archive the smoke and
diagnose rather than launch production. Smoke weights cannot be selected.

If all gates pass, restart fresh with the same seed and settings for exactly
**16,777,216 own base actions**, retaining full checkpoints every 1,048,576
actions. Each checkpoint gets ten complete fixed original-boot games at
seeds 10000–10009. Select successful mission first, then highest stage,
then fixed-game mean, earliest on a tie. Any stage-two/mission claim needs
independent original-boot neural replay verification. After normal target,
compare the frozen selection against the existing seed-41 selected
[parent](../ppo-balanced-fire-continuation-161/README.md) on two untouched
sets of **128 matched complete games** at seeds **615000–615127** and
**615200–615327**. Verify one replay per arm/set; no score-only tie or
unverified outcome promotes the protected global best. Archive selected and
terminal full states, all fixed evaluations, complete compressed metrics,
fresh comparison and verified replays before pruning live snapshots. An
exact-run disk guard gracefully stops below **5.1 GiB** free. If this
mechanism also remains stage-one-only, do not simply extend it without
reexamining the failure and changing the mechanism.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-grouped-duration-163-smoke \
  --artifacts runs/defense-ppo-grouped-duration-163-smoke-artifacts \
  --learned-durations 1 4 16 64 --grouped-duration \
  --grouped-duration-weights .88 .08 .03 .01 \
  --duration-explore-mix .05 --option-actor-gae --life-terminal \
  --steps 16384 --seed 43 --envs 16 --rollout 256 --batch-size 512 \
  --epochs 4 --learning-rate .00025 --entropy .01 \
  --gamma .997 --gae-lambda .95 --reward-scale .01 \
  --tstates 100000 --observation-stride 1 \
  --eval-every 16384 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512
```

The full **557-test** native suite passed before this smoke. The grouped
option path is opt-in; it does not change prior Defense trainer behavior or
the protected best model and replay.

## Smoke outcome

The fresh seed-43 smoke stopped normally at exactly **16,384** base actions.
All model, optimizer, and RNG checkpoint files exist; every stored optimizer
array is finite. It started **134** 64-action options. The fixed ten complete
original-boot games averaged **254**, best **340**, all stage one. The
[checkpoint](smoke/checkpoint) and [native-verified replay](smoke/verified-replay/replay.html)
are preserved with the full smoke log and status under `smoke/`. All
predeclared gates passed. The smoke checkpoint is *not* a production
initialization or a candidate for promotion.

The full fresh production run is now active at
`runs/defense-ppo-grouped-duration-163`. Its current configuration was
checked field-for-field against the archived smoke, changing only the run
and artifact paths, target action count and fixed-evaluation interval. An
exact-PID 5.1-GiB disk guard is active. The fail-closed
[`rl.defense_grouped_duration_compare`](../../../../rl/defense_grouped_duration_compare.py)
monitor waits for a normal exact-target stop, checks all sixteen fixed
evaluations and full states, independently verifies a later-stage claim if
one appears, then runs both predeclared matched fresh sets and four replay
checks. It never alters training or promotes a model. Its status is in the
live run's `comparison-status.json`.

After byte-comparing the checkpoint, replay bundle, config, metrics,
status and duplicate `latest` against the pushed smoke archive, the two
stopped local smoke directories were deleted (about **23 MiB** by `du`).
Every retained smoke state and replay is recoverable from this branch.
