# Full-action, screen-only recurrent Defense integration gate

## Why this is separate from trial 169

The active fresh recurrent comparison groups nine stage-one forward-fire
aliases into canonical Space and gives its treatment GRU the policy's own
previous key. That is a useful stage-one mechanism test, but it removes
the original game's moving-while-firing combinations in stages two/three
and is not literally screen-only under the strict reading of `GOALS.md`.
This trial prepares a policy with all **20** original keyboard actions
available on every screen and **only four raw video-memory frames** as
external input. Its GRU carries learned screen history, no explicit
previous action, game RAM, geometry parser, route, demonstration, replay
trajectory, scripted controller or extra reward.

A zero-update, fresh feedforward PPO initializer receives the existing
direction-neutral `-log(9)` bias on each of the nine forward-fire rows.
It is saved under the canonical profile solely to invoke the already
tested balanced initializer. A fresh ordinary full-20-action recurrent
learner then copies that actor/CNN/value base via `--initialize-policy`,
starts a new Adam state and zero-output GRU residual, and samples **all
20** actions without grouping. At initialization, the nine aliases have
the combined probability of one non-fire command in stage one; unlike
canonical evaluation, each distinct action remains selectable in later
stages. The source has **zero training actions**, so this is a fresh
prior, not a learned-policy transfer or demonstration.

This is an integration gate, not permission to launch another long
score-only PPO run while trial 169's predeclared treatment/control
comparison is incomplete. It does not claim that balanced prior or
screen-only memory solves the repeated course barrier. The historical
screen-only recurrent transfers and long-return PPO continuation were
negative; merely repeating those settings is not a new passage theory.

## Frozen gate

Use seed **41**, 100,000 T states per action, screen stride **1**, score
difference ×0.01 as the only reward, visible life-terminal learning
boundaries, 128 GRU units, 32-step sequences, four workers, 256-step
rollouts, 512-action minibatches, four epochs, learning rate 0.00025,
entropy 0.01, gamma 0.997 and GAE lambda 0.95. Keep the exact original
game and from-boot complete-game evaluation. Run only **16,384** own
training actions before review, with one fixed ten-game checkpoint on
seeds 10000–10009. Require finite model/optimizer/RNG state, all twenty
action probabilities nonzero on a reset screen, strict screen-only
architecture/config, ten complete evaluations and an independently
reloaded original-boot replay. Do not promote on integration score.

If the gate passes, preserve the source initializer, full recurrent
optimizer checkpoint, compact metrics and verified replay. Wait for the
trial-169 treatment/control fresh comparison before choosing any larger
budget. A larger run would need its own predeclared stage-first selection,
fresh complete-game confirmation and disk guard. The protected global
best remains unchanged unless a better *eligible learned* original-boot
replay is independently verified.

## Completed integration gate

The [zero-update balanced feedforward source](source-initializer/state.json)
and [zero-update screen-only recurrent initializer](recurrent-initializer/state.json)
were saved with full provenance. All **12** feedforward parameter arrays
were elementwise identical to the corresponding recurrent base arrays. On a
fresh reset screen, the recurrent initializer assigned nonzero mass to
all **20** actions (minimum **0.00925**); the nine forward-fire aliases
summed to **0.08347**, approximately one of the twelve distinct
stage-one command groups. The archived smoke replay actually sampled
all **20** keyboard actions, including **118** direction-plus-Space
choices. The source had zero training actions. The
recurrent config has `canonical_fire=false`, `recurrent_own_action=false`
and architecture `screen-cnn-residual-gru-v1`; its only external
observation is four raw 16×64 video-memory frames. The archived source
and recurrent initializer model SHA-256 values are
`10bffdd4490f328dc5d6abf4d8ba58b906bd1ea6a83579b42171d15a43bcde0c`
and `ae3f06980acb23353971c414357cd4850caef7e05d43aec9d1fd07683002f0d2`.

The [16,384-action smoke](smoke/checkpoint/evaluation.json) stopped
normally. Its ten complete original-boot games averaged **274** points
(median 280, best 300), all stage one. The final checkpoint contains
finite values in all **42** saved optimizer arrays plus policy RNG state;
its model SHA-256 is
`edbdc9881bdf0da7584691f4ba9df86c8642315e220fb723231c91caf8fdd63c`.
The [300-point replay](smoke/replay/replay.html) was reloaded and
independently reproduced from original boot twice, including the archived
copy: all **1,534** neural actions, displayed-score rewards, frames and
terminal result matched. The source, recurrent initializer, selected
optimizer checkpoint, [full compact metrics](smoke/metrics.jsonl),
[resume configuration](smoke/resume-config.json), status and replay bundle
were copied from their stopped local runs and compared byte-for-byte.

This passes the *plumbing* gate and proves neither stage passage nor an
advantage over trial 169. No production run starts until that trial's
frozen two-arm comparison and fresh-game report are reviewed. The
protected 10,480-point best model and replay remain unchanged.

The exact commands used for the initializer and smoke (paths may be
changed for a fresh reproduction) are:

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-screen-full20-balanced-170-source \
  --artifacts runs/defense-ppo-screen-full20-balanced-170-source-artifacts \
  --initialize-only --canonical-fire --balanced-canonical-init \
  --seed 41 --envs 2 --rollout 256 --batch-size 512 --epochs 4 \
  --learning-rate .00025 --entropy .01 --gamma .997 --gae-lambda .95 \
  --reward-scale .01 --tstates 100000 --eval-every 16384 \
  --eval-games 10 --eval-envs 2 --mlx-cache-mb 256

venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-screen-full20-balanced-170-initial \
  --artifacts runs/defense-ppo-screen-full20-balanced-170-initial-artifacts \
  --initialize-only \
  --initialize-policy runs/defense-ppo-screen-full20-balanced-170-source/latest \
  --recurrent-hidden 128 --sequence-length 32 --life-terminal \
  --seed 41 --envs 4 --rollout 256 --batch-size 512 --epochs 4 \
  --learning-rate .00025 --entropy .01 --gamma .997 --gae-lambda .95 \
  --reward-scale .01 --tstates 100000 --eval-every 16384 \
  --eval-games 10 --eval-envs 4 --mlx-cache-mb 256

venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-screen-full20-balanced-170-smoke \
  --resume runs/defense-ppo-screen-full20-balanced-170-initial/latest \
  --artifacts runs/defense-ppo-screen-full20-balanced-170-smoke-artifacts \
  --steps 16384
```
