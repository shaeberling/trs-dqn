# Persistent per-life exploration at the recurring barrier

The longer/nearer own-loss reset and longer-GAE tests all failed to pass the
visible stage-one barrier on hundreds of fresh complete games. They altered
practice allocation or return credit, but did not create successful
screen-conditioned behavior. This trial instead perturbs the learned PPO
actor's **output biases** by independent Gaussian offsets of standard
deviation 1, fixed across each training ship life and redrawn at visible
life loss. That can produce temporally coherent action preferences while
the base network still responds to screens. The PPO update uses the exact
sampled offset in its categorical likelihood, ratio, entropy and KL.
Noise is absent from evaluation and replay. It is not a steering rule,
route, collision signal, extra reward, demonstration or hidden-state input.

Run 118 resumes the same full model, optimizer and policy RNG as the
[confirmed ordinary-action score checkpoint](../ppo-canonical-control-113/milestone-000008538880/state.json).
Apart from the opt-in training-only bias noise, it matches the completed
[128-decision own-loss control](../ppo-early-loss-control-115/README.md):
16 workers, four protected full-boot workers, displayed-score reward,
128-decision own-loss resets, 256-step rollouts, entropy 0.002 and
524,288 new actions. Its per-life noise RNG is recorded in the run state.
This setting was previously tested on an older PPO lineage without a
stage clear; combining it with the current stronger parent and focused
practice is a distinct, unproven experiment.

Four fixed ten-game complete evaluations select a checkpoint before a
128-game fresh comparison on seeds 601000–601127 against the unperturbed
parent and ordinary control. A native visible stage transition is required
for a passage claim. The sole independent collector monitors this run's
isolated artifacts and never displaces a higher-ranked verified replay.

The first [full optimizer/RNG milestone](milestone-000008669952/state.json)
is preserved after 131,072 new actions. Ten unperturbed complete games
averaged **10,213**, median **10,440**, best **10,460**; all stayed in stage
one. No noise-assisted training score is being mistaken for an ordinary
learned-policy result. The planned run and independent fresh check continue.

The second [full optimizer/RNG milestone](milestone-000008801024/state.json)
is preserved after 262,144 new actions. Its ten fixed complete games
averaged **10,466**, median/best **10,480**, all stage one. This is a
stronger reused-seed score than the first check, but no visible barrier
passage or independent improvement has been established.

The run completed all four fixed checks, selecting the second checkpoint
above (mean 10,466) over the final checkpoint (10,464). On the first fresh
128-game set, seeds 601000–601127, its mean exceeded the common parent by
only **20.94** points; all games stayed in stage one. The paired median
change was zero, and a symmetric 10% trimmed mean was approximately **+0.39**
points. Therefore, before treating this as a new score parent, a second
untouched confirmation set is predeclared: **128 more complete matched
games, seeds 601200–601327**, for the same frozen selected checkpoint and
common parent. No further checkpoint selection, parameter update or training
will use either fresh set. Any stage transition still requires a native
screen-observed replay, regardless of score.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-persistent-noise-118 \
  --artifacts runs/defense-ppo-persistent-noise-118/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 128 \
  --curriculum-restored-life-only --life-terminal --policy-bias-noise 1
```
