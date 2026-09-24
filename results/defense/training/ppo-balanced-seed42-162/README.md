# Independent balanced-action learning trajectory: seed 42

The balanced canonical-fire initializer materially improved learning from
scratch in trial 160. Its seed-41 PPO continuation in trial 161 gained
**327.66** mean displayed points over its frozen parent on 128 matched
fresh complete games, but all games remained in stage one. Continuing that
same optimizer trajectory again is not the next test. This experiment
instead asks whether a new random initialization and independently sampled
own experience, with the effective balanced physical-action prior, can
discover a different route through the original stage-one bottleneck.
It does not assume a new seed will succeed. Earlier actor-head ARS searches
already explored parameter-space perturbations on other mature policies
without stage-two passage, so this is an initialization-diversity test,
not a repetition of those searches.

Predeclare one fresh seed-42 run to **16,777,216** own training actions
from scratch. Use the original unchanged game, 16 independent booted
emulator workers, four raw video-memory frames per neural input, the fixed
twelve-choice grouped physical action distribution, displayed score
difference times 0.01 as the sole reward, and visible-life learning
boundaries. Keep trial 160's PPO settings: 256-action rollouts, batch 512,
four epochs, learning rate 0.00025, entropy 0.01, gamma 0.997, lambda
0.95, 100,000 T-states per decision. No demonstration, scripted route,
hidden-state input, auxiliary reward, reset curriculum, or searched
diagnostic action enters learning.

Freeze and evaluate a full model/optimizer/RNG checkpoint every 1,048,576
own actions on ten complete original-boot games, seeds 10000–10009.
Select a checkpoint by successful mission first, then highest stage,
then fixed-game mean, earliest on a tie. Any stage-two or mission claim
needs an independent native original-boot replay verification. At target,
compare the frozen selection to trial 161's selected seventh checkpoint
on **128 new matched complete games** at seeds 614000–614127 and a second
untouched 128 at 614200–614327 before claiming a score-parent gain. The
protected global best is not overwritten by a same-stage score tie.
Archive selected and terminal full states, all fixed evaluations, full
metrics and verified best-effort replays before pruning other live
optimizer snapshots. An exact-run watchdog signals a clean stop below
**5.1 GiB** free. If this seed does not reach stage two, do not continue
it merely because the action budget ended; review the result against the
existing negative diagnostics and change the mechanism.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-balanced-seed42-162 \
  --artifacts runs/defense-ppo-balanced-seed42-162-artifacts \
  --canonical-fire --balanced-canonical-init --life-terminal \
  --steps 16777216 --seed 42 --envs 16 --rollout 256 --batch-size 512 \
  --epochs 4 --learning-rate .00025 --entropy .01 \
  --gamma .997 --gae-lambda .95 --reward-scale .01 \
  --tstates 100000 --observation-stride 1 \
  --eval-every 1048576 --eval-games 10 --eval-envs 10 \
  --mlx-cache-mb 512
```

No trainer or environment code changes are needed. The complete native
537-test suite passed earlier; the focused four-test continuation-watcher
suite also passed. This run uses the same tested trainer path and a fresh
new run/artifact directory.
