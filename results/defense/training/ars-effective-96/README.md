# Long effective-key search at the repeated barrier

This is a longer test of the movement-versus-firing action structure
calibrated in [pilot 94](../ars-effective-pilot-94/README.md). In stage
one, Space-containing commands do not translate the ship; the proposal
factorization reflects this audited keyboard behavior. It samples all
physical directions symmetrically with no preferred route or steering
instruction. Its visual proposal basis uses 46 own replay-verified
rendered-screen histories in the high-score failure cluster at 128 /
96 / 64 / 32 decisions before visible loss. The acting policy remains
screen-only with no death clock, native game memory, target action or intrinsic
reward. Only displayed score in complete native games from boot can
select an update.

The parent is the generation-five early-screen checkpoint, selected by
a separate 64-seed paired score test. The later generation-nine checkpoint
did **not** pass a fresh 64-seed comparison against it and is not used.
Each generation tests twenty symmetric directions with 4-game screening,
16-game comparison and 32-game independent confirmation. The same
predeclared 150-point score margin applies. Fixed seeds 10000–10009 are
for whole-game validation and independently verified replay only.
All 96 previous artifact roots remain watched by the sole global
collector, along with this run as source 97. The 10,480-point shared
best replay remains intact until a higher-ranked whole game is verified.

The live run is `runs/defense-ars-effective-96/`; full checkpoints,
population plans, per-game score records and exact source will be
archived here at milestones and completion. A stage-two game or native
mission completion—not a near-ceiling score—is the meaningful advance.
The full repository suite passed **442 tests** immediately before this
run; `fit-source.py`, `context-source.py`, `archive-source.py` and
`search-source.py` match the run's recorded source hashes.

At the [generation-five milestone](milestone-000005/state.json), the run
has played **1,392 complete training games** and **3,478,331 neural
actions**. No update or stage-two game has passed. A generation-four
candidate gained 304.38 mean points in the 16-game comparison but only
16.56 in fresh 32-game confirmation. Generation five's 192.5-point
comparison gain became **148.44 points** on 32 confirmation games,
1.56 points short of the predeclared 150-point gate; it too was rejected.
The fixed ten-game mean remains 10,388 and all games still end in stage
one. The full model/RNG milestone, evaluation and log prefix are
preserved. The collector has not replaced the 10,480-point global best.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-effective-96 --generations 11 \
  --direction-mode bottleneck-effective-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 430000 --seed 277 --eval-every 5
```
