# Long visible-approach subspace search

This completed native run started from the screen-only pilot candidate after a
separate 32-game-per-policy paired check showed a 380.63-point mean gain on
fresh training seeds. The pilot's visual encoder and value network were
verified identical to the original policy, and the run rechecks that exact
encoder lineage before constructing proposals. The initial untouched
ten-game validation mean is 10,266, all stage 1. No earlier checkpoint or
global best replay is overwritten.

The proposal subspace uses only the original policy's own 48 hash-verified
rendered-screen histories at 128, 96, 64 and 32 decisions before a visible
life-loss marker. A centered early-approach contrast plus four principal
variation axes allow screen-dependent key preferences across the failure
region. Twenty random five-key coefficient vectors are tested with both
signs at stratified radii 0.5–4.0. These axes provide no target action,
route, collision coordinate, hidden-state input, demonstration or extra
reward; the network still acts only from raw visible screen history.

Each of the 40 candidates plays four complete games from boot on shared
fresh training seeds. The top four face the incumbent over 16 separate
paired games; an update requires at least 150 mean displayed points of
improvement there and on 32 further fresh paired games. Validation seeds
10000–10009 are excluded from optimization. Any new best complete-game
rank is independently replay-verified before promotion. The sole global
collector monitors this run and all 88 prior artifact roots.

There was no generation or wall-clock limit. A graceful checkpoint-boundary
stop followed the unchanged generation-10 validation. A 5 GiB free-space
guard protected existing archives; the complete native run, including
all checkpoints, RNG states, population records and artifacts, is in `run/`.
`fit-source.py`, `context-source.py`, `context-basis.npz`,
`collector-source.py` and `collector-config.json` preserve the exact
initial implementation, visual proposal basis and monitoring scope.

At generation 5, the run has played **1,264 complete training games** and
used **3,098,088 neural actions**. No candidate passed both score gates;
one generation-4 confirmation gain was 149.375 points, just below the
predeclared 150-point cutoff, and was rejected. Fixed ten-game validation
mean remains 10,266, all stage 1. `milestone-000005/` preserves the exact
restorable model, RNG state and per-game validation record. Every archived
checkpoint file matched the live checkpoint by SHA-256. The global best
replay remains unchanged.

The run finished at generation 11 with **2,832 complete training games**
and **6,940,989 neural actions**. No update passed both gates, no game
reached stage 2, and fixed ten-game validation mean stayed 10,266 at
generations 0, 5, 10 and 11. The final full model/RNG checkpoint is at
`run/generation-000011/`. A checksum-based tree comparison found no
content differences between the live and archived run; the exact source
and final model hashes also matched. The strongest rejected generation-4
and generation-8 candidates are retained in their population plans for
a separate, larger fresh-training-seed paired check. That check must not
be confused with an accepted update or a new verified replay.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-subspace-81 --generations 0 \
  --direction-mode failure-subspace-key --directions 20 \
  --sigma .5 --sigma-max 4 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 200000 --seed 227 --eval-every 5
```
