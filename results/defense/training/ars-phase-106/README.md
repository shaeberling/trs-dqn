# Long score-only search over late own-screen visual phases

This active run starts from the last independently confirmed stage-one
policy, not either short-sample action-row update. Its frozen visual
encoder processes only the usual four rendered-screen frames. The
[own-screen phase basis](../ars-phase-pilot-105/README.md) is built from
46 verified high-score lives at 96 / 64 / 32 decisions before their
visible losses. It supplies two screen-feature proposal axes, not a
preferred command, route, action label, collision coordinate or reward.

Each generation tests **two symmetric random directions for every one of
the twenty original commands** (40 directions, 80 candidates). The first
gate screens each candidate in four fresh complete games. Six finalists
are compared with the incumbent on sixteen new complete games, and an
update requires at least 150 displayed points of mean gain in a separate
64-game confirmation. Exact screening-score ties alone are randomized
and diversified across command rows. Fixed ten-game validation seeds
10000–10009 are never used for update selection. Stage progression is
checked as an outcome, not substituted for score fitness.

The live output is `runs/defense-ars-phase-106/`. Full model/RNG
checkpoints, each candidate plan, complete-game scores and locally
verified replays will be archived here at milestones and completion.
The sole global best-replay collector watches this run and all 101
other artifact roots, independently re-executing any prospective best
before promotion. The global verified 10,480-point replay remains the
floor. The full **452-test** repository suite passed before launch;
`fit-source.py`, `phase-source.py` and `search-source.py` match the
hashes in [run-config.json](run-config.json).
The initial full [model/RNG checkpoint](milestone-000000/state.json)
and exact [phase basis](context-basis.npz) are already preserved here.

At [generation five](milestone-000005/state.json), the run has completed
**2,544 full training games and 6,414,937 actions**. None reached stage two.
No proposal has passed the independent 64-game confirmation margin, so
the parent weights remain in force. The fixed ten-game mean is **10,388**,
unchanged from the start; its best score is 10,480. The generation-five
model/RNG state and evaluation are archived without replacing the older
independently verified global replay.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-phase-106 --generations 11 \
  --direction-mode bottleneck-phase-action-row --subspace-components 2 \
  --directions 40 --sigma .5 --sigma-max 4 --shortlist 6 \
  --screen-games 4 --compare-games 16 --confirm-games 64 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 530000 --seed 311 --eval-every 5
```
