# Self-play failure-context keyboard search

This completed run started from the original strongest learned screen-only
policy and uses its own 12 complete training games to propose localized
neural-head changes. Only rendered screen arrays are read from 48 verified
source traces: frozen visual features at 128 decisions before a visible
life loss are contrasted with features at 64 and 32 decisions before the
same marker. The contrast is centered on earlier screens and embedded as
a fixed direction in the existing neural action head. It tells the search
*when* a change may matter; it supplies no route, target key, action label,
collision time or hidden game-state input. Twenty random physical-key
coefficient vectors, each tested with positive and negative signs over
0.5–4.0 radii, determine *which* keys change. The policy still receives
only raw rendered screen history, with no separate trigger or oracle.

All 40 candidates play complete games from boot. Four shared fresh
training seeds screen them; the top four face the unchanged incumbent on
16 separate paired seeds. A policy update requires at least 150 mean
displayed points of improvement there and again on 32 further fresh
paired training seeds. Displayed score is the only optimization fitness.
Validation seeds 10000–10009 are never used for selection. Any game with
a better stage/score rank is independently verified from saved weights by
reproducing every neural action, reward and screen before replay promotion.
No demonstration, scripted steering, intrinsic bonus, hidden RAM read or
native snapshot restore is used in this run.

There was no generation or wall-clock limit. A graceful checkpoint-boundary
intervention stopped the run after the generation-10 validation confirmed
a stage-one plateau. A 5 GiB free-space guard protected all prior archives.
The full native run, including every population, candidate comparison,
checkpoint, evaluation, RNG state and replay artifact, is in `run/`.

At generation 5, the frozen-encoder policy has played 1,392 complete
training games (3,407,318 neural actions). Two changes passed both fresh
training-seed comparison sets. Its untouched ten-game validation mean rose
from 9,981 to 10,155 displayed points, but all ten games still ended in
stage 1. `milestone-000005/` preserves the full generation-5 checkpoint,
RNG state, and per-game validation record. `fit-source.py`,
`context-source.py`, and `context-basis.npz` preserve the exact proposal
implementation and computed screen-only basis. All archived checkpoint
files and source hashes were verified against the live run; this is not a
new best replay or a stage advance.

The run finished at generation 11 with **3,024 complete training games**
and **7,410,176 neural actions**. Two generation-1/2 updates passed the
16-game comparison and separate 32-game confirmation gates; none of the
next nine generations did. Held-out ten-game mean was 10,155 at
generations 5, 10 and 11, versus 9,981 initially, and every observed game
remained in stage 1. The short comparisons often overestimated gains that
vanished on confirmation. `run/generation-000011/` preserves the final
restorable model/RNG state; the archived tree was checked against the live
tree using content checksums, with source and checkpoint hashes also
verified. The verified 10,480-point global best replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-context-79 --generations 0 \
  --direction-mode failure-context-key --directions 20 \
  --sigma .5 --sigma-max 4 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 180000 --seed 221 --eval-every 5
```
