# Early-screen visual-subspace pilot

This native one-generation pilot tested whether a much earlier
screen-history subspace can change the recurring stage-one approach.
It used the 48 independently replay-verified own screen histories in
`../ars-early-source-89/`, a frozen visual encoder, and twelve principal
variation axes plus an early-approach mean contrast. The source samples
are 370 / 350 / 300 / 256 decisions before visible life loss, rather
than the earlier experiments' 128 / 96 / 64 / 32. The proposal subspace
hash is `2dc184d769e23e8c1ede288fc6dc8ae8d88844a63c01908379c915da6d36491c`.
The actual policy still receives only normal rendered-screen history.

All 54 screening/comparison/confirmation games were complete boot runs
with 134,287 neural decisions. The shortlisted candidate led the parent
by 770 mean points in a separate two-game comparison, but by only 140
in a fresh two-game confirmation. That missed the predeclared 150-point
gate, so the parent weights were preserved. The fixed ten-game mean
remained 10,266; all training and validation games ended in stage one.
These small paired samples do not establish whether the early feature
proposal is useful; the longer run uses larger independent score gates.

The complete population plan, per-game results, model/RNG checkpoints,
config, metrics and local verified replay artifacts are preserved here.
`fit-source.py`, `context-source.py`, and `archive-source.py` preserve the
exact code used to construct and test this pilot. No native state, route,
target action, demonstration or reward shaping was supplied to the learner.
The global 10,480-point independently verified best replay is unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-early-source-89 \
  --output runs/defense-ars-early-pilot-90 --generations 1 \
  --direction-mode early-subspace-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 300000 --seed 241 --eval-every 1
```
