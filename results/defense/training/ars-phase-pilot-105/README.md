# Late own-screen phase-action pilot

This pilot tests a new, screen-conditioned proposal basis without changing
the game, reward, observation, frozen visual encoder, or action set. It
uses the previously verified [own rendered-screen source](../ars-bottleneck-source-92/README.md):
46 high-score stage-one lives, sampled at 96 / 64 / 32 decisions before
their visible loss. Two affine frozen-feature axes are formed from the
earlier→middle and independent middle→late screen changes. On this own
source, their mean projections are approximately `(0, 0)`, `(1, 0)` and
`(0.25, 1)` at those three offsets. These are **visual proposal axes**;
they prescribe no key, route, target action, hidden collision label or
reward. The policy still sees its normal four rendered screen frames.

Twenty symmetric directions cover every command row once. The model is
updated only if a candidate gains at least 150 displayed points in fresh
whole-game score gates. Fixed evaluation seeds 10000–10009 are never
training or selection data. The [complete pilot run](run/) played **54
complete training games and 136,333 actions**. Its best two-game comparison
gain was only **50 points**, so no update was accepted. The fixed ten-game
mean stayed **10,388**; every training and validation game ended in stage
one. The independently verified global 10,480-point replay was unchanged.

The exact [search source](fit-source.py), [phase basis source](phase-source.py),
[head arithmetic](search-source.py) and [verified screen loader](early-context-source.py)
are preserved with the population plan, model/RNG checkpoints, full-game
scores, context-basis bytes and local verified replay. Their hashes match
the run configuration. The larger test uses more symmetric directions and
fresh complete-game gates; it does not promote this pilot's outcome.
The full **452-test** repository suite passed before the larger run.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-phase-pilot-105 --generations 1 \
  --direction-mode bottleneck-phase-action-row --subspace-components 2 \
  --directions 20 --sigma .5 --sigma-max 4 --shortlist 6 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 520000 --seed 309 --eval-every 1
```
