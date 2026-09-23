# Stratified key-factor radius pilot

This one-generation native pilot checks a stratified mixture of symmetric
physical-key perturbation radii, from 0.012 to 0.05. Every candidate still
plays a complete boot game and is scored only by displayed points; the
learned policy sees rendered screens, not native state or a prescribed
direction. It is a functional/signal test with deliberately tiny boot
comparison sets, not a generalization claim. The original strong policy
and global verified replay remain separate.

The completed pilot played 50 full training games and 116,838 new actions;
all were stage 1. Its 20 paired radii really spanned 0.012–0.05. One-game
candidate scores ranged from 300 to 10,480, with the highest scores mostly
at the conservative end (about 0.014–0.025). The top candidate's two-game
comparison gain was 905 points, but its separate two-game confirmation
was 1,475 points worse, so the update was rejected. The fixed ten-game
mean remained 9,981. The [complete pilot](run/) and [exact source](fit-source.py)
are archived. This is a working multi-scale population and a reminder that
tiny samples are noisy, not evidence of stage progress. A longer run uses
4 / 16 / 32 fresh boot-game sets and a 150-point final margin.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-key-multiscale-pilot-76 --generations 1 \
  --direction-mode key-factor --directions 20 --sigma .012 --sigma-max .05 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --shortlist 2 --first-training-seed 150000 --seed 191 --eval-every 1
```
