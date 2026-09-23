# Wider physical-key perturbation pilot

This one-generation native test widens the symmetric screen-dependent
physical-key perturbations from 0.02 to 0.05. The original strong learned
policy remains the parent. All 40 candidates play complete boot games,
using displayed score only; no oracle, route, demonstration, or hidden
game memory enters training. Tiny comparison/confirmation samples test
software and score range, not generalization.

The completed pilot played 46 full training games and 91,442 new actions;
none reached stage 2. Screening scores ranged from 280 to 7,750 on its
single shared seed, and the two-game comparison rejected the best
shortlisted candidate, 8,180 versus 8,860 for the incumbent. The model
hash remained the original parent's; fixed ten-game validation stayed at
9,981. Full states, population scores and RNG are [archived](run/).
On 48 recorded visible pre-loss screens, these wider candidates' median
movement probability ranged from near zero to about 0.997, showing that
the larger radius can materially change behavior but is often destructive.
The next test mixes radii within each generation so it can explore widely
without discarding the conservative candidates.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-key-wide-pilot-75 --generations 1 \
  --direction-mode key-factor --directions 20 --sigma .05 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --shortlist 2 --first-training-seed 140000 --seed 181 --eval-every 1
```
