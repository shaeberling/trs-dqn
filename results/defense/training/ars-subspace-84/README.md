# Conservative-radius twelve-axis visual search

This ongoing native run begins from the independently confirmed pilot
policy. Its initial fixed ten-game complete-boot validation mean is 10,266,
all stage 1. The previous twelve-axis radius-1–8 pilot found that its
largest perturbations mostly destroyed the earlier route, so this run
uses symmetric radii 0.5–2.5 while retaining the broader visible-screen
proposal subspace. No earlier checkpoint or global best replay is replaced.

The subspace is derived only from the original policy's 48 hash-verified
own rendered screen histories at 128, 96, 64 and 32 decisions before a
visible life-loss marker. It contains an earlier-onset mean contrast and
twelve principal variation axes from frozen screen features. The encoder
is verified byte-identical to the screen-source policy; native snapshots
are never decoded or read. Random physical-key coefficients define
twenty symmetric directions. No direction, route, target action, collision
coordinate, demonstration or extra reward is supplied to the policy.

All 40 candidates play four complete boot games on shared fresh training
seeds. The top four face the incumbent on 16 other paired seeds, and an
update needs at least 150 mean displayed points of improvement there and
again on 32 further fresh paired seeds. Only displayed game score selects
updates. Validation seeds 10000–10009 are excluded. A genuinely better
stage/score rank requires independent action-by-action replay verification
before global promotion. The sole collector watches this run and all 90
prior artifact roots.

There is no generation or wall-clock limit; a 5 GiB free-space guard
protects all archives. Live complete population records, full model/RNG
checkpoints and artifacts are in `runs/defense-ars-subspace-84/`.
`fit-source.py`, `context-source.py`, `context-basis.npz`,
`collector-source.py` and `collector-config.json` preserve the exact
initial implementation, screen-only basis and monitoring scope.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-subspace-84 --generations 0 \
  --direction-mode failure-subspace-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 240000 --seed 233 --eval-every 5
```
