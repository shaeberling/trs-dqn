# Self-play failure-context keyboard search

This ongoing run starts from the original strongest learned screen-only
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

There is no generation or wall-clock limit. A verified mission or a
graceful checkpoint-boundary intervention stops it. A 5 GiB free-space
guard protects all prior archives. Live data are in
`runs/defense-ars-context-79/`.

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
