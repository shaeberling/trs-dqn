# Larger fresh-seed check of two rejected subspace candidates

The completed long visual-subspace search retained two candidates that
looked promising on short training comparisons but missed its 150-point
32-game confirmation gate. This read-only candidate check reconstructed
their heads from the exact saved generation-4 and generation-8 population
plans; both plans were verified to be centered on the unchanged pilot
policy. No checkpoint was modified or promoted.

All three policies played 64 complete games from boot on the same fresh
training seeds 220000–220063 and action-sampling seeds. Displayed final
score was the only comparison measure. The unchanged pilot parent averaged
**10,280.31** points; generation-4 candidate 9 averaged **9,752.66**
(-527.66), and generation-8 candidate 6 averaged **10,111.56** (-168.75).
All 192 games remained in stage 1. Neither candidate met the predeclared
150-point gain requirement, so neither is an eligible parent. Their earlier
near-miss gains were sampling noise on the larger independent check.

`report.json` preserves all per-game scores, stages, seed pairing, plan
hashes and selected candidate. `probe-source.py` matches the recorded
source hash. The probe read no native snapshots, collision coordinates or
route labels and wrote no policy weights. The original pilot checkpoint
and 10,480-point verified global best replay remain unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_plan_probe \
  --parent results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --trial results/defense/training/ars-subspace-81/run/population-000004/plan.npz 9 \
  --trial results/defense/training/ars-subspace-81/run/population-000008/plan.npz 6 \
  --output results/defense/training/ars-subspace-near-miss-82/report.json \
  --games 64 --first-training-seed 220000 --envs 16 --minimum-gain 150
```
