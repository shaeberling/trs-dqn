# Score-derived aggregate of symmetric visual proposals

The preceding twelve-axis run tested 220 symmetric directions over eleven
generations against the same unchanged screen-only policy. This trial
combined their paired **complete-game displayed-score differences** into
one neural-head direction, then tested four signed step sizes. The source
run, all population centers, plans and screening-score files were
hash-checked; none of those generations had accepted an update. Single-
pair score differences were clipped at 2,000 points before aggregation
to limit outlier influence. No stage, collision position, action target,
route, demonstration, native snapshot or intrinsic reward entered the
proposal or selection.

All 288 new training games were complete from boot: 64 games screened
eight signed proposals, 96 compared the top two with the unchanged
parent, and 128 independently confirmed the nominated winner. The
candidate gained 300 mean points in the 32-game comparison but lost
21.09 mean points in the separate 64-game confirmation. It was rejected
under the predeclared 150-point gate; all games remained in stage 1.
No new weights or replay were promoted. This longer confirmation also
rules out that simply averaging the saved score directions fixes the
shared failure.

`report.json`, all phase game files, `plan.npz`, `config.json`, full
model/RNG checkpoints and `metrics.jsonl` are preserved here.
`fit-source.py` matches the recorded source hash. The original confirmed
pilot checkpoint and verified 10,480-point global replay remain intact.

```bash
venv/bin/python -u -m rl.defense_ars_aggregate \
  --parent results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --source-run results/defense/training/ars-subspace-84/run \
  --generations 11 --output runs/defense-ars-aggregate-85 \
  --screen-games 8 --compare-games 32 --confirm-games 64 \
  --minimum-gain 150 --first-training-seed 250000 --envs 16 --seed 239
```
