# Verified early screen-history source

The recurring stage-one loss may require a control change well before the
last 64–128 actions. This archive contains only four 4-frame rendered-screen
histories per life, sampled 370, 350, 300 and 256 decisions before each
visible life-loss marker. Twelve fresh complete training games from the
independently confirmed pilot-80 policy yielded 48 eligible lives. Each
whole trace was reproduced by the frozen policy before its screens were
retained; the index records trace and checkpoint hashes. The individual
NPZ files contain only `screens`, never emulator memory, native snapshots,
actions, target directions or reward labels.

The source is proposal data, not policy input. A frozen visual encoder
turns these same observations into a search subspace; the actual agent
continues to observe only its normal screen history, and independent
complete-game displayed scores decide whether any proposed weight change
is kept. A visible loss marker is used offline to choose screen examples;
it is not available to the acting policy or used as a local fitness signal.
All games in this archive ended in stage one. No new capability is claimed.

`index.json` contains per-game verification and the hashes for the exact
source checkpoint and `source.py`. `seed-*-life-*.npz` and matching JSON
files preserve the screen histories and metadata. Seeds 290000–290011
are training-only; fixed validation seeds 10000–10009 were not used here.

```bash
venv/bin/python -u -m rl.defense_ars_early_source \
  --checkpoint results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --output runs/defense-ars-early-source-89 \
  --games 12 --first-training-seed 290000
```
