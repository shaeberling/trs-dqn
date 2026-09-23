# Verified screens at the recurring failure window

A separate fresh training-seed replay showed why the late intervention
window matters: on its first life, the pilot-80 policy scored 2,600, but
the right-opening barrier was visible while the ship was still far left
roughly 64 decisions before the visible loss. This archive samples the
policy's own rendered screen history 128, 96, 64 and 32 decisions before
each visible life loss. Twelve fresh complete training games from seeds
360000–360011 yielded 48 own-life samples. Every whole trace was exactly
replayed before recording; `index.json` and each sample's JSON preserve
checkpoint, trace and file hashes.

The two life samples with only 140 displayed points are retained in the
archive for audit, but the bottleneck proposal builder excludes them with
a 2,400-point *visible per-life score* cutoff fixed before candidate play. The other 46
own lives cluster near 2,500–2,640 points at the repeated stage-one
failure. This score threshold selects proposal screens; it is not an
action label, local reward or acting-time input. Whole complete-game
displayed score on fresh seeds remains the sole weight-update fitness.

Each NPZ contains only `screens`, not native snapshots, hidden game state,
actions or reward labels. The original game, screen-only policy input,
frozen encoder, stage-one failures and global verified best are unchanged.
`source.py` matches the recorded source hash.

```bash
venv/bin/python -u -m rl.defense_ars_early_source \
  --checkpoint results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --output runs/defense-ars-bottleneck-source-92 \
  --games 12 --first-training-seed 360000 \
  --offsets 128 96 64 32
```
