# Audited continuation of early-screen run 91

The original long run completed generation eight and was in the
generation-nine confirmation phase when a worker imported a temporarily
invalid code edit and the process exited. This was an operational code
edit/spawn race, **not** a failed game, stage passage, model regression or
disk failure. The complete generation-eight model, RNG, score history
and next training seed were intact: 2,176 complete games, 5,405,520
actions and next seed 310416. The original partial generation-nine files
remain in `runs/defense-ars-early-91/`; none was deleted or promoted.

This continuation restores that exact checkpoint and runs generations
nine through eleven into a new output root. The updated context code
only added an optional score filter with default zero and provenance
fields. Resume was allowed only after verifying the archived old source,
identical rendered-screen file hashes, **identical basis hash**
`2dc184d769e23e8c1ede288fc6dc8ae8d88844a63c01908379c915da6d36491c`
and unchanged diagnostics. `config.json` records this narrowly audited
migration. As an additional reproducibility check, the repeated
generation-nine proposal plan and all 160 screening-game records have
exactly the same SHA-256 hashes as the interrupted prefix. Full boot
displayed score remains the sole update fitness; fixed validation seeds
are evaluation-only.

The sole verified-best collector now watches this continuation and all
95 prior artifact roots. `collector-config.json` preserves its exact
source list. The existing global 10,480-point replay remains available
until an independently verified stronger result exists.

The continuation has now finished through generation eleven. The logical
lineage totals **3,024 complete training games** and **7,527,929 neural
actions**; the duplicate generation-nine screening/comparison games from
the interruption remain archived separately and are not double-counted
in those checkpoint totals. Generation nine accepted one score-gated
change (+351.88 / +219.69 points in 16-/32-game gates), but a separate
[64-seed-per-policy paired check](paired-generation-9.json) against the
previous confirmed milestone found only **+124.22** displayed points,
below the predeclared 150-point margin. Thus the generation-nine model
is *not* selected as the next independent parent. `pair-source.py` matches
the recorded source hash and verifies the frozen encoder/value network.
The fixed ten-game mean was 10,397 at generations ten and eleven versus
10,388 at the confirmed generation-five milestone. Every training and
validation game stayed in stage one. Full plans, whole-game outcomes,
model/RNG checkpoints and local verified replays are in `run/`; the
10,480-point global best is unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --resume runs/defense-ars-early-91/generation-000008 \
  --context-archive results/defense/training/ars-early-source-89 \
  --legacy-context-source results/defense/training/ars-early-91/context-source.py \
  --output runs/defense-ars-early-91-resume-95 --generations 11 \
  --direction-mode early-subspace-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 310000 --seed 251 --eval-every 5
```
