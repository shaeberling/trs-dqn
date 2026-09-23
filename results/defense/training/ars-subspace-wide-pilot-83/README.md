# Wider own-screen subspace and radius pilot

This one-generation native pilot tested twelve principal variation axes
from the original policy's own 48 verified visible pre-loss screen histories,
plus an earlier-onset mean contrast. It started from the independently
confirmed pilot checkpoint. The visual encoder was verified identical to
the screen-source policy; no native snapshot, route, action target,
collision coordinate, demonstration or extra reward entered the policy.
All fitness was displayed score in complete games from boot.

Twenty symmetric physical-key directions were tried with both signs at
stratified radii 1–8. In the 40 one-game screening trials, displayed
scores ranged from 360 to 10,460. The strongest screen-score candidates
were at radii roughly 1–2.4; several radii above 4 collapsed to 360–480.
The best candidate lost 20 mean points in the separate two-game comparison,
so there was no confirmation, accepted update or stage-2 game. The ten-game
validation mean stayed 10,266, all stage 1. This pilot suggests that the
wide end is destructive to the earlier route, not that twelve visual axes
cannot help at a more conservative radius.

The full pilot, including model/RNG checkpoints, exact candidate plan,
all complete-game results, validation and replay artifact, is here.
`fit-source.py` and `context-source.py` match the recorded source hashes.
The verified global best replay remains the prior 10,480-point game.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-subspace-wide-pilot-83 --generations 1 \
  --direction-mode failure-subspace-key --subspace-components 12 \
  --directions 20 --sigma 1 --sigma-max 8 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 230000 --seed 229 --eval-every 1
```
