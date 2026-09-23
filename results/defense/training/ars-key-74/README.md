# Coherent physical-key score search

This ongoing native run tests whether the repeated stage-one bottleneck
responds to a more coherent learned change than one output row at a time.
The original best policy's visual encoder is frozen. Each candidate adds
screen-dependent feature weights for physical Up, Down, Left, Right and Space
keys across every command containing those keys. Twenty random directions
are evaluated with symmetric signs; none prescribes a route or a preferred
key. The policy still reads only rendered screen history, and all update
fitness is displayed score in complete games from boot. No oracle,
demonstrations, source snapshots, scripted steering or extra reward enter
training.

Each generation screens all 40 candidates on the same four fresh training
seeds, compares the top four against the unchanged incumbent on another
16 shared seeds, and requires at least 150 mean displayed points of gain
on a separate 32-seed confirmation before an update. A better stage/score
game is independently verified by reloading frozen weights and reproducing
every action, reward and screen, whether or not the update gate passes.
The fixed validation seeds 10000–10009 never select a candidate. The shared
best replay remains separate and is promoted only when verification succeeds.
The run has no generation or wall-clock limit, subject to a 5 GiB free-space
guard; a verified mission or graceful checkpoint-boundary intervention ends
it. Its live data are in `runs/defense-ars-key-74/`.

The independently restorable [generation-5 checkpoint](milestone-000005/state.json)
is preserved after 1,328 complete training games and 3,243,653 new actions.
Only one change passed the stronger 32-game confirmation by that point;
its paired mean score gain was 465. The fixed ten-game boot validation mean
was 10,043, up slightly from the original 9,981 but still entirely stage 1.
The checkpoint model SHA-256 is
`69a8ad1ad65ac8525b65f03f07bed96a562d345f3d3bdcb00f87322b97e6efde`.
This is not a verified passage, mission, or new global best replay.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-key-74 --generations 0 \
  --direction-mode key-factor --directions 20 --sigma .02 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 130000 --seed 171 --eval-every 5
```
