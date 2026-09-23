# Multi-radius physical-key search

This ongoing native run starts from the unchanged strongest learned
screen-only policy and tests coherent physical-key perturbations at 20
stratified radii spanning 0.012–0.05 each generation. Both signs are
evaluated for every direction, with no preferred key, route, or obstacle
coordinate. Broader candidates can reveal new stage behavior; conservative
candidates can retain the strong original score. Every candidate is a
neural policy acting only on rendered screen history, and every fitness
measurement is displayed score in a complete game from boot. No oracle,
scripted steering, demonstrations, hidden memory, extra reward or
evaluation seed enters the optimizer.

All 40 candidates screen on four common fresh training seeds. The top four
are compared with the frozen incumbent on 16 separate paired seeds. A
candidate must improve mean displayed score by at least 150 points there
and again on a third, independent 32-seed paired confirmation set before
its weights are accepted. Any higher-ranked stage/score game is separately
verified by reloading weights and reproducing every neural action, reward
and screen, regardless of score-gate acceptance. Fixed seeds 10000–10009
are validation only; the shared best replay is promoted only when a new
complete game is strictly better and fully verified.

There is no generation or wall-clock limit. A verified mission or a
graceful checkpoint-boundary intervention stops the loop. A 5 GiB free-
space guard protects the existing archives. Live data are in
`runs/defense-ars-key-multiscale-77/`.

The independently restorable [generation-5 checkpoint](milestone-000005/state.json)
is preserved after 1,264 complete training games and 2,997,896 new actions.
Only one candidate passed the independent 32-game confirmation by then.
The fixed ten-game boot mean was 9,925, slightly below the original
9,981; every game remained in stage 1. Its model SHA-256 is
`64323a15bd14e88ca4f875ac54d5cee2f21c10ec9806b7693a2cfc244022eb99`.
This is a negative validation milestone, not barrier passage or a new
global best replay.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-key-multiscale-77 --generations 0 \
  --direction-mode key-factor --directions 20 --sigma .012 --sigma-max .05 \
  --shortlist 4 --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 160000 --seed 201 --eval-every 5
```
