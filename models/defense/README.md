# Defense / Obstacle Run learned policy

Training is from scratch using only four consecutive original screen frames.
The original categorical policy has 20 keyboard actions; a separate optional
21-action experiment adds policy-selected Enter. Check the selected bundle's
configuration for its exact action profile. No Breakdown weights, scripted
controller, expert demonstrations or hidden-state inputs are used.

Stable local artifacts (updated only after exact frozen-policy re-execution):

- [Best-effort model weights](../../results/defense/learned/best/model.safetensors)
- [Watch that model's best complete game](../../results/defense/learned/best/replay.html)
- [Model configuration and training counter](../../results/defense/learned/best/state.json)
- [Ten-game validation](../../results/defense/learned/best/evaluation.json)
- [Neural-action verification](../../results/defense/learned/best/verification.json)
- [Version and checksums](../../results/defense/learned/best/manifest.json)

These links refer to one atomically selected, versioned bundle. All previous
promotions remain under `results/defense/learned/versions/`. A best individual
effort is not a fresh-test success rate; use the accompanying full evaluation
to understand consistency. No successful mission has yet been verified.

The first preserved learned checkpoint, after 102,400 training actions, scored
320 in its best validation game (1,617 verified neural actions). Its ten-game
mean was 292, median 290, and all games remained in stage 1. This is an initial
baseline, not a completed training goal.

The next preserved improvement, at 1,133,312 cumulative actions, reached **380**
points with all **1,658** neural actions verified. Its ten-game mean was **358**,
median **360**, still entirely in stage 1. The 20-action run was paused cleanly
at 1,542,912 actions; a fresh learned-Enter experiment is now being evaluated.

See [DEFENSE.md](../../DEFENSE.md) for training, independent evaluation, controls,
reward handling, the original three-stage loop, and remaining completion checks.
Resume training from an original `runs/defense-ppo-*/latest` checkpoint with
optimizer state; the portable best bundle provides weights for evaluation.
