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

Current standard-policy best: **500 points**, all **1,901** neural actions
verified after reloading the frozen model. Ten complete validation games:
mean **462**, median **460**, all stage 1. This comes from the lower-entropy,
ship-loss-boundary continuation at counter **4,434,688**, with ordinary
temperature-1 sampling. The older versions below remain preserved.

The first preserved learned checkpoint, after 102,400 training actions, scored
320 in its best validation game (1,617 verified neural actions). Its ten-game
mean was 292, median 290, and all games remained in stage 1. This is an initial
baseline, not a completed training goal.

The next preserved improvement, at 1,133,312 cumulative actions, reached **380**
points with all **1,658** neural actions verified. Its ten-game mean was **358**,
median **360**, still entirely in stage 1. The 20-action run was paused cleanly
at 1,542,912 actions. Learned-Enter and self-imitation experiments did not
improve the preserved best; training now continues from this checkpoint
with a lower entropy coefficient. See DEFENSE.md for the separate live paths.

Highest separately preserved sampling diagnostic: **400 points**, all **1,743**
actions verified. [Replay](../../results/defense/sampling-probes/temperature-050/replay.html)
and [weights](../../results/defense/sampling-probes/temperature-050/model.safetensors)
use the same frozen model with explicit **temperature 0.5**. Ten reused validation
games: mean **384**, median **380**, all stage 1. This is not a new trained model
or mission completion, and does not replace the standard-policy links above.
Use `rl.defense_evaluate --temperature .5` to reproduce this sampling variant.

See [DEFENSE.md](../../DEFENSE.md) for training, independent evaluation, controls,
reward handling, the original three-stage loop, and remaining completion checks.
Resume training from an original `runs/defense-ppo-*/latest` checkpoint with
optimizer state; the portable best bundle provides weights for evaluation.
