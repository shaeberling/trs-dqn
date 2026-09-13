# Execution plan

The emulator already exists; use its Python API and compile its existing C
library only as required by README setup. Stay on the current branch. No prior
solutions, demonstrations, privileged policy inputs, or scripted paddle control.

User decisions: use the Mac Mini's available CPU/GPU, no wall-clock training
limit, and target clearing level 1 as the first completion milestone. All final
evaluation games must run to GAME OVER (at least ten).

1. Review and validate the existing emulator, screen observations, score and
   terminal detection; measure emulator and neural-network throughput.
2. Implement a screen-only RL environment and replay-based neural agent, with
   explicit reproducible seeds, logs, checkpoints, and resume support.
3. Train from the agent's own experience and score rewards, evaluating periodic
   checkpoints on a fixed validation protocol. Record experiments and changes.
4. Evaluate the chosen trained model on ten or more held-out complete games;
   report mean, median, best score, highest level, and number of level-1 clears.
5. Provide saved weights, watch mode, reproducible commands, and results notes.

## Review findings

- `TRS.boot`, `run_for_tstates`, `Keyboard`, and video RAM expose the necessary API.
- The native emulator has global state: parallel environments require processes.
- The graphics screenshot omits text; video RAM contains score/status text.
- Clean checkout had no `libtrs.so` or installed Python dependencies.
- The host is Apple Silicon with ten CPU cores. Benchmark Metal acceleration
  and CPU execution before committing to the training backend.

## Progress

- [x] Read GOALS.md and README.md; agree scope and completion target.
- [x] Review existing emulator implementation.
- [x] Validate environment and benchmark compute (MLX selected on Apple M4).
- [x] Implement training/evaluation/watch workflows; numerical and environment tests pass.
- [x] Train a model that clears level 1; frozen checkpoint verified by recorded level transition.
- [x] Run final evaluation: 50/50 held-out complete games, mean 44.3, median 45.5,
  best 88, highest level 2, three level-1 clears. Document results and save artifacts.

The final evaluation was expanded from the minimum ten games to 50 before
running it. Training stopped cleanly after the requested first-clear milestone;
the trainer's default, stronger automatic stop of three clears in five validation
games was not required for this milestone. That initial run was stopped.

## Level-5 follow-up (completed)

Approved follow-up: reach displayed level 5 (clear levels 1-4), then report
the selected checkpoint's performance on 100 fresh complete games. Preserve
the level-2 model and published replay. See [LEVEL5.md](LEVEL5.md).

- [x] Add configurable target levels, complete-game level reach rates, and a
  separate best-by-level checkpoint, with tests.
- [x] Measure the original checkpoint on the expanded validation suite:
  20/20 complete, mean 51.5, median 55.5, best 66, level 2 in 1/20.
- [x] Run the initial unchanged continuation (planned cap: 20 million additional
  actions), investigate its terminal-parser failure, and preserve its checkpoints.
  The cap was not exhausted; experiments adapted from the measured results.
- [x] Compare learning rate, then exploration strength, while preserving control
  runs. The lower-rate/lower-entropy continuation produced a frozen level-5 game.
- [x] Freeze the selected checkpoint and evaluate 100 fresh complete games:
  mean 83.32, median 72.5, best 167, highest level 3, 0/100 level-5 reaches.
- [x] Record the 278-point level-5 validation replay and package weights, optimizer,
  evaluation records, archived logs, checksums, and reproduction instructions
  without overwriting the original model or public replay.

The first-reach target is verified, not reliable mastery: the selected model
reached level 5 in 1/20 validation games, but not in the 100-game final test.
Training stopped automatically at the validation target and was not resumed
using the test results. Details and provenance are in [LEVEL5.md](LEVEL5.md).
