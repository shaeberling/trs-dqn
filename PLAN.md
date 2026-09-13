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
games was not required for this milestone. No training is left running.
