# Learning Breakdown

## Level-5 first-reach result

The continuation reached displayed **level 5** (cleared levels 1–4) in a complete
frozen-policy validation game: **278 points**, seed 10016. Training stopped at
17,502,208 PPO actions along the selected lineage, plus 500,000 earlier DQN
actions. The checkpoint was frozen before the reserved final test.

| Policy on seeds 30000–30099 | Complete | Mean | Median | Best | Highest level | Reached level 2 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen level-5-capable PPO | 100/100 | 83.32 | 72.5 | 167 | 3 | 50/100 |
| Uniform random | 100/100 | 1.54 | 1 | 4 | 1 | 0/100 |

**None of the 100 fresh test games reached level 4 or 5**; one reached level 3.
The 278-point replay is a reproducible **validation** showcase, not a held-out
test result or evidence of reliable level-5 play. No further model selection or
training used this final test. Both the test and recording ran without action
limits, using the sampled neural policy without gameplay overrides.

- [Frozen model, validation, test, and checksums](models/breakdown-level5/README.md)
- [Watch the level-5 validation game](results/level5/replay.html)
- [Full 100-game test](results/level5/trained.json) and [matched random baseline](results/level5/random.json)
- [Experiment history and reproduction commands](LEVEL5.md), [curve](results/level5/training-curve.png), and [archived logs](results/level5/logs/)

The existing emulator was retained. A visible GAME OVER parsing mismatch was
repaired, then controlled continuations compared learning rate and exploration
strength. The selected settings were MLX PPO, 32 environments, learning rate
0.0001 and entropy coefficient 0.003. Learning inputs, score rewards and all
network-selected controls stayed within GOALS.md. The original model and
published replay below remain unchanged.

## Original first-clear result

The saved model **clears level 1**. Final evaluation used 50 fixed, complete
held-out games (seeds 20000–20049), with no action limit, no gameplay overrides,
and no checkpoint selection using those games.

| Policy | Complete games | Mean | Median | Best | Highest level | Level-1 clears |
|---|---:|---:|---:|---:|---:|---:|
| Trained PPO | 50/50 | 44.30 | 45.5 | 88 | 2 | 3/50 |
| Uniform random | 50/50 | 1.88 | 2 | 4 | 1 | 0/50 |

This meets the first-clear milestone, **not reliable mastery**: most games still
end in level 1. The final set was expanded from the planned minimum of ten to
50 before evaluating the selected checkpoint; none of these test results were
used to alter the model.

- [Saved model and resume state](models/breakdown/README.md)
- [Full evaluation records](results/trained.json)
- [Watch the 88-point held-out game](results/replay.html) — standalone HTML
- [First verified validation clear](results/level-1-clear.html)
- [Training curve](results/training-curve.png) and [episode logs](results/logs/)

## Approach

The agent uses only four 1 KiB video-memory screens (consecutive by default). A fixed lookup
table renders graphics exactly and encodes visible ASCII cells in a separate
channel. Three convolutional layers and a dueling value/advantage head predict
six key combinations: none, left, right, space, left+space, right+space.
The model selects **all** gameplay controls. There is no ball/paddle tracker,
scripted steering, automatic serve, demonstration data, or non-video RAM input.

The final learner is PPO: clipped policy updates, GAE, a learned value baseline,
and entropy regularization. It was initialized from its own Double DQN training
with prioritized replay, five-step returns, Huber loss, and target networks.
Both use Adam and gradient clipping. The only reward is the change in the on-screen score. Game-over
ends the game. `--life-terminal` optionally ends each training return on a lost
ball (standard episodic-life DQN); it does not reset the game or choose a serve.
The two reserve-ball icons are read from the screen to detect these boundaries.
Do not hard-code three such boundaries per game: the verified Level-6 replay
shows the visible reserve increasing from one to two during play and remaining
there for 4,700 frames before the next decrease. The
[screen observation audit](results/level10/reserve-icon-observation-audit.json)
documents this without assuming a game-internal award rule or changing actions.
Evaluation always plays the full game until GAME OVER. Time-limit truncations bootstrap
from their final observation and are reported separately from complete games.

PPO supports `--observation-stride N`, a positive integer counting actions
between the four input screens. Stride 2 retains seven recent raw screens
and selects offsets -6, -4, -2, 0; the network input remains four screens.
With 50,000-T-state actions this preserves the nominal history span of the
original 100,000/stride-1 configuration. Evaluation, watch and recording
default to the checkpoint's stride, or 1 for legacy checkpoints. Replay
metadata records actual spacing and inspection reconstructs that history;
snapshots preserve intermediate raw frames and reject incompatible spacing.
Changing stride resets PPO's validation-selection records. It does not add
game-state inputs, reward shaping or action overrides. The fixed timing
comparison and its validation gates are documented in [LEVEL10.md](LEVEL10.md).

## Setup on Apple Silicon

From the repository root, with Python 3.12:

```sh
python3.12 -m venv venv
venv/bin/python -m pip install -r requirements-rl.txt
make
```

This compiles the **existing** emulator. The bindings now resolve `libtrs.so`
from the repository, so `LD_LIBRARY_PATH` is unnecessary. The native Makefile
enables `-O3` for faster execution. MLX requires access to the Mac's Metal GPU;
an ordinary local Terminal works. A restricted agent sandbox may need an
approved command with GPU access.

The top-level Makefile compiles the wrapper that includes the existing Z80
core directly, using the opcode sources already tracked in the submodule.
It does not rebuild the unused standalone `libz80.so` or regenerate tracked
opcode files.

## Reproduce the training pipeline from scratch

The delivered policy uses PPO after DQN visual pretraining. This recipe
recreates that method using the corrected screen wrapper throughout. Use fresh
run directories. It is not a promise to reproduce the old, pre-parser-fix
trajectories or attain an exact score at a particular action count.

```sh
venv/bin/python -m rl.train --run runs/repro-pilot --steps 100000 \
  --eval-every 25000 --epsilon-steps 100000 --eval-max-steps 5000

venv/bin/python -m rl.train --run runs/repro-extended \
  --resume runs/repro-pilot/latest --steps 450000 --batch-size 32 \
  --target-every 500 --epsilon-steps 200000 --epsilon-final 0.02 \
  --exploration-repeat 32 --eval-every 50000 --eval-max-steps 10000

venv/bin/python -m rl.train --run runs/repro-life \
  --resume runs/repro-extended/step-000450000 --steps 500000 \
  --life-terminal --eval-max-steps 20000

venv/bin/python -m rl.ppo --run runs/repro-ppo \
  --initialize runs/repro-life/step-000500000/model.safetensors --target-clears 1
```

The final command has no time or action limit and periodically validates until
at least one of five complete games clears level 1. The default without
`--target-clears 1` is the stronger three-of-five target. It can be interrupted and resumed.
To try PPO with no pretraining instead, omit `--initialize`.

To continue your own from-scratch checkpoint toward level 5 using the final
continuation settings:

```sh
caffeinate -i venv/bin/python -m rl.ppo --run runs/repro-level5 \
  --resume runs/repro-ppo/latest --learning-rate 0.0001 --entropy 0.003 \
  --target-level 5 --target-clears 1 --eval-games 20 --eval-seed 10000 \
  --eval-every 250000 --eval-max-steps 40000
```

This has no wall-clock or total-action limit. It stops only after a complete
validation suite contains a level-5 game, or on interruption. A truncated suite
cannot qualify. [LEVEL5.md](LEVEL5.md) records the actual multi-stage comparisons
and checkpoint choices; this shorter recipe reproduces the method, not its
exact historical trajectories. Reserve a new test set before future selection;
seeds 30000–30099 are now published test data.

## Train and inspect progress

```sh
venv/bin/python -m rl.train --run runs/mlx-dqn
tail -f runs/mlx-dqn/metrics.jsonl
```

The default run has no wall-clock or action-count limit: it stops after at least
three of five fixed validation games clear level 1 and all five finish. Pass
`--steps N` to limit total environment actions. Models, target weights, optimizer
state, seeds, settings, and evaluation records are saved every 50,000 actions.
Progress prints every ten seconds; the JSONL file also records every episode.

Plot recorded progress (optional plotting dependency):

```sh
venv/bin/python -m pip install -r requirements-report.txt
venv/bin/python -m rl.report runs/pilot runs/extended runs/life runs/ppo runs/ppo-settled
```

This writes `results/training-curve.svg`, `.png`, and the underlying `.json`.
The plot separates rolling training scores from complete-suite validation
scores and shows the fraction of validation games that completed.

Eight emulator processes for DQN, or 32 for PPO, feed one compiled MLX GPU learner. Each action advances
100,000 Z80 T-states (about 56 ms). A seeded initial no-op delay of 0–200,000
T-states varies the initial phase without modifying game memory. The same
procedure applies to training and evaluation. Validation seeds are 10000–10004;
additional diagnostic validation used 10100–10199. Final evaluation uses
20000–20049, never used to select checkpoints.

```sh
venv/bin/python -m rl.train --run runs/mlx-dqn --resume runs/mlx-dqn/latest
```

Resume restores online/target weights, optimizer, global counters, and the
exploration RNG. **Replay and emulator trajectories start fresh**, with a random
replay warmup; this is continuation training, not a bit-identical restart.
Algorithm/environment flags are inherited from the checkpoint unless explicitly
overridden on the command line. Each run records
its exact configuration in `config.json` (and `resume-config.json`). Ctrl-C or
SIGTERM asks the trainer to save at the next action batch and stop cleanly.

PPO also checks stop requests inside evaluation: cancellation saves the learner
and produces no selectable partial result. While a validation is running,
ten-second progress records show unfinished seeds, step counts, scores, displayed
levels, and serve-wait status. `rl.status` exposes this only for the current suite.
For continuing training, a finite `--eval-max-steps` can prevent a failed policy
from blocking all future updates; any incomplete suite is disqualified. Final
test games and a claimed target replay must still finish at actual GAME OVER.

## Evaluate complete games

```sh
venv/bin/python -m rl.evaluate models/breakdown/model.safetensors \
  --games 50 --seed 20000 --max-steps 0 --output results/re-evaluation.json
```

Evaluation uses the checkpoint's learned policy: greedy for DQN, sampled
categorical probabilities for PPO (with fixed per-game random seeds). There
are no scripted gameplay overrides. `--deterministic` can force PPO argmax as
a separate diagnostic. Evaluation reports every game, the mean, median, best
score, highest level, and number of games clearing level 1. It records the
checkpoint SHA-256. `--max-steps 0` runs each game until the visible GAME OVER
message; finite limits are available for diagnostics, but truncated games are
never counted as complete and make the command exit unsuccessfully.

Optional `--envs 8` evaluates several games in separate emulator processes with
batched neural inference. Each PPO game keeps its original independent random
stream, and output games stay in seed order. Defaults remain serial (`--envs 1`);
random-baseline and watch modes require serial execution. PPO training exposes
the same option as `--eval-envs 8`. Changing this count resets inherited
validation selection records, since batched floating-point inference is not
assumed identical on every possible trajectory. The complete 20-game check at
the level-10 experiment's 18,501,632-action checkpoint matched every historical
serial game record exactly; see [LEVEL10.md](LEVEL10.md) for measurements.

The emulator has an optional stop on a specific **visible screen string**.
This prevents a model-selected held SPACE from dismissing GAME OVER and starting
another game inside one environment step. Normal interactive play has this
stop disabled. It does not steer the paddle or alter game logic.

Changed score/level digits are allowed to finish their sequential Z80 redraw
before reward is parsed, by advancing short 2,000-T-state intervals with the
same model-selected keys until two reads agree. Otherwise a transition such
as 00009 → 00010 can briefly appear as 00019 and invent reward. The settled
screen becomes the next observation. Each transition records the extra HUD
settling time. This parser fix was added after PPO's first 1,109,536 actions;
the unfinished rollout was discarded and training resumed from the earlier
1,101,824-action checkpoint. Reproduction of earlier logs can differ because
the corrected wrapper advances these small extra intervals.

Random baseline, with the same starting protocol:

```sh
venv/bin/python -m rl.evaluate --random --games 50 --seed 20000 --max-steps 0 \
  --output results/random-repeat.json
```

## Watch the learned policy

```sh
venv/bin/python -m rl.evaluate models/breakdown/model.safetensors --watch --seed 20005
```

The existing TRS display opens at original game speed; `--speed 2` doubles it.
The model drives the CPU on the window thread, avoiding an independent CPU
thread racing the policy. After GAME OVER, the display pauses for two seconds
and resets for another game. Close the window to stop.

For a standalone browser replay with pause, seeking, and speed controls:

```sh
venv/bin/python -m rl.record models/breakdown/model.safetensors \
  --seed 20005 --max-steps 0 --output results/replay.html
open results/replay.html
```

The replay embeds the actual video-memory frames, glyph atlas, chosen actions,
seed, and checkpoint hash. It needs no Python or network connection to watch.

## Verification and method references

```sh
venv/bin/python -m unittest discover -s tests -v
# Optional replay artifact/control checks (requires Node.js):
node tests/test_replay.js
```

The tests cover score/status parsing and sequential digit redraws, deterministic
reset, terminal detection with SPACE held, n-step terminal versus truncation
returns, replay sampling, exclusion of incomplete evaluation scores,
gradient-based learning, frozen targets, GAE boundaries, and seeded policy /
saved-weight round trips. The learning tests need Metal GPU access.
The original first-clear milestone passed ten Python tests. The current level-5
release passes **25 tests**, including frozen-artifact checksums, complete seed
sets, independently recomputed statistics, and original-artifact preservation.
The original packaged checkpoint resumed successfully for
one separately saved rollout, and rebuilding the existing emulator reproduced
the 88-point test game exactly. Replay codec and play/pause/seek/restart/speed
logic were checked with a mocked DOM/canvas; this is not a full browser visual
test. The existing live TRS window also passed an open/close smoke check.

- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Dueling networks](https://arxiv.org/abs/1511.06581)
- [Prioritized experience replay](https://arxiv.org/abs/1511.05952)
- [Temporally extended epsilon-greedy exploration](https://arxiv.org/abs/2006.01782)
- [MLX compiled training](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [Episodic-life DQN wrapper reference](https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/common/atari_wrappers.html)

## Experiments and actual results

- Mac Mini: Apple M4, 10 CPU cores, 16 GiB unified memory, MLX 0.32.2.
- A synchronized 64-example convolutional benchmark measured 5.07 ms/update
  for compiled MLX, 6.58 ms for PyTorch MPS, and 39.20 ms for four-thread CPU
  PyTorch. Single-example inference was 0.38 ms, 2.11 ms, and 0.14 ms,
  respectively. These are workload measurements, not end-to-end training rates.
- Initial random baseline: 10/10 complete games; mean 1.9, median 2, best 3,
  highest level 1 (`results/random-10.json`). The final matched 50-game baseline
  is reported above and stored in `results/random.json`.
- Initial pilot: `--run runs/pilot --steps 100000 --eval-every 25000
  --epsilon-steps 100000 --eval-max-steps 5000`. 704 training episodes in 411
  seconds, best training score 6; best fully completed validation mean 3.0
  at 50,000 actions. Other checkpoints stalled between balls, showing the
  need for more learning. Initial episode records' `steps` field is episode
  length; later runs distinguish global `steps` and `episode_steps`.
- Continuation experiment: smaller 32-example minibatches, target sync every
  500 optimizer updates, and temporally extended random exploration (Zipf(2)
  action durations capped at 32 decisions). Random persistence is confined to
  training exploration; evaluated gameplay remains greedy model decisions.
- The extended experiment peaked at a fully completed validation mean of
  13.8 (best 14) at 300,000 actions; 450,000 actions scored 13.4 (best 14).
  Its final training checkpoint is at 474,592 actions; one exploratory training
  game reached 24. It was stopped cleanly to try episodic-life learning.
- Episodic-life continuation starts from the fixed 450,000-action checkpoint.
  Only return boundaries change; input, score reward, six model-selected
  controls, and full-game evaluation remain the same.

Continuation command used:

```sh
venv/bin/python -m rl.train --run runs/extended --resume runs/pilot/latest \
  --batch-size 32 --target-every 500 --epsilon-steps 200000 \
  --epsilon-final 0.02 --exploration-repeat 32 --eval-max-steps 10000
```

```sh
venv/bin/python -m rl.train --run runs/life \
  --resume runs/extended/step-000450000 --life-terminal --eval-max-steps 20000
```

PPO continuation, initialized from the agent's own RL-trained visual network:

```sh
venv/bin/python -m rl.ppo --run runs/ppo \
  --initialize runs/life/step-000500000/model.safetensors
```

Omit `--initialize` to start PPO from random weights. This uses 32 independent
emulators, 128-step rollouts, clipped policy updates, GAE, a learned value
baseline, and entropy regularization. The score remains the only environment
reward. It samples its trained action probabilities in training and evaluation;
there is no random-action override. The initialized head's logits are scaled
by 10 once before PPO training. This reuses weights learned from the agent's own
experience, with no expert, demonstration, or supervised policy labels.
The PPO action counter starts at zero; its DQN initialization represents
500,000 prior actions along the checkpoint lineage. Keep `state.json` alongside
the weights so evaluation knows whether to use a DQN or PPO policy.

PPO resume restores optimizer, RNG, counters, and saved settings, with fresh
emulator episodes. Explicit command-line settings override saved settings:

```sh
venv/bin/python -m rl.ppo --run runs/ppo --resume runs/ppo/latest
```

To continue learning from the delivered checkpoint:

```sh
venv/bin/python -m rl.ppo --run runs/continue --resume models/breakdown
```

The episodic-life DQN experiment stopped at 763,176 actions, with a best fully
completed validation mean of 14.0 (best 15). Later greedy checkpoints still
stalled in some games. PPO's first 602,112 new actions produced a fully completed
validation mean of 14.8 (best 20).

After the HUD parser correction, PPO continued from its last fully optimized
checkpoint in a separate run, preserving the earlier experiment's logs:

```sh
venv/bin/python -m rl.ppo --run runs/ppo-settled \
  --resume runs/ppo/step-001101824
```

PPO's corrected-wrapper run improved from validation mean 29.6 at 2,002,944
actions to 57.0 at 6,201,344. The **selected** checkpoint at 6,402,048 actions
was the first with a verified complete validation game reaching level 2:
mean 50.6, median 50, best 66, one clear out of five. Selection favored the
requested level-clear milestone rather than maximum validation mean.
Including DQN initialization, its lineage contains 6,902,048 training actions.
The run was stopped cleanly at 6,660,096 PPO actions after recording this clear.
The log's `target_met: false` refers to the stricter automatic three-of-five
criterion; it does not negate the verified first-clear milestone.

The exact validation replay visibly changes `LEVEL:00001` to `LEVEL:00002`
at score 60 and ends at score 66. Completion uses the game's displayed level,
not an assumed score threshold from the approximate brick count in GOALS.md.
The final held-out replay (seed 20005) ends at 88, level 2, after 3,648 actions.

Argmax-only PPO diagnostics stalled in some games, so evaluation retains the
trained categorical sampling policy. No serving or paddle heuristics were
introduced. Training on this M4 Mac Mini ran around 1,900 actions/second late
in the run. Historical logs, including weaker checkpoints and the HUD parser
failure, are retained rather than presenting only successful trials.

## Optional self-generated curriculum

PPO accepts `--curriculum-probability 0.5 --curriculum-min-level 2
--curriculum-per-level 8` for a training-only start-state curriculum. The default
probability is zero (disabled). Rebuild the existing emulator with `make` first.
Each training worker stores opaque snapshots only at level entries reached by
its own live policy, then may revisit them on automatic episode reset. It cannot
load states from replays, validation, or test games. All policy inputs remain
four screen frames, every action is learned, and reward is only newly earned
score. See [experiment design and checks](LEVEL10.md#self-generated-start-state-curriculum).

Restored segments are labelled `full_game=false`, logged as `curriculum_episode`,
and excluded from full-game scores and target success. `rl.status` reports their
counts separately. Validation, recording, and final testing still start from
the normal boot screen; they never enable this curriculum. Worker archives are
in memory only and rebuild after resume; model/optimizer checkpoints do not
preserve them or ongoing emulator trajectories.

With `--curriculum-share`, workers additionally receive level-entry snapshots
newly reached by peers in the same training run. Sharing is disabled by default
and requires positive curriculum probability. The scheduler removes opaque
payloads before returning observations/metadata to the learner; receipt does not
change an active game. A later automatic reset may choose a peer entry, and its
recorded source worker/action identifies provenance. There is no archive-file
loading option. Validation never enables this path.

`--curriculum-boot-envs 16 --envs 32` reserves the first 16 training workers
for ordinary from-boot games. They still discover and share archive entries,
but never restore one. The remaining workers use the configured reset
probability. This guarantees at least half the action samples in every rollout
come from from-boot trajectories, regardless of restored-segment duration.
The default is zero (the original mixed-reset behavior on every worker).
Reservation requires shared curriculum and must leave at least one unreserved
worker. This changes only training starts, never policy actions or evaluation.
Startup runtime records show each curriculum worker's reset role. New progress
records include exact `rollout_action_origins` and `training_action_origins`
(the latter is cumulative since this process started, not since the entire
checkpoint lineage began); stop records retain the cumulative counts.

On memory-constrained hosts, `--mlx-cache-mb 512` sets MLX's reusable free-buffer
cache target without changing the network or PPO objective. Excess buffers may
remain until subsequent allocation; this is not an instantaneous memory ceiling.
`-1` (the default) preserves
MLX's default cache setting. Progress and stop records include MLX active, cached,
and peak allocation bytes. These do not include all process or system memory;
also monitor host memory pressure when scheduling concurrent learners.

PPO defers loading MLX until learner construction/main execution so spawned
emulator-only workers do not initialize the GPU backend. A startup
`worker_runtime` record reports each worker PID and whether MLX is loaded;
it performs no emulation step and consumes no gameplay RNG.

## Optional checkpoint-averaging diagnostic

`rl.average` uniformly averages finite FP32 weights from distinct checkpoints
with identical PPO run configurations. Choose the input window before viewing
its validation outcomes, then evaluate the resulting single network separately:

```sh
venv/bin/python -m rl.average \
  runs/level10-curriculum-balanced-lr1e5/step-024002560/model.safetensors \
  runs/level10-curriculum-balanced-lr1e5/step-024252416/model.safetensors \
  runs/level10-curriculum-balanced-lr1e5/step-024502272/model.safetensors \
  runs/level10-curriculum-balanced-lr1e5/step-024752128/model.safetensors \
  runs/level10-curriculum-balanced-lr1e5/step-025001984/model.safetensors \
  --output runs/level10-balanced-lr1e5-average
venv/bin/python -m rl.evaluate \
  runs/level10-balanced-lr1e5-average/model.safetensors \
  --games 20 --envs 20 --seed 10000 --max-steps 100000 \
  --output results/level10/validation-balanced-lr1e5-average.json
```

The output directory must be new. Source weights are unchanged; their hashes,
steps, and equal weights are recorded in the output state. This is evaluation
only: it adds no training actions, inherits no performance claim, contains no
optimizer/RNG state, and PPO rejects it as a resume source. Resume learning from
an original learner checkpoint. This exploratory averaging recipe did not
improve the level-10 experiment: 20/20 complete, mean 112.25, highest level 4.

## Optional own-experience self-imitation

`--sil-updates 4` adds opt-in [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)
after each PPO rollout. It replays only this process's live training observations,
actions, and score rewards. Positive `return - value` weights the policy loss;
a one-sided value loss shares the existing optimizer. No validation/replay files
are imported, and inference remains the same single screen-only network.

Defaults: replay capacity 32,768, retained suffix 2,048 actions per worker,
batch size 512, loss weight 0.1, value weight 0.01, priority exponent 0.6,
priority correction 0.1. These are exposed as `--sil-capacity`,
`--sil-suffix-steps`, `--sil-batch-size`, `--sil-loss-weight`,
`--sil-value-weight`, `--sil-priority-alpha`, and `--sil-priority-beta`.
The default `--sil-updates 0` allocates no replay/collector and consumes no
additional RNG. This is an experimental adaptation, not a proven improvement.

The update count is per whole rollout, not per worker. With 32 workers and
128 steps, four batches of 512 give **0.5 nominal replay draws per new action**;
20 batches give 2.5. Increasing this changes the auxiliary optimization dose
relative to PPO. Priorities affect sampling, not FIFO eviction, and nominal
draw counts do not establish how often a specific transition supplies a
positive-advantage gradient.

Only a real learning terminal (life loss when enabled, otherwise game over)
commits a suffix with exact discounted future score rewards. Old prefix states
may be dropped to bound memory; this does not create a terminal or shorten
the future return of retained states. Any truncated segment is discarded,
including its PPO bootstrap. At 32 workers the default screen-storage bound
is 384 MiB plus small array/queue overhead; model/GPU memory is additional.

`sil_segment` logs are learning-boundary records, not complete-game results.
They expose suffix length, dropped prefix, visible initial/final score, and
new-score sum. Progress records include replay occupancy, origin counts,
positive-advantage fractions, and applied auxiliary updates. An all-zero batch
skips Adam entirely, including its momentum update. Primary PPO remains compiled;
the auxiliary path uses ordinary lazy MLX operations with explicit evaluation.

Checkpoints save model, optimizer, action RNG, and a separate SIL sampling RNG.
SIL replay and pending trajectories are in-memory only and start empty on resume;
unfinished suffixes are never committed at shutdown. This limitation is recorded
as `sil_replay_saved=false`. Existing frozen models remain compatible.

## Optional training-only self-reference penalty

`--reference-policy PATH/model.safetensors --reference-kl-weight 0.1` adds
`weight * mean KL(reference || learner)` to PPO on the current learner's
rollout screens. The reference must be this agent's own RL-trained PPO model
with matching environment version, action timing and screen-history stride.
This adapts the student-trajectory auxiliary objective in
[Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835),
with a fixed weight rather than the paper's population-based schedule.

The reference is frozen and queried once per rollout in bounded batches;
six detached log-probabilities per observation are reused across PPO epochs.
It never selects actions, supplies trajectories or changes score rewards.
SIL's loss is unchanged. All live actions and restored-state discoveries still
come from the learner, and no validation/replay data enters training. The
penalty is experimental: it may help retention or constrain further learning.

Default weight zero with no reference retains the historical update path.
Progress logs expose mean PPO-minibatch `reference_kl` only when enabled.
This is measured during PPO updates, not after the subsequent SIL updates.
Checkpoints retain the
reference path, SHA-256 and weight; resume verifies identity before GPU or
worker initialization. Resuming regularized training needs that reference
file; evaluating or replaying the saved learner does not. Inference remains
the same single screen-only categorical policy, with no reference dependency.

## Continuous local supervision

**Authorized target: beat all eight original levels.** The original game ends
after the eighth; see the [exact-binary audit](results/level10/game-level-cap-audit.json).
The prior Level-10 run stopped cleanly at 120,102,912 actions. The revised
all-eight goal is now verified and its 100-game fresh test is complete; all
processes have exited. See [ALL_EIGHT.md](ALL_EIGHT.md). Reaching Level8
does not count as winning. The CLI now requires `--game-win`.

`rl.outcome` proves a win using only terminal video: `GAME OVER`, level8 and
at least one visible reserve ball (HUD columns30–34). Both loss branches
exhaust lives; the reserve HUD is already empty on the last life. Victory
does not decrement lives. Binary-contract tests verify those paths. This is
a sufficient, conservative proof, not a complete classifier: last-ball Level8
endings are reported as `unverified_final_level`, not asserted losses or wins.
The verified win rate is therefore a lower bound. No live non-video memory,
CPU state, score threshold or policy override is used. The separate
`outcome_version` identifies this reporting; observations/stepping/rewards and
the existing environment version are unchanged.

The local supervisor avoids idle gaps at arbitrary action budgets. It starts
one declared learner with `--steps 0`, keeps a15-second heartbeat, and audits
each new complete primary win/depth/count record on the reused50-game secondary
set. Promotion requires a better complete70-game rank, an uncapped replay,
and exact verification of every recorded neural action. The learner can
continue while a frozen checkpoint is audited. No hyperparameters are tuned
automatically and no teacher/validation/replay trajectories enter training.

Historical launch command for the now-stopped progress4 trial (not an active
run or instruction to restart it):

```bash
caffeinate -i venv/bin/python -m rl.supervise \
  --run runs/all-eight-supervised-progress4 \
  --resume runs/level10-supervised-refkl01-v2/learner/step-116006912 \
  --baseline runs/level10-supervised-refkl01-v2/learner/step-116006912 \
  --baseline-primary results/level10/all-eight-baseline-primary.json \
  --baseline-secondary results/level10/all-eight-baseline-secondary.json \
  --baseline-selection results/level10/all-eight-baseline-selection.json \
  --game-win --curriculum-score-interval 4 \
  --artifacts results/level10/best \
  --effort-artifacts results/level10/best-effort
```

The run directory must be new; check for an existing process before launching
this example, and do not launch a duplicate or reuse a stopped directory.
Check `status.json` in that directory for phase, PIDs, heartbeat, actual latest
learner step/log age, and selected validation results. `learner/metrics.jsonl`
is the normal training log. `selected.json` and `audits/` hold automatic
selection records; `learner-output.log` and per-audit logs retain subprocess
output. A file lock prevents two supervisor instances. An old heartbeat does
not prove a process is still alive; check its PID too.

Create a file named `STOP` in that exact run directory to request a graceful
stop. The supervisor also stops for runtime/identity errors, fewer than5GiB
free disk space,15 minutes without a learner log event, or five consecutive
complete primary suites missing Level5. The log watchdog is a fault detector,
not a wall-clock training limit. It records `needs_attention` with a reason
and preserves checkpoints; it never silently deletes data or changes methods.

Verified full-game wins rank before depth counts or mean score. Only a promoted
checkpoint with a complete70-game audit and an uncapped, action-verified winning
replay can stop learning, freeze exact weights in `frozen-winner/`, and run100 uncapped
fresh games on seeds40000–40099. Those results never select the model. It
records an exclusive final-test-start marker to prevent accidental retries.
The replay's outcome is recomputed directly from its final screen, not trusted
from JSON. A practice-state win never counts as a full-game victory. The
learner continues after a primary win until the supervisor finishes its audit.
The September20 final test used seeds40000–40099 once and completed all100
games naturally. Those seeds are no longer fresh. Any future independent
test requires a newly declared seed set before evaluation, not a retry of
this result. No fresh-test games entered training or model selection.

This process uses the Mac's normal Python/MLX runtime, not a Codex agent or
API. It cannot restore a usage-limited chat goal, send chat updates, or restart
after a machine reboot. Resume automatic agent follow-up with the app's goal
progress controls; the local supervisor continues independently while its
process is alive. The failed first launch (both mutually exclusive budget
flags passed) is preserved in `runs/level10-supervised-refkl01`; it performed
no training. The corrected launcher passes only `--steps 0` and has a real
CLI regression test.

### Progress archives and always-available replays

`--curriculum-score-interval 16` adds training-only snapshots after16 newly
earned visible score points since the last archive opportunity in an eligible
level. Zero (the default) preserves the level-entry-only behavior. Boot,
restore and level change rebase the offset. The bounded per-level reservoir,
same-run peer provenance, fresh score rewards and protected boot workers are
unchanged. Snapshots are never policy inputs or demonstrations. This is the
September19 experiment, now paused without a selected-model improvement.
The user-authorized all-eight trial uses interval4 for finer late-level practice,
with the same reservoir and protected workers; it is not yet proven better.

`--artifacts` exports every verified validation-selected improvement;
`--effort-artifacts` separately records new complete-game level/score records
from primary/secondary validation, even if that checkpoint is not selected.
Both keep immutable `versions/<model-hash>-<replay-hash>/` bundles and a stable
standalone `replay.html`. `current.json` identifies the current version and
checksums. Older versions are retained. Copy a replay HTML file to share it;
it includes its frames, font and actions, with no model or network dependency.
The original public replay is not changed by these local exports.

To inspect a recorded screen without running new gameplay:

```bash
venv/bin/python -m rl.replay_frames results/level10/best/replay.html \
  --frames 44743 50470 52538 --output results/level10/replay-diagnostic.png
```

Frame indices above belong to the September19 selected replay; choose valid
indices from the current replay metadata if the stable file has advanced.
