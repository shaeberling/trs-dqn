# Learning Breakdown

## Result

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

The agent uses only four consecutive 1 KiB video-memory screens. A fixed lookup
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
Evaluation always plays the full three-ball game. Time-limit truncations bootstrap
from their final observation and are reported separately from complete games.

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
All ten Python tests passed. The packaged checkpoint resumed successfully for
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
