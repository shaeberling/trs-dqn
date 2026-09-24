# Longer continuation of per-life exploration

The first persistent-bias-noise PPO trial selected its 8,801,024-action
checkpoint on fixed complete-game validation and retained the full model,
optimizer, policy RNG and per-life noise RNG. It produced more exact-10,480
games on two independent fresh sets, but no stage-two reach and no confirmed
mean-score advantage over its parent. This continuation asks the narrower
question whether substantially more **own screen/score experience** with
that temporally coherent training exploration can discover a behavior beyond
the repeated obstacle. It is not a new score parent or a claimed success.

Run 119 resumes [that exact full state](../ppo-persistent-noise-118/milestone-000008801024/state.json)
and all its inherited settings: 16 workers, four boot-only, PPO with per-life
actor-bias noise standard deviation 1, 128-decision own-visible-loss training
resets, screen-only observations and displayed-score reward. Emulator episodes
restart on optimizer resume, and own-state archives refill from new play.
Unperturbed complete games are evaluated every 131,072 actions, for **16
checks across 2,097,152 new actions** to counter 10,898,176. Stage
progression is only measured in evaluation, not used to select training
resets or reward actions. If a native mission or higher-ranked result appears,
the sole independent collector replays it before promotion. Otherwise the
global verified best remains untouched.

At the predeclared endpoint, select a checkpoint by fixed ten-game mean,
then independently compare it with the frozen parent on 128 new matched
complete games, seeds 601400–601527. Score gains alone will not count as
barrier passage. The full optimizer/RNG and local replays will be archived
regardless of outcome.

The first [full optimizer/RNG milestone](milestone-000008932096/state.json)
is preserved after 131,072 new actions. Ten ordinary complete games
averaged **9,933**, median **10,330**, best **10,430**, all stage one.
This is an early regression from the frozen parent, not stage progress;
the longer predeclared continuation remains active.

The second [full optimizer/RNG milestone](milestone-000009063168/state.json)
is preserved after 262,144 new actions. Ten fixed complete games averaged
**10,468**, best **10,480**, all stage one. This is a reused-seed score
rebound, not an independently confirmed improvement or barrier clear.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-persistent-noise-long-119 \
  --artifacts runs/defense-ppo-persistent-noise-long-119/artifacts \
  --resume results/defense/training/ppo-persistent-noise-118/milestone-000008801024 \
  --steps 10898176 --eval-every 131072
```
