# Learned continuation of the policy's own prior key

The run-119 score parent repeatedly loses near the visible right-opening
barrier. A frozen wider-frame-history probe and 524,288-action adaptation
recovered ordinary stage-one scoring but did **not** cross it; the fresh
adapted model regressed. The next hypothesis is that a one-action-at-a-time
categorical policy lacks a convenient way to sustain a movement choice.

Run 121 adds one **learned** choice, `CONTINUE_PREVIOUS`, alongside the same
twenty ordinary keys. It repeats only the policy's own last executed key,
initially `NOOP`, and resets that one-key memory on visible life loss or game
boundary. The neural policy receives only its four raw 16x64 visible frames;
the executor never inspects obstacles, selects a direction, injects an action
on an outcome, reads hidden game RAM, or uses a demonstration. PPO computes
the likelihood of the **sampled 21-way choice**, while the emulator executes
its corresponding ordinary key. This differs from the earlier DQN learned
multi-step holds: every step remains a policy decision and an ordinary
100,000-T-state transition, so the existing one-step PPO return math applies.

The run begins from the **full own screen-policy weights** of the independently
confirmed run-119 score parent, preserving all twenty learned actor rows,
encoder and value head exactly. The new row is the arithmetic mean of the
twenty existing actor rows, with no favored key. The optimizer, action RNG,
noise RNG, episode counters and training experience are fresh. The
resumable source checkpoint remains untouched. Training keeps the run-119
PPO settings, per-life training-only output-bias noise, own-visible-loss
rewinds, four boot-only workers, screen-only observations and displayed-score
reward. Stage progression is evaluated only; it does not steer actions or
select curriculum states.

A no-learning two-worker initialization smoke check completed. Its frozen
21-choice policy averaged **10,450** points on ten complete fresh games,
all stage one. A separate 64-action PPO-update smoke completed ten native
games, saved a full checkpoint and independently re-executed a **2,537**-
action 10,480-point stage-one replay. Neither smoke run is a score parent or
a collector source. Transfer tests prove the twenty original logits and the
value head are unchanged at initialization. The full regression suite must
pass before the production run starts.

The planned first gate is **1,048,576 new base actions**, eight complete
ten-game unperturbed checks at 131,072-action intervals. Select the
checkpoint with highest fixed mean score, breaking exact ties in favor of
the earliest. A verified stage-two or mission game warrants immediate
independent confirmation. Otherwise compare the selected checkpoint to the
unchanged run-119 parent on **128 new matched complete games** only if its
fixed ten-game mean reaches 10,450; below that, archive the full negative
pilot without promoting a score parent. The global native-verified best
replay remains independent and can change only after a strictly higher-ranked
candidate is re-executed from frozen weights.

The planned eight checks completed at **1,048,576** new actions, all stage
one, with fixed ten-game means **10,148 / 10,426 / 10,388 / 10,450 /
10,114 / 10,314 / 10,328 / 10,472**. The final saved checkpoint is selected
under the rule above. Its first independent 128 complete games, seeds
**602500–602627**, averaged **10,464.14** versus **10,434.69** for the
untouched run-119 parent: +29.45 displayed points, 57 paired wins, 15
losses and 56 ties. Both remained stage one. The new model had 104 exact
10,480 scores versus 61 for the parent, but each had one sub-9,000 game.
Its local best replay independently reproduced the frozen learned policy;
the shared global best remains unchanged.

Before any additional evaluation or checkpoint selection, a second untouched
matched confirmation set is fixed as seeds **602700–602827** (128 complete
games per frozen policy). The selected checkpoint and parent will not change.
Only the combined 256-game comparison can support a new *score* parent; even
that would not establish barrier passage or mission completion.

The second predeclared set gave **10,445.86** selected versus **10,422.34**
parent, +23.52 points, with 57 paired wins, 14 losses and 57 ties. Across
both untouched sets, [256 complete games per policy](comparison.json), the
selected policy averages **10,455.00** versus **10,428.52** (+26.48), wins
114 paired games, loses 29 and ties 113. It hits the known 10,480 ceiling
202 times versus the parent's 119, but each has three sub-9,000 outcomes.
All **512** games remain in stage one; no native mission appears. This is a
confirmed **score-consistency** training parent, not a barrier solution.
Both [first](fresh-selected-replay/replay.html) and
[second](fresh-confirm-selected-replay/replay.html) frozen best-effort
replays are native-verified and preserved locally. The global best remains
the older 10,480-point result because the new policy has not outranked it.

A [read-only loss-screen and sampled-choice check](../../diagnostics/continue-121-losses/README.md)
is decisive about this particular best replay: all four lives again gained
2,620 points near the recurring right-opening barrier, while the neural
policy chose `CONTINUE_PREVIOUS` only **once in 2,551 actions**, never in
the four pre-loss 64-action windows. The extra action was available but was
not meaningfully exercised there. The 256-game score gain cannot be called
a learned-hold solution. More identical training is not a sound test of
this mechanism without first establishing that the action is explored;
the verified new score parent and complete pilot state remain usable.

The full [run archive](run/) retains all eight optimizer/RNG checkpoints,
fixed validation records, training metrics and local artifacts; exact
trainer, loader, model, environment, curriculum, executor and vector sources
are copied beside this README. The production run finished normally after
**1,048,576** new actions, **112** complete boot training games and **4,002**
own-restored segments. No training, fixed-validation or fresh game entered
stage two. The full **468-test** repository suite passed before launch.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-continue-121 \
  --artifacts runs/defense-ppo-continue-121/artifacts \
  --initialize-repeat-policy results/defense/training/ppo-persistent-noise-long-119/milestone-000009718528 \
  --repeat-previous-action --steps 1048576 --envs 16 --rollout 256 \
  --batch-size 512 --epochs 4 --learning-rate .00025 --entropy .002 \
  --policy-bias-noise 1 --gamma .997 --gae-lambda .95 \
  --life-terminal --curriculum-probability 1 --curriculum-lookback 128 \
  --curriculum-trigger life-loss --curriculum-restored-life-only \
  --curriculum-share --curriculum-boot-envs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --eval-seed 10000 \
  --mlx-cache-mb 512
```
