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
breaking any exact tie in favor of the earliest checkpoint,
then independently compare it with the frozen parent on 128 new matched
complete games, seeds 601400–601527. Score gains alone will not count as
barrier passage. The full optimizer/RNG and local replays will be archived
regardless of outcome.

For clarity, the fresh comparison includes **both** the immediate noise-run
parent at 8,801,024 and the older independently confirmed ordinary-action
score parent at 8,538,880, all on those same seeds. The immediate parent
tests the effect of this longer continuation; the confirmed parent prevents
a regression from being disguised by a weak intermediate baseline.

The first [full optimizer/RNG milestone](milestone-000008932096/state.json)
is preserved after 131,072 new actions. Ten ordinary complete games
averaged **9,933**, median **10,330**, best **10,430**, all stage one.
This is an early regression from the frozen parent, not stage progress;
the longer predeclared continuation remains active.

The second [full optimizer/RNG milestone](milestone-000009063168/state.json)
is preserved after 262,144 new actions. Ten fixed complete games averaged
**10,468**, best **10,480**, all stage one. This is a reused-seed score
rebound, not an independently confirmed improvement or barrier clear.

The seventh [full optimizer/RNG milestone](milestone-000009718528/state.json)
is preserved after 917,504 new actions. Its ten fixed complete games
averaged **10,470**, median/best **10,480**, all stage one: a new
validation-score leader by two points, but no native passage or fresh
confirmation. The remaining nine checks continue as planned.

The eleventh check at counter 10,242,816 tied the seventh check's **10,470**
fixed-game mean; all ten games again ended in stage one. The earlier saved
checkpoint remains provisional under the tie rule above.

The run finished all sixteen checks without a stage-two training or
validation game. The earliest 10,470-point checkpoint above remains selected.
On the first 128 fresh complete games (seeds 601400–601527), it averaged
10,429.22 versus 10,401.72 for its immediate noise parent and 10,362.97
for the independently confirmed ordinary-action parent. All 384 games
stayed in stage one. The selected model improved only 31 paired games
versus its immediate parent, worsened 58 and tied 39; its mean edge rests
largely on fewer low-score failures. To avoid treating this ambiguous
first sample as confirmation, **before any further evaluation** a second
untouched 128-game matched set is specified: seeds **601600–601727** for
the same three frozen checkpoints. No checkpoint will be reselected or
updated from these fresh results. Mission success still requires native
screen-observed evidence, not a score difference.

The entire planned **2,097,152-action continuation** finished cleanly with
**220 boot games, 8,206 restored segments and sixteen fixed ten-game
evaluations**. No training or validation game reached stage two. Fixed-game
means were **9,933 / 10,468 / 10,267 / 10,457 / 10,200 / 10,208 /
10,470 / 9,142 / 10,375 / 10,227 / 10,470 / 9,540 / 10,234 /
10,229 / 10,408 / 10,465**. The earliest 10,470-point checkpoint at
**9,718,528** was selected under the committed tie rule, not the final
weights. The full model, optimizer, both RNGs, metrics and native-verified
local replay history are archived in [run/](run/); the selected full
[optimizer/RNG milestone](milestone-000009718528/state.json) is separate.

On [the first 128 new complete games](fresh-selected-128.json), seeds
601400–601527, selected / [immediate noise parent](fresh-noise-parent-128.json)
/ [confirmed ordinary parent](fresh-confirmed-parent-128.json) means were
**10,429.22 / 10,401.72 / 10,362.97**, all stage one. The selected model
had only one sub-9,000 score versus three and five, respectively. It lost
more paired games than it won against the immediate parent (31 improved,
58 worsened, 39 tied), so the mean advantage warranted confirmation.

The predeclared [second 128-game set](fresh-confirm-selected-128.json),
seeds 601600–601727, gave selected **10,444.77**,
[immediate parent](fresh-confirm-noise-parent-128.json) **10,304.22** and
[confirmed parent](fresh-confirm-confirmed-parent-128.json) **10,377.97**,
again all stage one. It had **zero** sub-9,000 games versus eight and five.
Across both untouched sets (**256 games per policy**), selected / immediate
parent / confirmed parent means were **10,436.99 / 10,352.97 / 10,370.47**.
The selected model is **+66.52 mean points** over the confirmed score parent
on these seeds and cut sub-9,000 games from ten to one. It improved 107
paired games against that parent, worsened 86 and tied 63. This supports a
new **score** training parent, not barrier passage: all **768** fresh games
ended in stage one. Its lower-tail gain also comes with fewer exact-10,480
games than the immediate noise parent (114 versus 176), so the change is
not a uniform improvement on every seed.

The selected policy's [first](fresh-selected-replay/replay.html) and
[second](fresh-confirm-selected-replay/replay.html) fresh best-effort
replays independently reproduce **2,544** and **2,543** neural actions.
The corresponding parent and noise-parent replay bundles are preserved.
No fresh replay showed a mission; the global verified best remains the
older 10,480-point game and is not overwritten by this score-parent update.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-persistent-noise-long-119 \
  --artifacts runs/defense-ppo-persistent-noise-long-119/artifacts \
  --resume results/defense/training/ppo-persistent-noise-118/milestone-000008801024 \
  --steps 10898176 --eval-every 131072
```
