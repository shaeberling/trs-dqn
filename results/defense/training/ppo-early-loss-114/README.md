# Earlier own-loss practice: 256-action lead-in

The repeated stage-one losses in native-verified replays occur near the same
visible barrier sequence, usually after roughly 2,550–2,620 points per life.
Score is not an exact collision locator, and equal scores need not mean equal
positions. This trial tests whether a learner needs more lead-in time to move
before the barrier arrives. It changes only the **training-only** rewind from
128 to 256 decisions before this policy's own *visible* life loss. The model
still receives four screen frames; the reward is only displayed score change.
No collision memory, route, extra reward, scripted steering or demonstration
enters training or evaluation.

Run 114 and [same-parent run 115](../ppo-early-loss-control-115/README.md)
resume the exact same full model, optimizer and policy RNG from
[control 113's confirmed score checkpoint](../ppo-canonical-control-113/milestone-000008538880/state.json).
Their only intended setting difference is the 256-versus-128-action own-loss
lookback. Both start new emulator episodes, retain four boot-only workers, and
continue for 524,288 actions to 9,063,168. Four ten-game fixed-seed checks
are scheduled every 131,072 actions. A checkpoint will be selected using only
those checks, then compared with its parent and control on new, matched,
complete-game seeds 600600–600727. A stage clear requires a screen-observed
transition; score alone is not enough.

The sole independent collector watches this run's isolated `artifacts/` and
preserves the globally verified best replay if no better candidate appears.

The first [full model/optimizer/RNG milestone](milestone-000008669952/state.json)
is preserved after 131,072 new actions. Its ten complete fixed-seed games
averaged **10,385**, median **10,460**, best **10,480**; all stayed in stage
one. The matched control averaged **10,430**. This early reused-seed check
does not select a winner or demonstrate barrier passage; the predeclared
continuations and fresh comparisons remain necessary.

Both arms finished their planned 524,288 new actions without a stage-two
training or evaluation game. This longer-lookback arm completed **64 boot
games and 1,224 restored segments**; its four fixed ten-game means were
**10,385 / 10,193 / 10,292 / 10,429**, all stage one. The final
[9,063,168-action checkpoint](run/step-000009063168/state.json) was selected
by these fixed games *before* the fresh test. The entire full model,
optimizer, RNG, log, and verified local replay history is archived in
[run/](run/). The full optimizer/RNG for the selected checkpoint allows
subsequent experiments; it is not a confirmed successor.

On [128 new complete games](fresh-selected-128.json), seeds 600600–600727,
the frozen selected model averaged **10,399.69**, median **10,440**, best
**10,480**, with **zero stage-two reaches**. The same seeds gave the
[confirmed parent](fresh-parent-128.json) **10,416.48** and the
[matched 128-action control](../ppo-early-loss-control-115/fresh-selected-128.json)
**10,291.80**, also all stage one. Against the parent, the longer rewind
improved **29** paired seeds, worsened **88**, and tied **11**; it had one
game below 9,000 versus the parent's two, but only three 10,480-point
games versus the parent's 55. A reduced catastrophic tail does not by
itself make the model stronger. The longer rewind is **-16.80 mean points**
versus the parent and did not solve the navigation bottleneck.

The selected model's [fresh best replay](fresh-selected-replay/replay.html)
re-executes **2,505 learned actions** with neural-action verification. The
[same-seed parent replay](fresh-parent-replay/replay.html) is separately
verified over **2,530 actions**. Neither is a mission completion. The
read-only [first-milestone loss panels](../../diagnostics/early-loss-114-115-first-milestone/README.md)
show all four lives in each selected replay earning 2,620 points, with the
familiar barrier still visible. The global verified best is preserved.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-early-loss-114 \
  --artifacts runs/defense-ppo-early-loss-114/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 256 \
  --curriculum-restored-life-only --life-terminal
```
