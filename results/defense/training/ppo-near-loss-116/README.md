# Nearer own-loss practice: 64-action lead-in

The prior 256-versus-128-action same-parent comparison did not clear the
recurring stage-one barrier. Its longer rewind began much earlier in a life,
and 88 of 128 fresh paired games scored below the confirmed parent. This
trial tests the opposite allocation: practice from the learner's own rendered
state **64 decisions before a visible ship loss**, when separate native
diagnostics found keyboard inputs still changed graphics. That does not prove
the state is recoverable or locate the collision.

Run 116 resumes the exact full model, optimizer and policy RNG from the
[confirmed ordinary-action parent](../ppo-canonical-control-113/milestone-000008538880/state.json).
It uses the same 16 workers, 4 boot-only workers, PPO settings, visible-score
reward, screen input, fixed evaluation games, and 524,288-action duration as
the completed [128-decision control](../ppo-early-loss-control-115/README.md).
The sole intended change is training-only own-loss lookback **128 → 64**.
Evaluation always starts from boot and never restores a saved state. No
scripted navigation, collision signal, reward shaping or demonstration is used.

Four ten-game fixed-seed checks at 131,072-action intervals select one
checkpoint before a fresh matched 128-game comparison on seeds 600800–600927
against the common parent, the previous 128-decision control, and the
[longer-return arm](../ppo-long-credit-117/README.md). Passage requires an
observed native stage transition; a score ceiling is insufficient. The sole
independent collector watches this run's isolated artifacts and preserves
the verified global best unless a higher-ranked result passes replay checks.

The first [full model/optimizer/RNG milestone](milestone-000008669952/state.json)
is preserved after 131,072 new actions. Its ten fixed complete games
averaged **9,949**, median **10,435**, best **10,480**; all stayed in stage
one. This small reused-seed check is not a reason to promote the checkpoint
or end the planned continuation.

The full 524,288-action run finished cleanly, with **63 boot games and
4,203 restored practice segments**. Its four fixed ten-game means were
**9,949 / 9,729 / 10,346 / 9,680**, with no stage-two game. The third
[full optimizer/RNG checkpoint](run/step-000008932096/state.json) was
selected by those fixed games before testing fresh seeds. The complete
checkpoint, log and verified local replay history is preserved in [run/](run/).
The substantially larger number of short restored segments confirms that
the closer reset changed practice allocation, not that it taught passage.

On [128 new matched complete games](fresh-selected-128.json), seeds
600800–600927, the frozen selected model averaged **9,764.30**, median
**10,360**, best **10,480**; all were stage-one losses. The
[common parent](fresh-parent-128.json) averaged **10,398.91**, and the
[ordinary 128-decision control](fresh-control-128.json) averaged
**10,377.27**. Against the parent, near-loss practice improved **24**
paired games, worsened **92**, and tied **12**. It produced **30** scores
below 9,000 versus **four** for the parent, a **-634.61** mean difference.
The [longer-credit arm](../ppo-long-credit-117/fresh-selected-128.json)
averaged **10,121.80** on the same seeds; all 512 combined fresh games
stayed in stage one. Near-loss practice is not a confirmed successor and
did not break the repeated barrier.

The selected policy's [fresh best replay](fresh-selected-replay/replay.html)
independently verifies **2,581 neural actions**. Separately verified
[parent](fresh-parent-replay/replay.html) and
[ordinary-control](fresh-control-replay/replay.html) fresh best efforts
are kept for comparison. These are not mission wins. The globally ranked
best replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-near-loss-116 \
  --artifacts runs/defense-ppo-near-loss-116/artifacts \
  --resume results/defense/training/ppo-canonical-control-113/milestone-000008538880 \
  --steps 9063168 --envs 16 --rollout 256 --batch-size 512 --epochs 4 \
  --eval-every 131072 --eval-games 10 --eval-envs 10 --mlx-cache-mb 512 \
  --curriculum-probability 1 --curriculum-share --curriculum-boot-envs 4 \
  --curriculum-trigger life-loss --curriculum-lookback 64 \
  --curriculum-restored-life-only --life-terminal
```
