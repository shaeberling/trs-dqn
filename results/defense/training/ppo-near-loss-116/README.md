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
