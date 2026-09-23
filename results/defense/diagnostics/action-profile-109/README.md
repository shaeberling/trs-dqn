# Frozen-model test of distinct stage-one commands

The strongly checked PPO model is played from boot on matched fresh
training-only seeds with its ordinary 20 commands and with a fixed subset
of commands 0–9 (NOOP, eight directions, Space). In stage one, Space plus
each direction does not translate the ship; those combinations are
alternate fire commands, not additional movement choices. The subset is
fixed even if a game reaches a later stage; no screen parsing, direction
choice, route, steering override or extra reward is introduced.

This is a **read-only diagnostic**. The frozen model was not trained with
the smaller action set, so its result is not a new learned-policy result
or promotion candidate. The paired full-game report and exact source hash
will show whether removing stage-one aliases merits a separate learned
policy. The independently verified global best remains unchanged.

The [completed 64-seed report](report.json) is sharply negative for this
**frozen, unretuned model**: ordinary policy mean **10,157.97** versus
fixed-ten-command mean **996.88**. Every masked game scored below 3,000;
both policies remained in stage one. Removing eight fire-alias logits also
removes their *learned probability mass*, so the mask does not preserve
the policy's fire-versus-move choice. This is not evidence that a policy
trained from scratch with a compact action profile would fail. A separate
probability-preserving alias collapse is needed before attributing the
regression to the command profile itself. No weight or global replay was
changed.

```bash
venv/bin/python -u -m rl.defense_action_profile_probe \
  results/defense/training/ppo-own-loss-107/checkpoint-000008407808/model.safetensors \
  --output results/defense/diagnostics/action-profile-109/report.json \
  --first-training-seed 600200 --games 64 --envs 16
```
