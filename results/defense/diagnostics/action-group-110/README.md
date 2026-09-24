# Probability-preserving fire-key canonicalization

This separate read-only diagnostic addresses the flaw in the fixed-ten
action mask: removing eight fire aliases also discarded their learned
probability. For the same frozen strong PPO checkpoint, commands 9–17
are now merged by **summing their softmax mass** and sending the draw to
canonical Space. Commands 0–8 and 18–19 retain their original mass and
keyboard action. The mapping is fixed on every screen and cannot use
game-state hints. It exactly preserves the 12 grouped categorical
probabilities by construction, though physical equivalence of the fire
key combinations is established only for stage-one movement, not later
stages. No weights, reward, policy input or replay best are changed.

Matched complete games on fresh training-only seeds 600300–600363 compare
ordinary frozen sampling with this canonicalized control. The result is
diagnostic only and cannot be promoted as a newly trained policy.

The [completed 64-seed report](report.json) found **10,251.09 mean** for
both the ordinary and canonicalized policy, with **zero differing paired
scores** and no stage-two reach. Grouping preserves the original frozen
fire-versus-movement probabilities by construction; this native result
shows canonical Space did not change the displayed complete-game scores
on these seeds. It does not establish byte-for-byte trajectory identity,
later-stage equivalence, or a new learned navigation capability. It does
justify testing a *trained* policy whose PPO likelihood and entropy use
these grouped choices instead of the redundant 20-command distribution.

```bash
venv/bin/python -u -m rl.defense_action_group_probe \
  results/defense/training/ppo-own-loss-107/checkpoint-000008407808/model.safetensors \
  --output results/defense/diagnostics/action-group-110/report.json \
  --first-training-seed 600300 --games 64 --envs 16
```
