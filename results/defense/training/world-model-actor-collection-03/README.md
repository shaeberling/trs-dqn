# Fresh real experience from the learned actor

The learned imagined-return actor played 24 **new complete games**, seeds
67000–67023. The first 20 provide **30,716 training actions**, mean score 283;
the final four provide **6,092 held-out actions**, mean 280. Every game lost
in stage 1. This is data collection, not progress beyond the barrier.

Behavior is the [uniform-world actor at 500 updates](../imagination-boundary-comparison-01/uniform/update-000500/state.json).
It chooses every action from its learned categorical policy. There is no
external exploration override, so recurrent memory always records the action
actually executed. Memory resets at each boot game and survives life losses.
Inputs are visible screens and own neural memory/previous actions only.

The train/held-out seed assignment was fixed before collection. Original
screen/action/visible-score/continuation arrays, checkpoint provenance and
SHA-256 hashes are retained in the [manifest](manifest.json) and
[archive audit](archive-audit.json). No previous evaluation trace was imported,
no action-imitation loss is introduced, and the behavior weights stayed fixed.

A separate native test collected two complete games, then independently
reproduced both action/screen/reward streams. It checks correct memory reset,
dataset reader compatibility, split isolation, refusal to overwrite output
and checksum rejection. That additional test passed after the 393-test suite.

The union tool now requires explicit `--allow-policy-mixture` to combine
collections from different learned policy parents. Each source retains its
parent hashes and manifest identity. Game/timing checks, immutable episode
bytes, original held-out labels and seed-overlap rejection remain mandatory.
A separate new test covers that opt-in and provenance preservation.

The first feedback fit continues the uniform world model from 12,000 updates
using the new checked union: **60 training games / 116,495 actions**, with
**12 held-out games / 23,304 actions**. It targets 14,000 total dynamics
updates and has now finished; see the [full feedback world archive](../world-model-feedback-01/README.md).
This is the first world update incorporating actual new actor
experience; an explicit full-state actor continuation follows. It is not
yet a fully automated online Dreamer loop or evidence of successful play.

```bash
venv/bin/python -m rl.defense_world_actor_data \
  results/defense/training/imagination-boundary-comparison-01/uniform/update-000500/model.safetensors \
  --output runs/defense-actor-collection-reproduction --games 24 --heldout 4 --seed 67000
```
