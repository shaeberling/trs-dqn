# Real-game check of the two world-model sampling variants

Neither variant supports useful play yet. This is an offline, frozen-world
actor calibration, not a completed online Dreamer loop. All sixty complete
native games, including untrained baselines, lost in stage 1.

| Imagined updates | Uniform mean / median / best | Focused mean / median / best |
| ---: | ---: | ---: |
| 0, untrained baseline | 280 / 280 / 300 | 274 / 280 / 280 |
| 500 | 282 / 280 / 300 | 284 / 280 / 300 |
| 1,000 | 280 / 280 / 280 | 270 / 270 / 280 |

Each row uses ten uncapped boot games on reused seeds 10000–10009; these
are validation comparisons, not fresh success-rate estimates. Actor settings
and initial actor/critic seed match. Each freezes its own corresponding
[12,000-update world](../world-model-boundary-01/README.md) and starts imagined
rollouts only from the forty training games. Neither uses held-out games or
evaluation replays for gradients.

Both local-best 300-point replays were independently verified in the original
emulator: [uniform](uniform/artifacts/best/replay.html), 1,568 commands, and
[focused](focused/artifacts/best/replay.html), 1,536 commands. All complete
actor/critic/target/Adam/RNG states, evaluations and logs are preserved with
[hashes](comparison.json). Both final world-model files remain byte-identical
to their parents. These runs are not added to the shared collector and cannot
replace the stronger 10,480-point best.

Focused loss sampling's improvement on selected forecast cases does not
translate into playing progress here. Frozen off-policy model errors and
limited state/action coverage remain concerns; this comparison does not
isolate a unique cause. The next data path collects **new actual games from
the learned actor**, without demonstrations, action overrides or evaluation
data, to prepare a real experience/model/actor feedback cycle.
