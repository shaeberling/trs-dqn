# Playing test of four-step latent regularization

Neither arm improves navigation. All sixty complete baseline and trained
validation games lose in stage 1. Both finish 1,000 additional imagined
updates from the same validation-selected full 1,500-update behavior parent,
using their corresponding frozen 18,000-update world.

| Total actor updates | Original objective: mean / median / best | Four-step: mean / median / best |
| ---: | ---: | ---: |
| 1,500, refreshed baseline | 274 / 280 / 280 | 280 / 280 / 280 |
| 2,000 | 300 / 300 / 340 | 202 / 200 / 220 |
| 2,500 | 120 / 120 / 120 | 112 / 120 / 120 |

Each row uses ten uncapped boot games on reused seeds 10000–10009, not fresh
success-rate estimates. The parent is the earlier checkpoint of the second
feedback cycle, explicitly selected before this comparison; its regressed
final state remains preserved separately. The two refreshes retain actor,
critic, target, optimizers and RNG; only the new world differs. Both final
world weight files remain byte-identical to their frozen parents.

The [340-point control replay](uniform/artifacts/best/replay.html) verifies
1,557 neural commands. The [280-point four-step replay](focused/artifacts/best/replay.html)
verifies 1,532. The directory label `focused` comes from a reused archive
helper: it means the overshooting arm here, **not loss-focused sampling**.
All full actor/critic/target/Adam/RNG states and logs are hash-checked in the
[archive comparison](comparison.json). The global 10,480-point replay is
unchanged; these local bests are weaker, not new game milestones.
