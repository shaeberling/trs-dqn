# Second actor feedback cycle: regression

Continuing the actor after its world learns from another fresh collection
does **not** improve play. All thirty complete validation games lose in stage 1.

| Total actor updates | Mean | Median | Best |
| ---: | ---: | ---: | ---: |
| 1,500, refreshed-world baseline | 276 | 280 | 280 |
| 2,000 | 244 | 250 | 260 |
| 2,500 | 110 | 110 | 120 |

The refreshed world has 16,000 dynamics updates. Actor/critic/target tensors,
both Adam states and both RNG states exactly retain the previous actor's
1,500-update state before a further 1,000 imagined updates. The archived
audit independently checks the actual starting checkpoint, not only settings.
The world remains frozen and byte-identical throughout behavior fitting.

Each row is ten uncapped boot games, reused seeds 10000–10009, not a fresh
success-rate estimate. The [280-point local-best replay](artifacts/best/replay.html)
verifies all 1,504 actions. It is the refreshed baseline, not an improvement
from the additional behavior updates. Full learned state and logs are retained.
The shared 10,480-point best remains unchanged.

This earlier 1,500-update checkpoint, rather than the regressed final actor,
is the planned common behavior parent for the next matched dynamics-objective
comparison. That is a declared validation-selected initialization, not a
claim that the final model improved or an import of evaluation trajectories.
