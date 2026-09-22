# First actor continuation after real-experience feedback

The first complete experience → world update → actor update cycle is
preserved, but **does not improve play**. All thirty complete validation
games lost in stage 1. The 10,480-point shared best remains unchanged.

| Total actor updates | Mean | Median | Best |
| ---: | ---: | ---: | ---: |
| 500, refreshed-world baseline | 284 | 280 | 300 |
| 1,000 | 278 | 280 | 280 |
| 1,500 | 276 | 280 | 280 |

The actor retains its previous 500 updates, critic/target, both Adam states,
sampling RNG and model RNG. Only its world is refreshed to the
[14,000-update feedback model](../world-model-feedback-01/README.md), trained
on old data plus 24 new actor games with original train/held-out labels.
The archive audit independently compares actual initial actor/critic tensors,
optimizer arrays and RNG state against the preserved parent, and confirms
that world tensors changed. Another 1,000 imagined updates then ran with
that new world frozen. The final frozen-world bytes match the declared parent.

Each row uses ten uncapped original-emulator games on reused seeds
10000–10009. The [300-point baseline replay](artifacts/best/replay.html)
has all **1,542 actions** independently verified; it uses an already-trained
actor after world refresh, not random actor initialization. It is retained
locally only and does not replace the stronger global best. All three full
actor/critic/target/optimizer/RNG bundles and completed logs are archived.

This validates the first orchestrated feedback cycle and preservation of
learned state, not useful navigation or faithful Dreamer reproduction.
The world model still forecasts visible loss poorly. Further own-experience
feedback and architectural/training investigation remain necessary; simply
reporting more imagined updates as progress would be misleading.
