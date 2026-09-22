# Frozen world model: visible text reconstruction

This is a read-only diagnostic of the original-objective world at 18,000
updates, using the same 64 held-out windows as its uniform audit. It evaluates
1,536 **observed-posterior arrival frames**, not future forecasts. No policy,
reward, game state or training dataset is changed.

The current visible-text channel represents an ASCII cell as its byte value
divided by 127, repeated over its six rendered pixels. The diagnostic averages
those six decoded values and rounds to the nearest byte. It independently
checks that this inversion recovers the original nonblank ASCII exactly when
given the actual renderer output. These rounded cells are never policy inputs.

| Visible cells | Count | Exact byte | Within two code values |
| --- | ---: | ---: | ---: |
| Nonblank ASCII | 60,173 | 21.76% | 77.28% |
| Changed nonblank ASCII | 7,281 | 3.23% | 15.86% |
| Digits | 8,378 | 20.36% | 65.92% |
| Stars | 4,881 | 16.12% | 44.72% |
| Changed player-one HUD cells | 645 | 2.33% | 10.39% |

Because exact-byte accuracy can exaggerate small numerical errors, the
[expanded report](report-with-tolerance.json) also includes tolerances and a
simple star-versus-space midpoint check, restricted to actual stars/spaces
in the first sixteen HUD cells. It separates 1,844/2,421 stars (**76.17%**)
and 9,263/10,806 spaces (**85.72%**) correctly. That classifier is diagnostic
only, never a learned-policy input or training target.

This supplies evidence that visually meaningful changed text is not faithfully
reconstructed, even with the arrival screen observed. It does **not** prove
that the latent representation lacks all such information, that rounding is
the intended decoder, or that text reconstruction causes the navigation
failure. Repeated cells/windows are correlated; these are not independent
success probabilities. ASCII reconstruction is only one part of a scene that
also contains small moving graphics.

The result motivates checking the representation/reconstruction objective,
including a categorical visible-byte target, before assuming that more
imagined updates alone will help. Such a target would still describe only
visible screen content; no hidden RAM, collision oracle or extra reward has
been introduced by this diagnostic.
