# Coherent key exploration still converged to the same visible failure

The [read-only report](report.json) checks hashes and prior native replay
verification, then renders recorded screen bytes from the selected
[key-factor PPO policy](policy-1-losses.png) and the
[confirmed run-121 score parent](policy-2-losses.png). Both selected
best-effort games earned **2,620 displayed points on each of four lives**,
with the familiar broad right-opening barrier approaching a ship left of
the opening. The replay seeds differ and the white flash is not a precise
collision label; equal life scores do not prove identical physical failure.

The selected key-factor policy was re-executed at its verified replay seed.
All **2,528** recorded screens, physical actions, rewards and the final
result matched exactly. Instrumenting only its sampled 21-way choices found
**zero** `CONTINUE_PREVIOUS` selections in the full game, hence zero in all
four 64-action pre-loss windows. The physical pure-movement fractions in
those windows were **50.0%, 48.4%, 57.8%, 56.3%**. These counts do not
measure net displacement, and they do not prove that the factorized noise
was absent during training: frozen evaluation intentionally has no noise.
They do show that the selected learned policy did not turn the training
perturbations into persistent action use at this obstacle.

The analysis does not change model input, reward, actions, policy selection
or replay. No diagnostic example enters training. The exact source trace
and verification files remain in the archived evaluation bundle.
