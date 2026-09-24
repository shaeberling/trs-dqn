# Stronger phase-changing exploration at the repeated obstacle

The [zero-gated spatial comparison](../ppo-spatial-residual-143/README.md)
showed no stage passage and worse fresh-game score than both its unchanged
control and frozen input. A representation-only change did not fix the
repeated course-row-33/34 loss. The earlier [phase-changing option learner](../ppo-duration-phase-137/README.md)
did make symmetric physical-key preferences renew within a life, but used
standard deviations 2 for keys and 4 for durations and likewise stayed
in stage one. This follow-up tests **exploration amplitude**, not a
preselected route: increase those training-only factor standard deviations
to **6 and 8**, retaining independent uniform 24–64 own-action key-factor
renewal and duration-factor redraw only at visible life/episode boundaries.

Resume the exact [option-credit source](../ppo-duration-credit-136/run/step-000001179648/)
model, Adam state and policy/noise RNG. The source has already been the
common input to the milder phase-changing trial. Apart from the two
noise magnitudes, preserve its 20 physical keys × learned holds 1/4/16/64,
semi-Markov option-start score credit, 16 workers (four boot-only),
128-cell own-screen life-loss reset archive, original 100,000-T-state
cadence, learning rate 2.5e-5, gamma .997 and lambda .95. The policy
continues to receive only four raw video-memory frames. The only RL
reward is the original displayed-score difference; the factors depend
on neither screen geometry nor the forensic course pointer. Fixed
evaluation removes all noise and starts at original boot. Searched or
intervention actions are never loaded as examples.

First run a separate **16,384-action plumbing smoke** from the same
source to verify finite updates, actual option/noise draws, full
checkpoint/resume state and a native-verified local replay. It does not
select the production model. Then a **524,288-action first gate** to
absolute counter **1,703,936**, with four complete ten-game fixed checks
every 131,072 actions on seeds 10000–10009. Archive all full
model/optimizer/RNG checkpoints, logs and local replays. Select the
earliest highest stage rank, then fixed-game mean. Stop early on two
consecutive fixed means below 5,000; do not claim that low score means
the barrier cannot be passed.

Any stage-two or mission game must be reexecuted independently from
boot with the frozen learned policy, then checked on fresh complete
games before global-best promotion. If all games remain stage one,
extend to the same full 1,048,576-action budget as the milder run 137
**only if** its first-gate selection has fixed mean at least **10,470**
and no fixed check below **9,000**. Otherwise archive it as a negative
exploration test rather than spending more on a collapsed policy. If the
full extension also remains stage one, compare only an eligible selected
checkpoint on 64 untouched matched complete games, seeds
608000–608063, against the frozen option-credit input and the milder
run-137 selected checkpoint. A stage-one score gain can at most inform a
future training parent; it cannot replace the protected global best or
count as stage passage. Report actual noise/hold counts and complete-game
outcomes, not only planned exploration.

This is a bounded test of whether stronger unbiased phase changes cover
the needed behavior. It is not a promise that extra noise will solve a
sparse-reward barrier, and the negative result from the same source at
lower amplitude remains the comparison baseline.

The [16,384-action smoke](smoke/) finished normally at absolute counter
1,196,032. Its four complete fixed games averaged **10,455**, all stage
one; a separate local 10,480-point replay reproduced all **2,546** neural
actions from boot. The last progress report counted **394** key-factor
redraws, **48** duration-factor redraws and **22** 64-action options started;
the stronger noise settings were actually exercised. This is an
implementation check only, not a stage pass or score-parent claim. The
full **501-test** suite had passed on this trainer before this parameter-
only trial; no trainer source changed for the smoke.
