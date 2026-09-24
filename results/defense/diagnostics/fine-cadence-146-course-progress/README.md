# Fine-cadence selected replay: same stage-one bottleneck

The [forensic report](report.json) reexecutes the selected fine-cadence
[learned replay](../../training/ppo-duration-fine-cadence-146/fresh-selected-replay/replay.html)
from the original boot, matching every one of its **5,014** learned
actions, score changes and screen frames. On the four visible life-loss
events, the original stage-one stream pointer decodes to rows
**33 / 33 / 33 / 34 of 126**. The game remains in stage one and scores
10,480 in this one selected replay. This matches the longstanding
early-course bottleneck despite finer 50,000-T-state decisions and
history-matched retraining.

The pointer is hidden RAM used strictly after the fact for diagnosis;
it is neither the collision coordinate nor a measure of successfully
cleared rows. Neither the trainer, model, score reward, checkpoint
selector nor live replay uses it. This one replay is not a fresh-sample
performance estimate; see the separate complete-game evaluations in the
[training record](../../training/ppo-duration-fine-cadence-146/README.md).
