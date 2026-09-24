# Screen-diverse duration learner: same early loss

The [read-only report](report.json) re-executes the selected checkpoint's
[native-verified fresh replay](../../training/ppo-duration-screen-135/run/fresh-selected-replay/replay.html)
exactly on the original emulator. Its **2,508** learned actions earn
10,480 points, but four visible life losses occur at decoded first-stage
course stream row **34, 34, 34, 33** of **126**. A 128-cell screen-diverse
training reset archive therefore did not visibly move the repeated loss
past the obstacle seen in earlier policies.

The stream pointer is nonvideo RAM sampled only *after* replay for
forensic analysis. It was not a policy input, training reward, reset
selection, action override, model-selection signal or replay-promotion
criterion. Pointer progress is not collision time or proof that the
ship safely traversed those rows. No model was updated from this audit.
