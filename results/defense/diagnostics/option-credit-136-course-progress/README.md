# Option-aware credit still loses at the familiar early course segment

The [read-only report](report.json) re-executes the selected option-credit
model's [fresh native-verified replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html)
exactly on the original emulator. Its **2,537** neural decisions earn
10,480 displayed points, but the four visible life losses occur when the
immutable first-stage stream pointer is at decoded row **33, 34, 34,
33** of **126**. This matches earlier ordinary, learned-duration and
screen-diverse policies: changed credit assignment has not demonstrated
actual passage.

The pointer is hidden nonvideo RAM sampled only after replay for forensic
analysis. It was never a training observation, reward, reset target,
policy input, model-selection signal or replay-promotion criterion.
Stream position is not exact collision time or proof of safely traversed
obstacles. No model was updated from this report.
