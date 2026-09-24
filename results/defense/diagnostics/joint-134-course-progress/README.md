# Joint key-duration model: the shared early failure remains

The [read-only report](report.json) exactly replays the selected joint
policy's [fresh verified game](../../training/ppo-duration-joint-134/run/fresh-selected-replay/replay.html)
on the original emulator. Its **2,546** neural decisions earn 10,480 points,
but all four visible life losses occur when the immutable first-stage course
stream pointer has reached decoded row **34, 33, 33, 33**, respectively,
of **126**. Earlier ordinary and duration-only policies lost at rows
33–34 too. High four-life score therefore remains repeated early progress,
not near-completion of stage one.

The pointer is nonvideo RAM used only *after* replay for forensic analysis;
it is not a policy input, reward, curriculum target, action choice, training
selection or replay-promotion signal. Pointer progress is not exact collision
time or proof that the ship safely navigated those rows. No model was
updated from this report. The source file and exact source/replay hashes are
preserved beside the report.
