# Spatial-residual and matched-control loss positions

The [read-only forensic report](report.json) reexecutes the spatial
[fresh native-verified replay](../../training/ppo-spatial-residual-143/fresh-spatial-replay/replay.html)
and matched [control replay](../../training/ppo-spatial-residual-143/fresh-control-replay/replay.html)
exactly on the original emulator. Both score 10,480 in their recorded
single games but lose all four lives before passing stage one. At the
visible losses, the immutable stage-one course stream pointer has decoded
rows **34/33/34/34** for spatial and **34/34/34/33** for control, out of
126. This matches the longstanding repeated failure point; it is not a
verified stage passage.

The pointer is hidden nonvideo RAM sampled **only after replay**. It is
not a training observation, reward, curriculum feature, action rule,
checkpoint selector or replay-promotion signal. Its row count marks
stream progress, not the exact collision instant or proof of a route.
No model was updated from this report.
