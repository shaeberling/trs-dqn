# Final 128-seed check of the modest firing-key bias

Two smaller diagnostic-only paired samples suggested that a -1 logit bias
on Space-containing commands might improve complete-game displayed score,
but neither reached stage 2. Before treating that bias as a possible
training-selected parent, this final check predeclared the same 150-point
mean-gain requirement and played 128 new complete boot games per variant
on shared training seeds 280000–280127.

The unchanged parent averaged **10,037.66** points; the -1 diagnostic
variant averaged **10,042.42**, a gain of only **4.77 points**. All 256
games remained in stage 1. The candidate fails the predeclared margin
and is rejected. No altered weights were saved, no training update was
made, and no replay was promoted. This rules out a simple global
Space-bias adjustment as a demonstrated escape from the shared failure;
it does not rule out context-sensitive learned control.

`report.json` preserves all paired complete-game results. `probe-source.py`
matches the recorded source hash, and the original model and verified
10,480-point global best replay remain unchanged.
