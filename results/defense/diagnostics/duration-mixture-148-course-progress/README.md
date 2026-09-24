# Long-duration mixture: no later course passage

The [forensic report](report.json) exactly reexecutes two independently
verified learned-policy fresh replays from original boot: the
[training-mixture treatment](../../training/ppo-duration-mixture-148/fresh-treatment-replay/replay.html)
and [no-mixture control](../../training/ppo-duration-mixture-148/fresh-control-replay/replay.html).
At each visible life-loss update, the original stage-one course-stream
pointer decodes to rows **34 / 34 / 32 / 34** for treatment and
**33 / 34 / 34 / 33** for control, out of 126 stream rows. Neither
enters stage two. This does not prove the exact physical collision
row: the hidden pointer is sampled when visible loss bookkeeping
appears, after possible animation delay.

The pointer and static row decoder are strictly post-hoc diagnostics.
They are not policy observations, rewards, training targets, archive
selection features, action overrides or replay-promotion criteria.
The selected single-game replays cannot replace the separate 64-game
complete-game estimates in the [training record](../../training/ppo-duration-mixture-148/README.md).
