# Recurring failure: current training pair

The user's observation remains supported by the latest recorded screens:
the stronger new policy still fails in the familiar right-opening barrier
sequence, with the ship near the centre/left. This is a shared navigation
bottleneck, not proof of identical collisions or of a wall-versus-projectile
cause. The control fails earlier in the approach. All three screen sheets
were visually reviewed.

| Selected verified replay | Points earned per life | Original-screen panels |
| --- | --- | --- |
| Preserved global best | 2620 / 2620 / 2620 / 2620 | [Best](policy-1-losses.png) |
| DQN 57, long-discount control | 2530 / 2550 / 2450 / 2450 | [Control](policy-2-losses.png) |
| DQN 58, temporal consistency | 2570 / 2570 / 2570 / 2570 | [Consistency](policy-3-losses.png) |

At 7,293,216 actions, 200,000 after each calibration, the full ten-game
evaluations average 9,890 (control) and 10,262 (consistency), with medians
9,890 and 10,280 and best scores 9,980 and 10,280. All 20 games terminate
in stage 1; none reaches stage 2 or completes a mission. These reused seeds
are validation, not fresh success-rate estimates. Neither trial improves
the global best. Both complete checkpoint/optimizer states and compressed
log prefixes are archived; the selected replays originally verified all
2,399 and 2,541 base commands respectively.

Pure directional commands comprise 35.9–46.9% of the control's four
64-action pre-alignment windows and 37.5–45.3% of consistency's, versus
51.6–71.9% for the global best. These counts do not measure displacement:
commands can occur during animations, and stage-1 Space-plus-arrow fires
instead of moving. The agent is not simply choosing no movement at all.

The plausible explanation remains that learning reliably earns the earlier
score but has not discovered a sequence that survives the next obstacle.
This is a hypothesis, not a demonstrated cause. The present stabilizer has
not fixed it. Repeated near-ceiling scores or lower learning losses must not
be reported as navigation progress; a learned trajectory passing the
obstacle and then reaching a later stage is the useful test.

The current trainers already practice from their own opaque states saved
64 actions before visible life loss. The log audit confirms correct rewind
offsets, protected boot-only workers, and restored segments ending at their
first visible life loss. Thus "practice near the failure" is already being
tested, not an untried remedy. Longer rewind, action persistence, learned
hold durations and auxiliary representation tasks also have preserved
negative results; blindly repeating those is not a new diagnosis.

This review uses only hash-checked recorded screens/actions/rewards and
original native-verification records. The [report](report.json) records
provenance and action windows. White flashes and visible life decrements
are alignment markers, not precise collision timestamps. No hidden RAM,
hand-coded route, reward bonus, evaluation training examples, or policy
update was introduced by this diagnostic. The stronger global replay stays
unchanged while the controlled training pair continues.
