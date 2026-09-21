# Level-10 experiment logs

Complete JSONL logs copied after the corresponding learner stopped. Each
`start` record includes the configuration and source checkpoint SHA-256;
`stopped` records distinguish the target from a budget limit or manual pause.
These are historical experiment records, not evidence that level 10 was reached.
September 19 correction: the original game ends after eight levels; that run
was paused. The user subsequently authorized beating all eight; that goal is
now verified, and all training/evaluation processes have exited. See
[the completed report](../../../ALL_EIGHT.md).
Earlier zero final-level-clear counts only
measured transitions to displayed9, which the game never shows, and therefore
cannot count wins. See the [exact-binary audit](../game-level-cap-audit.json).
Exception: the first long continuation was externally stopped in a stalled
validation, so its raw log ends at `validation_start`. Its verified exit,
checkpoint hashes, and recovery are in [the recovery record](../long-run-recovery.json).

| Log | Starting PPO actions | Stopping PPO actions | Additional actions | Reason |
|---|---:|---:|---:|---|
| `level10-continuation.jsonl` | 17,502,208 | 19,582,976 | 2,080,768 | Paused after weaker validation results; retain checkpoints and prioritize smaller learning rate |
| `level10-lr5e5.jsonl` | 17,502,208 | 19,505,152 | 2,002,944 | Initial trial budget completed; extend its best-by-level checkpoint |
| `level10-lr5e5-extended.jsonl` | 18,501,632 | 20,680,704 | 2,179,072 | Paused after scheduled validation lagged source; retain final checkpoint for separate review and compare entropy/reward horizon |
| `level10-entropy001.jsonl` | 18,501,632 | 20,504,576 | 2,002,944 | Completed trial budget; final scheduled validation improved mean/later-level counts and reproduced level 5; continue from that validated checkpoint |
| `level10-gamma999.jsonl` | 18,501,632 | 20,504,576 | 2,002,944 | Completed trial budget without improving on the source; retain checkpoint and prioritize the lower-entropy continuation |
| `level10-entropy001-long.jsonl` | 20,500,480 | 22,253,568 | 1,753,088 | Externally stopped in an unfinished serve-stalled validation; recovered from the verified pre-validation checkpoint with a validation guard |
| `level10-entropy001-long-recovered.jsonl` | 22,253,568 | 27,598,848 | 5,345,280 | Gracefully paused after seven consecutive complete validations missed level 5; retain checkpoints and use freed capacity for the verified self-generated curriculum comparison |
| `level10-timing50k.jsonl` | 20,500,480 | 24,506,368 | 4,005,888 | Finer-cadence budget completed; seven complete validations and one disqualified serve-stalled suite, no complete validation level 5; retain original timing |
| `level10-curriculum.jsonl` | 20,500,480 | 22,503,424 | 2,002,944 | Local-archive budget completed; seven complete validations, none reaching level 5, and one disqualified suite; compare peer sharing to increase rare-state practice |
| `level10-curriculum-shared.jsonl` | 20,500,480 | 22,503,424 | 2,002,944 | Shared-archive budget completed; all eight validations complete, best depth checkpoint has two level-5 reaches and improved secondary mean; retain it and compare focused level-5-plus practice |
| `level10-curriculum-focused.jsonl` | 21,000,192 | 23,003,136 | 2,002,944 | Focused budget completed; seven complete validations and one disqualified suite, no new depth; same-start level-5 practice improved, so continue final weights for a longer tranche while retaining the audited reference |
| `level10-curriculum-focused-long.jsonl` | 23,003,136 | 28,045,312 | 5,042,176 | Paused cleanly after restored practice reached level 8 but plateaued and full-game validation regressed; 19 complete suites and one disqualified suite, no new full-game depth; give compute to the protected-boot comparison |
| `level10-curriculum-balanced.jsonl` | 23,003,136 | 28,815,360 | 5,812,224 | Gracefully paused after 19 consecutive complete validations missed level 5; retained logs and learner state, prioritize smaller learning rate |
| `level10-curriculum-balanced-lr1e5.jsonl` | 23,003,136 | 25,006,080 | 2,002,944 | Initial tranche completed; from-boot changing-policy training reaches 7, frozen validation only 5; extend final resumable learner, not the unsuccessful averaged diagnostic |
| `level10-curriculum-balanced-gae099.jsonl` | 25,006,080 | 30,007,296 | 5,001,216 | Longer-trace tranche completed cleanly; all 20 primary suites complete, four reach level 5 versus eight for matched control; no new frozen depth, preserve learner and use freed slot for predeclared SIL comparison |
| `level10-curriculum-balanced-lr1e5-long.jsonl` | 25,006,080 | 35,008,512 | 10,002,432 | Unchanged continuation completed cleanly; all 40 primary suites complete, highest frozen level 5; retain its 28.5M higher-mean reference and final learner, use freed slot for predeclared SIL20 dose comparison |
| `level10-curriculum-balanced-sil.jsonl` | 25,006,080 | 30,007,296 | 5,001,216 | SIL4 tranche completed cleanly; all 20 primary suites complete, frozen level 6 verified in both reused sets and full replay; continue the validated 28.25M depth checkpoint with unchanged settings after later primaries fail to surpass it |
| `level10-sil-level6-continuation.jsonl` | 28,250,112 | 33,251,328 | 5,001,216 | Unchanged tranche completed cleanly; all 20 primaries complete, selected 31.25M checkpoint establishes verified frozen Level 8 in the secondary audit and full replay; continue that source unchanged |
| `level10-curriculum-balanced-sil20.jsonl` | 25,006,080 | 30,007,296 | 5,001,216 | Higher-dose tranche completed cleanly; 19 eligible primaries, only two reaching 5, final suite disqualified by one incomplete game; do not adopt higher dose, retain final learner and all guard accounting |
| `level10-sil-level8-continuation.jsonl` | 31,252,480 | 36,253,696 | 5,001,216 | Unchanged Level-8-source control completed cleanly; 19 eligible primaries and one disqualified suite, highest primary 7, six secondary audits do not replace the source; preserve learner/log and launch declared same-source larger-buffer comparison |
| `level10-sil-level8-cap128k.jsonl` | 31,252,480 | 36,253,696 | 5,001,216 | Same-source 128k-buffer trial completed cleanly; 18 eligible primaries and two disqualified suites, highest primary 6, no secondary replaces the Level-8 source; preserve final learner/log and verify the promising fixed timing/history diagnostic end to end |
| `level10-timing50k-stride2-sil4.jsonl` | 31,252,480 | 35,258,368 | 4,005,888 | Initial timing/history fine-tuning completed cleanly; all eight primaries complete, selected 32M checkpoint improves reused70-game later-level counts and has a verified Level-8 replay; resume it at a smaller learning rate, not weaker terminal weights |
| `level10-timing50k-stride2-lr5e6-long.jsonl` | 32,006,144 | 40,009,728 | 8,003,584 | Lower-rate stage completed cleanly; 14 eligible primaries and two disqualified suites, no reference improvement; zero clears in 98 Level-8-start practice segments motivates matched Level-8-focused allocation |
| `level10-timing50k-stride2-min8.jsonl` | 32,006,144 | 40,009,728 | 8,003,584 | Focused allocation completed cleanly; all 16 primaries complete, no reference improvement; zero clears in 548 Level-8-start practice segments; preserve checkpoints and test a training-only self-reference retention penalty |
| `level10-timing50k-stride2-min8-refkl01.jsonl` | 32,006,144 | 40,009,728 | 8,003,584 | Reference-penalty trial completed cleanly; all16 primaries complete, selected35.5M checkpoint has fourLevel8 reaches across70 reused games and a verified576-point replay, with lower mean than prior reference; continue under local supervision |
| `level10-supervised-refkl01-v2.jsonl` | 35,504,128 | 236,003,328 | 200,499,200 | Intentional graceful stop after depth plateau;400 finished primaries363complete, selected116M improves to tenLevel8 reaches in70 games;11,644complete restored segments, final-win count unknown; resumed selected weights with within-level progress archives |
| `level10-supervised-progress16.jsonl` | 116,006,912 | 120,102,912 | 4,096,000 | Intentional graceful stop after confirming the original game ends after eight levels; all8 primaries complete, no selected-model improvement; checkpoints/log preserved pending user's revised goal |
| `all-eight-supervised-progress4.jsonl` | 116,006,912 | 116,506,624 | 499,712 | Stopped after a historical checkpoint proved a win; first validation cancelled, no practice archive, no performance improvement attributed to this trial |

`all-eight-fresh-test.log` is the byte-identical raw evaluation output from the
frozen winner's single 100-game fresh test, not a training log. All games
completed naturally. Summary and per-game records: [fresh test](../all-eight-fresh-test.json).

The frozen starting checkpoint also contains 500,000 earlier DQN actions.
Action counters follow a checkpoint lineage; adding counters from forks would
double-count their shared history. See [LEVEL10.md](../../../LEVEL10.md).

Separate implementation diagnostics (not performance candidates and not charted):

- `curriculum-smoke.jsonl`: 65,536 additional actions, verifies self-generated
  archive entries and restored-segment reward accounting; deliberately incomplete
  12-action validation cannot select a model. Clean exit, target false.
- `curriculum-cache-smoke.jsonl`: 4,096 additional actions, verifies the optional
  MLX cache target and allocation telemetry. No validation, clean exit, target false.
- `worker-import-smoke.jsonl`: matched 4,096 actions after deferred backend loading;
  all emulator workers omit MLX, with byte-identical model/optimizer output and
  identical RNG/counters versus the cache smoke. No validation, clean exit.
- `curriculum-local-refactor-smoke.jsonl`: 65,536 actions with sharing disabled;
  byte-identical model/optimizer and identical RNG/counters versus the original
  local curriculum smoke. Deliberately incomplete validation, clean exit.
- `curriculum-shared-smoke.jsonl`: 65,536 actions, ten same-run publications and
  three completed peer-sourced segments with verified origins and reward accounting.
  Deliberately incomplete validation, clean exit; no model selected.
- `curriculum-origin-logging-smoke.jsonl`: 65,536 matched actions with reservation
  disabled; model/optimizer byte-identical and learner RNG/counters identical to
  the shared smoke. Exact action-origin counters; incomplete validation, clean exit.
- `curriculum-reserved-boot-smoke.jsonl`: 65,536 actions with two of four workers
  reserved for from-boot play. Protected workers never restore populated archives;
  other workers exercise 2,798 restored actions. Incomplete validation, clean exit.
- `curriculum-reserved-boot-prob1-smoke.jsonl`: 65,536 actions with two protected
  workers and probability-one resets for the others. Seven complete practice
  segments, three collector-sourced; verified origins/new reward, and at least
  half of each logged rollout from boot. Incomplete validation, clean exit.

All start from the verified 20,500,480 checkpoint; no diagnostic checkpoint
will be used as the source of the curriculum comparison.

Additional optional self-imitation diagnostics, also from that checkpoint:

- `sil-disabled-compat-smoke.jsonl`: 65,536 actions, model/optimizer byte-identical
  to the reserved-boot probability-one smoke; identical learner RNG/counters.
- `sil-enabled-smoke.jsonl`: 65,536 actions, 240 auxiliary updates, 39 completed
  learning suffixes, exact newly earned score and bounded/action accounting.
- `sil-truncation-smoke.jsonl`: 4,096 actions, all 256 artificial episode
  cutoffs discarded; zero replay data and zero auxiliary updates.

All exit 0, deliberately incomplete validation selects no model, and none is
a performance candidate or future training source. Exact hashes/counts and
checks are in [the SIL smoke audit](../sil-smoke-verification.json).

Training-only self-reference diagnostics from the selected 32M timing/stride
checkpoint (each adds 65,536 actions, four workers/two protected):

- `reference-kl-disabled-before-smoke.jsonl` and
  `reference-kl-disabled-after-smoke.jsonl`: optional penalty disabled before
  and after implementation; model/optimizer bytes and nonconfig saved state
  match exactly, as do all learning/episode/validation records.
- `reference-kl-enabled-smoke.jsonl`: weight 0.1, nine reconciled learning
  suffixes, finite losses/KL, unchanged reference weights, clean exit.

All three exit 0. Their intentionally 12-action validation guards disqualify
performance claims; no diagnostic checkpoint is selected or resumed. See the
[implementation audit](../reference-kl-implementation-verification.json) and
[byte-identical log archive checks](../timing50k-stride2-min8-refkl01-launch-verification.json).

Progress-archive diagnostics from selected116M (32workers/16protected):
`progress16-disabled-before-smoke.jsonl` and
`progress16-disabled-after-smoke.jsonl`,65,536 actions each, both exit0.
Models/optimizers and nonconfig learner state/learning events match exactly.
No diagnostic weights or states enter the performance experiment; see
[verification](../progress16-implementation-verification.json).

Original-game outcome reporting compatibility: `all-eight-outcome-compat-smoke.jsonl`,
65,536 actions from selected116M, exit0. Model/optimizer bytes and saved RNG
match `progress16-disabled-after-smoke` exactly; only declared config and
selection-rank bookkeeping differ. No diagnostic weights enter training.
See [the all-eight implementation audit](../all-eight-implementation-verification.json).
