# Level-5 experiment logs

These are complete JSONL copies of the six main continuation/comparison logs,
including weak checkpoints, the terminal-parser failure, and graceful pauses.
Each `start` record contains the run configuration and source checkpoint hash;
episode, progress, validation and stopped records preserve the actual outcomes.
Small smoke-test runs are excluded from this archive and the totals below.

| Log | Start PPO counter | Stop PPO counter | Additional actions |
|---|---:|---:|---:|
| `level5-baseline.jsonl` | 6,402,048 | 9,502,720 | 3,100,672 |
| `level5-terminal-fix.jsonl` | 9,502,720 | 14,733,312 | 5,230,592 |
| `level5-lr1e4.jsonl` | 12,500,992 | 14,503,936 | 2,002,944 |
| `level5-lr1e4-extended.jsonl` | 14,503,936 | 18,366,464 | 3,862,528 |
| `level5-entropy003.jsonl` | 15,503,360 | 17,506,304 | 2,002,944 |
| `level5-entropy003-extended.jsonl` | 17,002,496 | 17,502,208 | 499,712 |
| **Total** | | | **16,699,392** |

Counters follow checkpoint lineage; branches overlap and must not be added as
absolute counters. The final run alone stopped with `target_met: true`. Its
20-game validation had a complete level-5 game. The reserved 100-game test is
stored separately in [trained.json](../trained.json), not in these training logs.

The chart can be regenerated directly from these archived files using the
command in [LEVEL5.md](../../../LEVEL5.md), without ignored `runs/` directories.
