# All eight original levels: verified

Completed September 20, 2026, 07:31 UTC. The learned screen-only policy has
beaten the original game without changing its levels, physics or rewards.

[Watch the complete 585-point winning replay](results/level10/best/replay.html).
It is standalone HTML, with all 49,159 neural actions independently checked
against the frozen model: zero mismatches. One reserve ball remains at the
terminal GAME OVER screen. The exact game's loss paths cannot produce this
screen with a reserve remaining; this proves a final-level victory, not
merely reaching Level 8. Last-ball final-level endings remain ambiguous.

## Results and limitations

| Evaluation | Complete games | Mean | Median | Best | Reached Level 8 | Verified wins | Unverified final-level endings |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reused validation, used for selection | 70 | 298.94 | 280 | 585 | 11 | 1 | 10 |
| Fresh frozen-model test | 100 | 278.75 | 277.5 | 583 | 3 | 0 | 3 |

This establishes **winning capability, not reliable winning**. The conservative
screen-only detector cannot distinguish a last-ball final-level victory from
a loss, so the verified win count is a lower bound. Unverified endings are
not asserted victories or losses.

The model was selected before fresh testing. Seeds40000–40099 were each used
once, with unchanged sampled policy, no action overrides and no step limit.
All 100 games ended naturally, including two unusually long games taking
392,602 and 447,196 actions. No seeds were replaced and no unfinished games
were excluded. These seeds are now used and must not be called fresh again.

## What changed

The original game ends after eight levels and does not display Level 9 on
victory. The former level-transition measurement therefore missed wins.
After the user confirmed the eight-level target, a screen-only outcome test
and win-first selection replaced the unreachable displayed-Level-10 gate.
Outcome reporting does not change policy inputs, actions, rewards or physics.

A predeclared reassessment compared three strong historical checkpoints on
the same reused 20+50 games. Checkpoints122.5M and226.5M each produced a
verified win;226.5M won the declared complete-suite ranking. Candidate222.5M
was disqualified because one secondary game hit its validation guard. No
replacement seed or successful retry was used to make that suite eligible.

The new progress4 training trial stopped cleanly after499,712 additional
actions when the historical win was discovered. Its initial validation was
cancelled and it had not yet produced practice archives. The victory came
from the earlier long training run, not from that short trial or fresh data.

## Preserved evidence and current state

- [Frozen model package](models/breakdown-all-eight/README.md), including the
  exact model, optimizer, checkpoint configuration and final selection record.
- [Final selection](results/level10/all-eight-final-selection.json) and
  [three-candidate comparison](results/level10/all-eight-prior-reassessment-completion.json).
- [All 100 fresh games](results/level10/all-eight-fresh-test.json),
  [raw evaluation log](results/level10/logs/all-eight-fresh-test.log), and
  [completion/hash audit](results/level10/all-eight-completion-audit.json).
- [Winning replay action inspection](results/level10/all-eight-winner-226m5-inspection.json).
- [Stable best-effort replay](results/level10/best-effort/replay.html), now the
  same verified winner. Prior598-point nonwinning and581/576-point bundles
  remain in the version archives; no earlier effort was deleted.

All142 unit tests pass. The game binary and emulator library hashes are
unchanged. The final supervisor records `target_verified_and_tested`; its
process, evaluator and worker family have exited. No learner remains live.
The goal was met and monitoring concluded after the complete fresh test.

The HTML replay is local and portable. At the user's subsequent request,
[the existing public replay](https://breakdown-learned-replay.saschah.chatgpt.site)
was updated to this winning recording as site version 2; its URL and public
audience are unchanged. See the [deployment record](results/level10/public-replay-deployment.json).
Only the separate site's source was pushed for deployment. The training
code, winning model and evidence are now preserved in this repository's
commit history; committing does not publish or push them. Original Level-5 artifacts and
the earlier saved site version remain unchanged.

For a future reliability goal, retain this winner and test improvements on
training/reused validation only, with a new untouched final seed set declared
in advance. The completed fresh test must not guide model selection.
