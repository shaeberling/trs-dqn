# Artifact retention

This repository is for emulator and learning code, concise experiment records,
and a few reproducible public results. It is not the archive for every training
step. Generated `runs/`, Defense training/diagnostic outputs, and Cosmic results
are ignored by Git. They may remain on a research machine, but important
checkpoints need a separate backup before that machine's copy is pruned.

Keep in Git only a deliberately selected frozen model, its configuration and
evaluation, and a portable verified replay for each game milestone. The stable
Breakdown and Defense best-replay links in the README must continue to work.
An optimizer state is useful for resuming training, but is not needed to view a
replay or evaluate a frozen model; resume checkpoints belong in local or
external storage. Raw JSONL logs, old model versions, exploratory diagnostics,
and every scheduled checkpoint do not belong in Git.

Historical Defense experiment README files remain as compact research notes.
Some of their links point to local-only archival inputs or outputs and will
not resolve in a fresh source clone. Tests that require these archives are
archival integration tests; ordinary environment and training-code tests do
not require the old runs. Do not use archival traces as demonstrations or
reward labels for new games.

The current Breakdown best and best-effort archive versions remain for their
existing stable links and checksum manifests. Older versions and full logs
are local-only. Future promotions should not automatically commit another
version tree.

Before publishing a new best, verify its frozen replay from original boot,
copy the selected weights, configuration, evaluation and replay to a small
curated release path, and deliberately add only those files. Never run
`git add -f` on an entire ignored experiment directory.
