# Shared workspace context

This repository can be used independently or as part of a multi-repository workspace.
Before substantial work, locate the coordinator: use `ROBOT_PROJECT_ROOT` when set,
otherwise inspect parent directories for `repos.yaml` and `context/PROJECT.md`.
On the original Jetson the coordinator is `~/workspaces/robot_ws`. Confirm that its
manifest includes this checkout before using its context. If unavailable, use this
repository's README and report that shared context could not be loaded.

Read the coordinator's `context/WORKING_AGREEMENTS.md`, `PROJECT.md`, and `STATUS.md`;
read `HARDWARE.md` for physical work and `WORKFLOWS.md` for validation. These files
are shared by Claude and Codex. Inspect this repository's actual branch and changes.
Keep implementation-specific detail here and project handoffs in the coordinator.
Do not copy private shared context or raw transcripts into a public repository.
Preserve user changes and do not activate physical hardware during documentation work.
