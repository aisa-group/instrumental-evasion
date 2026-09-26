# Base image for the stateful terminal tasks: ToolSandbox, ClawBench, and nine
# of the ten authored tasks (every authored task except authored10_git_restore,
# which needs git/make/pytest and uses images/git.Dockerfile instead). It also
# backs the ToolSandbox and authored sidecar services.
#
# Built into  tb-agent-stateful.sif  by scripts/build_images.sh.
#
# Deliberately plain: only CPython. No task state is baked in -- the harness
# seeds every task's files into the workdir at sample setup, so the starting
# state is identical for every rollout and can change without a rebuild. This
# is the exact base the original build used (`docker://python:3.12-slim`).
FROM python:3.12-slim

WORKDIR /app
CMD ["sleep", "infinity"]
