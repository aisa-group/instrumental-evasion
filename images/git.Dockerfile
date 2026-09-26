# Base image for authored10_git_restore, the one authored task whose honest
# route uses git and whose verifier runs pytest.
#
# Built into  gitpush.sif  by scripts/build_images.sh.
#
# The full (non-slim) python:3.12.13-bookworm base already ships git and make,
# so only pytest is added. This matches the original build in effect: git, make,
# python 3.12.13 and pytest 9.1.1 present, no task state baked in (the harness
# seeds the repository at sample setup). pytest is pinned so the verifier runs
# the same collector across rebuilds.
FROM python:3.12.13-bookworm

RUN python -m pip install --no-cache-dir pytest==9.1.1

WORKDIR /workspace
CMD ["sleep", "infinity"]
