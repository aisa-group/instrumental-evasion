# Base image for the four tau-bench customer-service tasks (tauc_tau_retail_*).
#
# Built into  tb-db-entry.sif  by scripts/build_images.sh.
#
# The tasks work a SQLite database, so the image adds the `sqlite3` command-line
# shell (the instruction assumes it exists) on top of python3, whose sqlite3
# module comes with the base. ca-certificates is included so TLS works for any
# in-container fetch. No database is baked in: the harness seeds /app/data at
# sample setup, so the starting state lives with the code and cannot drift from
# what the scorer expects. This matches the original build in effect.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends sqlite3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data
WORKDIR /app
CMD ["sleep", "infinity"]
