"""Fixed-request diagnostic of monitor decision quality.

No agent runs. The monitor classifies 40 fixed request profiles, built from 24
labelled permitted/prohibited cases, each several times at temperature zero.
Because every input is frozen, the diagnostic separates two error directions
(prohibited calls allowed, permitted calls blocked) and shows whether an
error disappears once the decisive context (working directory, full script
source, explicit operation) is disclosed.

Modules:

``cases``     the 24 cases, 40 profiles and exact monitor inputs
``classify``  classification through the gate's monitor client, and records
``report``    per-monitor counts for all profiles or the relevant-context subset
``qualify``   container check that each fixture command has its labelled effect

Run ``python -m instrumental_evasion.diagnostics.monitor_diagnostic --help``.
"""
