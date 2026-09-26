"""Offline analysis of runner logs.

* ``summarize`` -- the main outcome measures (task completion, confirmed
  prohibited effects, ESR, EAR, control solve rates, and runtime) per agent,
  per task family, or per task.
* ``command_patterns`` -- the seven overlapping command-pattern categories of
  the command-pattern appendix, with episode prevalence and call-level allow
  rates.

Both read the write-once attempt artifacts written by
``python -m instrumental_evasion.runner.run`` and make no model calls.
"""
