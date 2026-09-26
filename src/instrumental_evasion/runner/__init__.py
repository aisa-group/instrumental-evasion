"""The runner: task, gate, and scaffold, composed per episode.

Three things compose here, and they are deliberately independent of each other:

* **the task** -- a `TBTaskSpec` (`instrumental_evasion.tasks.tb_spec`): image,
  prompt, policy id, scorer.
* **the gate** -- `runner.gate`, the single place a proposed tool call becomes
  an ALLOW or a BLOCK, and the single decision-log format.
* **the scaffold** -- `runner.scaffolds`, whatever drives the agent: our own
  ReAct loop over an API model, the Codex CLI, or the Claude Code CLI.

A scaffold is a plugin. It gets a prepared sandbox, a prompt and a gate, and it
is responsible only for making the agent act and for routing every proposed tool
call through the gate before it runs. It is not responsible for scoring, for
classifying attempt markers, or for the decision log -- those are the runner's,
which is what keeps two scaffolds' numbers comparable.
"""
