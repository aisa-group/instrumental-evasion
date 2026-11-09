# Experiment requirements

- Run all non-Claude model experiments through Codex: `--scaffold codex`.
- Run Claude models through Claude Code (`claude_code`).
- Never launch a live ReAct experiment or fall back to ReAct when a Codex route
  needs work. Historical ReAct code, results, and frozen campaign snapshots are
  retained for provenance and offline tests, not for new model runs.
- Retain full original Codex rollout JSONL, including encrypted reasoning when
  generated, and verify session identity, model identity, tool interception,
  and file checksums. Input-only subsets do not satisfy full trace collection.
- Preserve requested model IDs and record the exact scaffold, reasoning effort,
  continuation setting, fixture seed, and monitor settings in each run manifest.
- When the user requests OpenRouter, use the OpenRouter API, not subscription
  auth. Choose an available provider appropriate to the experiment, pin it, and
  disable provider fallbacks. Honor any provider explicitly requested by the
  user; Azure is not a general requirement. Record the serving provider without
  silently substituting it. Keep API keys outside retained logs and configuration
  files; manifests may record private credential file paths.
- Keep completed campaign snapshots and results unchanged. Create a fresh run
  directory for reruns.
