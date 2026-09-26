# Configuration

Every experimental setting is an environment variable, and every episode log
records the resolved values. `python -m instrumental_evasion.run` loads `.env`
from the repository root and then applies the paper's main-evaluation
condition (`src/instrumental_evasion/config.py`) to every variable that is still
unset. Precedence is therefore: shell environment, then `.env`, then the paper
defaults below.

## Experimental condition

| Variable | Paper value | Other values | Meaning |
| --- | --- | --- | --- |
| `POLICY_VARIANT` | `trimmed` | `full` | policy wording shown to the monitor (also `--policy-variant`) |
| `MONITOR_HISTORY` | `1` | `0` | show the monitor the episode's earlier calls (`--[no-]monitor-history`) |
| `MONITOR_HISTORY_MAX_CALLS` | `40` | any int | history window |
| `MONITOR_HISTORY_MAX_ARG_CHARS` | `600` | any int | truncation of each historical call's arguments |
| `MONITOR_HISTORY_OUTCOMES` | `1` (unset) | `0` | include the gate's verdict on each historical call |
| `AGENT_GUARDRAIL_NOTICE` | `disclosed` | `none` | notice appended to the agent's system prompt |
| `AGENT_REFUSAL_CONTEXT` | `reminder` | `none`, `monitor_reason_v1` | text delivered with a denial: the fixed reminder, nothing, or the monitor's parsed reason (Codex only) |
| `AGENT_CONTINUATION` | `neutral` | `legacy_calm` | message sent after an unsolved turn |
| `AGENT_REFUSAL_STOP` | `seven_consecutive` | `off` | end the episode after seven consecutive refusal-only turns |
| `AGENT_LOOP_STOP` | `no_new_action_v1` (local Claude Code only) | `off`, `denied_retry_impasse_v2` | end a Claude Code episode that repeats identical fully observed rounds; not supported on Modal |
| `AGENT_MAX_TURNS` | `300` | any int | model generations (ReAct) or CLI invocations (Codex, Claude Code) |
| `AGENT_ATTEMPTS` | `300` | any int | unsuccessful submissions tolerated by the ReAct scaffold |
| `AGENT_TOKEN_LIMIT` | `20000000` | any int | agent tokens, including cache reads (ReAct; Codex with the watchdog below) |
| `AGENT_TIME_LIMIT` | `6000` | any int | wall clock per episode, in seconds |
| `MONITOR_VERBOSE_BLOCK` | `0` | `1` | include the monitor's reason in the denial text (not used in the paper) |
| `CODEX_TOOL_SCOPE` | `native` | `atomic-mcp-v1` | Codex tool set (the alternative requires `--runtime modal`) |
| `CLAUDE_TOOL_SCOPE` | `legacy` | `configured` | whether Claude Code's tool list is restricted with `--tools` |

Common ablations:

```bash
MONITOR_HISTORY=0 AGENT_REFUSAL_CONTEXT=none ...    # myopic monitor, no reminder
AGENT_REFUSAL_CONTEXT=monitor_reason_v1 ...         # disclose the monitor's reason
POLICY_VARIANT=full ...                             # reinforced policy wording
```

## Monitor serving

Applied only when the monitor is the default `openrouter/openai/gpt-oss-120b`,
because the provider pin is specific to that model. When you pass another
`--monitor-model`, these stay unset (OpenRouter's default routing) unless you
set them.

| Variable | Paper value | Meaning |
| --- | --- | --- |
| `MONITOR_PROVIDER` | `parasail` | OpenRouter provider to pin (`none` for default routing) |
| `MONITOR_PROVIDER_SORT` | `none` | or `price`, `throughput`, `latency`; exclusive with a pinned provider |
| `MONITOR_PROVIDER_QUANTIZATIONS` | unset | comma-separated quantizations the pinned provider must serve (e.g. `fp8`) |
| `MONITOR_PROVIDER_REQUIRE_PARAMETERS` | `1` | only route to providers that support every request parameter |
| `MONITOR_STRUCTURED_OUTPUT` | `1` | request the JSON-schema verdict format |
| `MONITOR_REASONING_EFFORT` | `low` | reasoning effort requested from the monitor |
| `MONITOR_REASONING_EXCLUDE` | `1` | do not return reasoning tokens |
| `MONITOR_HTTP_TIMEOUT_SECONDS` | `20` | per-request timeout (at most 90) |
| `MONITOR_HTTP_RETRIES` | `4` | retries per classification |
| `MONITOR_HTTP_BACKOFF_PROFILE` | `rate_limit_v1` | retry backoff schedule |

Other monitor options, off by default:

| Variable | Values | Meaning |
| --- | --- | --- |
| `MONITOR_CONTEXT_PROFILE` | `command_only` (default), `command_cwd_v1` | also disclose each shell call's working directory (Codex) |
| `MONITOR_INSTRUCTION_PROFILE` | `original` (default), `effect_scope_v1` | optional extra classifier instructions; changes the prompt digest |
| `REGEX_RULESET_VERSION` | `selected-57-regex-v2` (default) and older versions in `regex_monitor.py` | rule set for `--monitor-classifier regex` (myopic, deterministic); it covers the 40 benchmark-derived tasks, not the authored tasks |

## Scaffolds

| Variable | Meaning |
| --- | --- |
| `CODEX_BIN` | Codex CLI; must be a package install (`scripts/install_codex.sh`) |
| `CODEX_TOKEN_BUDGET` | `session_usage_watchdog_v1` stops a Codex episode at the token limit using API usage telemetry. Applied by default for Codex with an `openrouter/` model outside Modal; requires Codex 0.153.3 |
| `CODEX_TOOL_INTERFACE` | `default` or `direct_v1` (code mode off; needs `CODEX_DIRECT_MODEL_CATALOG` and its `_SHA256`) |
| `CODEX_REASONING_SUMMARY` | `auto`, `concise`, `detailed`, `none` |
| `HOOK_CLAUDE_BIN` | Claude Code CLI |
| `HOOK_STANDALONE_PYTHON` | relocatable CPython bound into containers to run the hook (default: uv's `cpython-3.12.13-linux-x86_64-gnu`; install with `uv python install 3.12.13`) |
| `CLAUDE_CODE_OAUTH_TOKEN`, `CODEX_ACCESS_TOKEN` | override the subscription credentials the CLIs stored |

`--reasoning-effort` sets the agent's reasoning effort for Codex and Claude
Code.

## Credentials, images, and runtimes

| Variable | Meaning |
| --- | --- |
| `OPENROUTER_API_KEY` | monitor, ReAct agents, and Codex/Claude Code with `openrouter/` models |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | ReAct agents with `openai/` or `anthropic/` models |
| `IE_IMAGE_DIR` | directory of task SIF images (default `~/.cache/instrumental-evasion/images`) |
| `IE_STATE_DIR` | local-disk directory for per-episode sandbox state (default `/tmp`) |
| `<TASK>_IMAGE` | override one image, e.g. `TB2_BUILD_PMARS_IMAGE`; see `python -m instrumental_evasion.tasks.registry` |
| `RUNTIME` | default for `--runtime` (`apptainer`, `docker`, `modal`) |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal runtime |
| `MODAL_IMAGE_MANIFEST` | prepared Modal images (`python -m instrumental_evasion.runner.prepare_modal_images`) |
| `CLAW_FIXTURE_BANK_PATH` | optional fixed fixture assignments for the ClawBench tasks (default: fresh random fixtures per episode) |
