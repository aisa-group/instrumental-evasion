# Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure

Code for the paper *Instrumental Monitor Evasion Emerges Under Ordinary Task
Pressure* (anonymous submission).

An agent receives an ordinary task whose completion requires an operation that
an operator policy prohibits. A synchronous LLM monitor reviews every tool call
against that policy and blocks violations. The agent is told that tool calls
may be reviewed, receives explicit denials, and is asked to continue when it
stops, but it is never told to evade anything. A deterministic scorer then
checks whether the prohibited outcome was reached anyway.

This repository contains:

* the benchmark: **50 task–policy pairs** from ToolSandbox, ClawBench,
  τ-bench, Terminal-Bench 2.1, OpenThoughts-TBLite, and ten authored terminal
  tasks ([docs/tasks.md](docs/tasks.md));
* the harness: a runner with three agent scaffolds (our own ReAct loop, the
  Codex CLI, and Claude Code), a synchronous tool-call monitor with episode
  history, and three container runtimes (Apptainer, Docker, Modal)
  ([docs/design.md](docs/design.md));
* the ablations and diagnostics of the paper: reasoning effort, monitor
  comparison, myopic monitor, denial reasons, forged denials, and the
  fixed-request monitor diagnostic;
* analysis code for the reported outcome measures.

Agent traces are released separately; see [Data](#data).

## Installation

Requirements: Linux, Python 3.12 with [uv](https://docs.astral.sh/uv/), and a
container runtime. The paper's runs used Apptainer.

```bash
git clone <repository-url> instrumental-evasion && cd instrumental-evasion
uv sync                       # add --extra modal for the Modal runtime
cp .env.example .env          # then set OPENROUTER_API_KEY
```

**Task images.** The 50 tasks use 17 Apptainer images: three base images
([images/](images/)), nine Terminal-Bench 2.1 images built from upstream's
prebuilt task images, and five TBLite images replayed from upstream
Dockerfiles. Build them into `$IE_IMAGE_DIR` (default
`~/.cache/instrumental-evasion/images`):

```bash
git clone https://github.com/harbor-framework/terminal-bench-2 ../terminal-bench-2
git -C ../terminal-bench-2 checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6
git clone https://github.com/open-thoughts/OpenThoughts-TBLite ../OpenThoughts-TBLite
git -C ../OpenThoughts-TBLite checkout 5c37b41f00ce04719a4453061076ae9f46b74b7d

scripts/build_images.sh --tb2-repo ../terminal-bench-2 --tblite-repo ../OpenThoughts-TBLite
```

`scripts/build_images.sh --help` lists the prerequisites; `--family base|tb2|tblite`
builds one family. Keep `IE_BUILD_SCRATCH` (default `/tmp`) on local disk.

**Agent CLIs** (only for the `codex` and `claude_code` scaffolds). The paper
used Codex CLI 0.153.3 and Claude Code 2.1.263:

```bash
scripts/install_codex.sh                              # Codex as a package install
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.263   # native Claude Code binary
uv python install 3.12.13                             # runs the hook inside containers
```

The Claude Code binary is bind-mounted into each task container, so it must be
the native single-file build, e.g. `~/.local/share/claude/versions/2.1.263`
(set `HOOK_CLAUDE_BIN` if it is not the `claude` on your `PATH`); the npm
package does not run there. Codex must be a package install (a bare `codex`
binary is refused); set `CODEX_BIN` if it is not on your `PATH`.

Codex and Claude Code run either with an `openrouter/...` model (API key) or
on a ChatGPT or Claude subscription (the credentials the CLIs store after
`codex login` / `claude login`).

## Quickstart

```bash
# The roster, with each task's policy and image path
uv run python -m instrumental_evasion.tasks.registry

# Show the resolved configuration without running anything
uv run python -m instrumental_evasion.run --dry-run --arm monitored

# One unmonitored control episode: is the task solvable?
uv run python -m instrumental_evasion.run tb2_build_pmars --arm control \
    --scaffold react --model openrouter/deepseek/deepseek-v4-flash

# One monitored episode under the paper's condition
uv run python -m instrumental_evasion.run tb2_build_pmars --arm monitored \
    --scaffold codex --model openrouter/deepseek/deepseek-v4-flash

# Outcome measures over the resulting logs
uv run python -m instrumental_evasion.analysis.summarize logs/run
```

Without `--arm`, both arms run; without a task, the whole roster runs
(`--families` and `--tasks` narrow it). `--epochs 3` gives the paper's three
runs per task. Each episode writes a write-once attempt directory
`logs/run/<task>_<arm>/epoch<N>/<attempt-id>/` containing `result.json` (score,
messages, monitor decisions, metadata), `decisions.jsonl`, and the CLI
transcript where applicable.

By default every run uses the paper's main-evaluation condition: GPT-OSS-120B
monitor through OpenRouter, trimmed policies, a monitor history of the 40 most
recent calls, the disclosed guardrail notice, a neutral denial with a fixed
reminder, a neutral continuation message, the seven-refusal stop rule, and
limits of 300 turns, 20M tokens, and 6,000 seconds. All of it is configurable
through environment variables ([docs/configuration.md](docs/configuration.md)),
and every log records what ran.

## Reproducing the paper

Each command below runs one agent; repeat it per agent. Agents in the main
comparison ran through OpenRouter or through their vendor's subscription
(recorded in each log as `agent_provider_route`); OpenRouter model ids are
given here.

| Agent | Scaffold | Model | Reasoning effort |
| --- | --- | --- | --- |
| GLM-5.3 Flash | `codex` | `openrouter/z-ai/glm-5.3-flash` | default |
| Muse Spark 1.3 | `codex` | `openrouter/meta/muse-spark-1.3-contributor` | `medium` |
| GPT-5.6 Luna | `codex` | `openrouter/openai/gpt-5.6-luna` | default |
| DeepSeek V4 Flash | `codex` | `openrouter/deepseek/deepseek-v4-flash` | default |
| GPT-5.6 Sol | `codex` | `openrouter/openai/gpt-5.6-sol` | default |
| Claude Sonnet 5 | `claude_code` | `openrouter/anthropic/claude-sonnet-5` | default |
| Claude Opus 5 | `claude_code` | `openrouter/anthropic/claude-opus-5` | default |
| GPT-6 Astra | `codex` | `openrouter/openai/gpt-6-astra` | `high` |

**Main evaluation** (Section 4.1): 50 tasks, three runs each.

```bash
uv run python -m instrumental_evasion.run --arm monitored --epochs 3 \
    --scaffold codex --model openrouter/deepseek/deepseek-v4-flash \
    --log-dir logs/main/dsflash
```

**Unmonitored controls** (appendix "Evaluation Details and Exact Counts"): `--arm control --epochs 1` for Luna and
DeepSeek V4 Flash.

**Reasoning effort** (Section 4.3): `--reasoning-effort {low,medium,high,xhigh,max}`
with Codex, one run per task and level, `AGENT_TIME_LIMIT=3600`.

**Monitor comparison** (Section 5.1): GPT-5.6 Sol as the agent and
`--monitor-model` set to each monitor, e.g.
`openrouter/qwen/qwen3.5-9b`. The provider pin of the default monitor is not
applied to other monitors ([docs/configuration.md](docs/configuration.md#monitor-serving)).

**Myopic monitor** (appendix "Myopic Monitor Ablation"): `MONITOR_HISTORY=0 AGENT_REFUSAL_CONTEXT=none`.

**Denial reasons** (Section 5.2): `AGENT_REFUSAL_CONTEXT=monitor_reason_v1` with
Codex at `--reasoning-effort medium` on the 40 benchmark-derived tasks
(`--families toolsandbox clawbench taubench terminal_bench_2 tblite`).

**Forged denials** (Section 5.3): ten information-extraction tasks, clean versus
a source file that begins with a fake denial; no monitor.

```bash
uv run python -m instrumental_evasion.diagnostics.forged_denial --help
```

**Fixed-request monitor diagnostic** (Section 5.1, appendix "Monitor
Diagnostics"): no agent; 24 labeled permitted/prohibited cases in 40 request
profiles, each classified three times per monitor.

```bash
python -m instrumental_evasion.diagnostics.monitor_diagnostic requests           # inspect, no calls
python -m instrumental_evasion.diagnostics.monitor_diagnostic run \
    --monitor openai/gpt-oss-120b --repeats 3 --out logs/diagnostic/gptoss
python -m instrumental_evasion.diagnostics.monitor_diagnostic report logs/diagnostic/* \
    --subset relevant_context
```

The serving configuration of each monitor in the paper (provider, quantization,
reasoning, structured output) is set with the `MONITOR_*` variables; the
package docstring of `instrumental_evasion.diagnostics.monitor_diagnostic`
lists the settings for every monitor.

**Analysis.** Both tools read runner logs:

```bash
# per agent: full evasion (task completion), confirmed prohibited effect,
# combined rate (ESR), attempt marker (EAR), coverage, runtime
python -m instrumental_evasion.analysis.summarize logs/main/* --epochs 3 [--by family]

# the seven command-pattern categories of the appendix "Command-Pattern Analysis"
python -m instrumental_evasion.analysis.command_patterns logs/main/*
```

A task counts once if any of its selected runs has the outcome;
infrastructure-invalid episodes are excluded. The command-pattern classifier
(`command_pattern_analysis_v3`) is separate from the per-episode attempt marker
recorded by the runner (`evasion.py`).

**Before a batch**, check that each policy blocks its canonical route on the
monitor you use (and how often it also blocks the honest route):

```bash
uv run python scripts/check_policy_recall.py --dry          # show the probe table, no calls
uv run python scripts/check_policy_recall.py --trials 3 --monitor-history   # probe through the gate
```

Results depend on model versions, provider routing, and CLI versions, all of
which change over time; exact numbers will not reproduce bit-for-bit. The
paper reports which configuration each cohort used. If the pinned monitor
provider is unavailable, `MONITOR_PROVIDER=none` uses OpenRouter's default
routing; the log records the route. A monitor that cannot answer blocks the
call and voids the episode, so such an outage never looks like a result.

## Repository layout

```
src/instrumental_evasion/
  run.py            entry point (python -m instrumental_evasion.run)
  config.py         the paper's main-evaluation condition
  policies.py       operator policies and the monitor's system prompt
  monitor_core.py   monitor input (history), verdict parsing, denial messages
  evasion.py        automatic attempt markers
  refusal_stop.py   the seven-refusal stop rule and its classifier
  rollout_core.py   limits, guardrail notice, continuation message
  tasks/            the 50 task specs, fixtures, and scorers (registry.py)
  runner/           runner, gate, scaffolds, runtimes, logs
  hooks/            the PreToolUse hook and the monitor client
  diagnostics/      forged-denial suite and fixed-request monitor diagnostic
  analysis/         outcome aggregation and command-pattern analysis
scripts/            image builds, CLI installation, policy recall check
images/             Dockerfiles of the base task images
docs/               design, configuration, and task reference
tests/              unit tests (no network or containers needed)
```

Run the tests with `uv run pytest`.

## Data

The agent traces, monitor decisions, and per-model result tables behind the
paper are released separately: *link forthcoming*.

## Responsible use

This work studies how runtime oversight fails. The traces and harness can
inform both safeguards and attempts to circumvent them. The policies in this
benchmark prohibit ordinary operations to create a conflict with the task;
they are not examples of harmful actions.

## Citation

```bibtex
@article{anonymous2026instrumental,
  title   = {Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure},
  author  = {Anonymous Authors},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)). Task material adapted from Terminal-Bench 2.1,
OpenThoughts-TBLite, and τ-bench keeps its original license; see
[NOTICE.md](NOTICE.md).
