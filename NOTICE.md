# Third-party material

The code in this repository is released under the MIT License (`LICENSE`).
Some task material is adapted from public benchmarks and keeps its original
license.

| Material | Location | Source | License |
| --- | --- | --- | --- |
| Instructions and verifiers of nine Terminal-Bench 2.1 tasks | `src/instrumental_evasion/tasks/terminal_bench_2/tasks/` | [harbor-framework/terminal-bench-2](https://github.com/harbor-framework/terminal-bench-2) at `2fd12b88aafdd04a52c298e3940bcb189f9766d6` | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| Instructions and verifiers of five OpenThoughts-TBLite tasks | `src/instrumental_evasion/tasks/terminal_bench_lite/tasks/` | [open-thoughts/OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite) at `5c37b41f00ce04719a4453061076ae9f46b74b7d` | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| tau-bench retail domain: tools, data, policy wiki, and tasks | `src/instrumental_evasion/tasks/taubench/vendor/` | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) | MIT (`src/instrumental_evasion/tasks/taubench/vendor/LICENSE`) |

The container images for the Terminal-Bench 2.1 and OpenThoughts-TBLite tasks
are built from the upstream repositories at the commits above (see
`scripts/build_images.sh`); they are not redistributed here.

The ToolSandbox and ClawBench tasks are re-implementations of scenarios from
[apple/ToolSandbox](https://github.com/apple/ToolSandbox) (Lu et al., 2025) and
ClawBench (Zhang et al., 2026) as terminal tasks with generated fixtures. No
upstream files are included. Please cite the original benchmarks when you use
these tasks.
