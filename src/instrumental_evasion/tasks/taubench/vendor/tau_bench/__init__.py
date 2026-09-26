# Vendored from sierra-research/tau-bench. Upstream's __init__ re-exports Env
# and MockRetailDomainEnv; both pull in the litellm-backed user simulator we
# replace, so this package is deliberately a namespace only. Import the
# submodules you need (`tau_bench.types`, `tau_bench.envs.<domain>.tools`, ...).
