"""Entry point: `python -m instrumental_evasion.run --help`.

Runs the evaluation with `instrumental_evasion.runner.run`. Unless overridden,
every episode uses the paper's main-evaluation condition
(`instrumental_evasion.config`).
"""

from instrumental_evasion.runner.run import main

if __name__ == "__main__":
    raise SystemExit(main())
