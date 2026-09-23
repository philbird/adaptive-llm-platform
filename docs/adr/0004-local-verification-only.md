# ADR 0004: verification runs locally, not in hosted CI

Status: accepted, 2026-09-23 (owner decision).

Context: the scaffold shipped a GitHub Actions workflow that ran the checks and SBOM audit on
every push. The owner decided that all verification stays local; the public repository is a
record of reviewed work, not an execution environment.

Decision: the workflow is removed. `make ci` is the single verification gate and runs lint,
formatting, strict typing, unit, contract and integration tests, the SBOM vulnerability audit,
the security and load suites, the resilience drills and the smoke lifecycle. A branch is merged
only after `make ci` passes on the reviewer's machine, and the pull request description records
the run. The specification's "generate an SBOM in CI" is satisfied by `make sbom` inside `make ci`.

Consequences: no third-party runner sees the code or dependencies; verification depends on the
reviewer's environment, so the Python version and `uv` version used must be stated in each pull
request. Optional dependency groups such as future training dependencies are installed and
verified locally only.
