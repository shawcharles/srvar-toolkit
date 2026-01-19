# Project Overview

`srvar-toolkit` is a research-friendly Python library for Bayesian VAR / shadow-rate VAR
modeling and macroeconomic forecasting, with a clean programmatic API (`srvar.api.fit/forecast`)
plus a config-driven CLI for data fetching, fit/forecast runs, and rolling backtests
(metrics + artifacts).

## Repository Structure

- `srvar/` — core library code (models, samplers, evaluation, CLI entrypoints).
- `tests/` — pytest suite (unit + lightweight integration tests; includes property-based tests).
- `docs/` — Sphinx/MyST documentation site sources (plus `_build/` outputs).
- `examples/` — small runnable scripts demonstrating common workflows.
- `config/` — example YAML configs for `srvar run`, `srvar backtest`, and `srvar fetch-fred`.
- `data/` — sample or cached data inputs used by examples/replication.
- `papers/` — replication harnesses and paper-specific scripts/assets.
- `functions/` — MATLAB reference code and utilities (replication support).
- `scripts/` — utility scripts (e.g., memory benchmarks).
- `memory-bank/` — project planning docs (roadmaps, design notes).
- `plots/` — generated plot assets used in README/docs.
- `outputs/` — generated run outputs/artifacts (local; not source of truth).
- `arXiv-preprint/` — paper/preprint assets.
- `.windsurf/` — IDE/assistant context files (not library runtime code).
- `.benchmarks/` — benchmark-related artifacts/configs.

## Build & Development Commands

### Install (editable + dev tools)

```bash
python -m pip install -e ".[dev,cli,fred,docs,plot]"
```

### Run tests

```bash
pytest
```

### Lint and format

```bash
ruff check srvar tests
ruff check --fix srvar tests
black srvar tests
```

Optional (matches README):

```bash
ruff format --check
```

### Type checking (optional)

```bash
mypy srvar
```

### Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

Or:

```bash
make -C docs html
```

### Run the CLI (config-driven)

```bash
srvar validate config/minimal_config.yaml
srvar run config/demo_config.yaml
srvar backtest config/backtest_demo_config.yaml
```

### Fetch FRED data (optional; requires API key and `.[fred]`)

```bash
srvar fetch-fred config/fetch_fred_demo_config.yaml --dry-run
srvar fetch-fred config/fetch_fred_demo_config.yaml --validate-series
```

### Debugging

```bash
pytest -k sv -vv
python -m pdb -m pytest -k test_sv_factor -vv
```

### Deploy / release

> TODO: No release/publish workflow is documented in-repo (PyPI, tags, etc.).

## Code Style & Conventions

- Formatting:
  - Black with `line-length = 100` (see `pyproject.toml`).
- Linting:
  - Ruff with `line-length = 100` and rulesets `E/F/I/B/UP/NPY` (see `pyproject.toml`).
  - `E501` is ignored; prefer keeping lines ≤ 100 anyway.
- Types:
  - `mypy srvar` is configured with `ignore_missing_imports` for `pandas/scipy/numba`.
  - Prefer type annotations for public APIs; internal helpers may be partially typed.
- Naming:
  - `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for
    constants.
  - “Internal” functions use a leading underscore (e.g., `_fit_*` in samplers).
- Docstrings:
  - Public functions/classes should use NumPy-style docstrings (see `CONTRIBUTING.md`).
- Commit messages:
  - Template example in `README.md`: `feat: add amazing feature`.
  - > TODO: Formal commit convention (e.g., Conventional Commits) is not explicitly required.

## Architecture Notes

```mermaid
flowchart LR
  subgraph User
    cfg[config/*.yaml]
    code[Python API calls]
  end

  cfg --> cli[srvar CLI]
  cli --> config[srvar.config]
  config --> ds[Dataset]
  config --> spec[ModelSpec/PriorSpec/SamplerConfig]

  code --> api[srvar.api]
  ds --> api
  spec --> api

  api --> samplers[srvar.samplers_*]
  samplers --> fit[FitResult]
  fit --> forecast[srvar.api.forecast]
  forecast --> fc[ForecastResult]

  cli --> backtest[srvar.backtest]
  backtest --> api
  backtest --> eval[srvar.evaluation + srvar.metrics]
  eval --> metrics[metrics.csv]

  fit --> analysis[srvar.analysis (IRF/FEVD/HD)]
```

Key components:

1. `srvar.api` is the public façade (`fit`, `forecast`) over sampler implementations.
2. `srvar.samplers_*` implement model variants (homoskedastic, SV diagonal, SV triangular,
   factor SV (FSV), ELB augmentation) and return a `FitResult` with draws.
3. `srvar.backtest` orchestrates rolling/expanding runs from YAML configs and calls
   `srvar.evaluation/srvar.metrics` to compute scoring rules and summaries.
4. `srvar.data` provides a simple `Dataset` container and optional FRED fetching utilities.
5. `srvar.analysis` implements structural analysis on top of posterior draws (IRFs/FEVD/HD).

## Testing Strategy

- Unit tests: `tests/` targets pure functions (linear algebra, metrics, transforms).
- Integration tests: small “fit + forecast” runs with low draw counts for key model variants.
- Property-based tests: Hypothesis is used for invariants and edge cases
  (`tests/test_property_based.py`).
- Network isolation:
  - FRED tests mock network boundaries; avoid real HTTP calls in tests.
- CI:
  - > TODO: No CI workflow is configured in this repo; run the local commands above.
  - If CI is added later, it should run at least `pytest` and `ruff check srvar tests`.

## Security & Compliance

- Secrets:
  - Do not commit API keys. FRED access uses `FRED_API_KEY` (see `srvar/data/fred.py`).
  - A `.env` file exists for local development; treat it as local-only.
- Dependencies:
  - Core runtime deps are `numpy`, `scipy`, `pandas` (see `pyproject.toml`).
  - Optional extras add CLI/docs/plotting/FRED/acceleration (`numba`) and labeled outputs
    (`xarray`, `arviz`).
- Vulnerability reporting:
  - Follow `SECURITY.md` (email maintainer listed in `pyproject.toml`).
- License:
  - MIT (see `LICENSE`).
- Scanning / compliance automation:
  - > TODO: No dependency scanning or SBOM workflow is documented in-repo.

## Agent Guardrails

- Prefer small, test-backed changes; do not refactor unrelated areas “for cleanliness”.
- Avoid editing generated or large binary assets unless explicitly requested:
  - `outputs/`, `plots/`, `docs/_build/`, `paper.pdf`, and most of `papers/**` figures/artifacts.
- Do not add mandatory heavyweight dependencies; use optional extras in `pyproject.toml`.
- Do not introduce network calls in tests; mock external APIs.
- Keep numerical code reproducible:
  - accept an explicit `rng` where applicable and prefer `np.random.default_rng`.
- For changes affecting scientific results, require human review of:
  - model math (samplers), evaluation conventions, and replication scripts.
- If a subtree includes its own `AGENTS.md`, follow the most specific instructions for files in
  that subtree.

## Extensibility Hooks

- Optional extras:
  - `.[cli]`, `.[docs]`, `.[plot]`, `.[fred]`, `.[accel]`, `.[xarray]`, `.[arviz]`
  - Prefer adding new optional integrations as extras, not core deps.
- Environment variables:
  - `FRED_API_KEY` (FRED access; required for `srvar fetch-fred` unless config provides a key).
  - `SRVAR_USE_NUMBA` (`0/1`, enables Numba-accelerated SV mixture sampling in `srvar/sv.py`).
- Adding new model variants:
  1. Extend `ModelSpec` / `VolatilitySpec` as needed.
  2. Implement a new sampler module (pattern: `srvar/samplers_*.py` returning `FitResult`).
  3. Wire dispatch in `srvar.api.fit` and update docs/config reference.
  4. Add tests in `tests/` (small draws, deterministic seed).
- Adding new evaluation metrics:
  - Implement in `srvar/metrics.py` and wire through `srvar/evaluation.py` + config parsing.

## Further Reading

- `README.md` — overview, features, quickstart, CLI usage.
- `docs/index.md` — documentation home.
- `docs/getting-started/development.md` — local development workflow.
- `docs/user-guide/configuration-reference.md` — YAML config reference.
- `docs/user-guide/evaluation.md` — evaluation/scoring conventions.
- `CONTRIBUTING.md` — contribution workflow and style guidance.
- `SECURITY.md` — vulnerability reporting process.
- `memory-bank/development/` — internal roadmaps and design notes (e.g., FSV design doc).
- `papers/` — replication harnesses and paper-specific workflows.
