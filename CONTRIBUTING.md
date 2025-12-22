# Contributing

Thanks for your interest in contributing to **srvar-toolkit**.

This project aims to be a clean, transparent, research-friendly implementation of Bayesian VAR / SRVAR methods.

## Development environment

### Requirements

- Python 3.11+

### Install

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

### Run tests

```bash
pytest
```

### Lint / format

```bash
ruff check srvar/
ruff format srvar/
```

### Type checking

```bash
mypy srvar/
```

### Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

## Pull requests

Please ensure:

- Tests pass (`pytest`).
- Formatting/linting passes (`ruff`).
- Any public API changes include docstring updates and docs updates.
- New features include at least one test.

## Style guidelines

- Prefer small, composable functions.
- Keep public functions/classes documented with NumPy-style docstrings.
- Maintain backwards compatibility where practical (this is an alpha project, but stability is still valued).

## Reporting issues

Please include:

- Minimal reproducible example
- Package version (`srvar.__version__`)
- Python version
- OS and installation method
