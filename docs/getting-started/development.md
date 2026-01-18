# Development workflow

This page covers a minimal local workflow for developing `srvar-toolkit`.

## Install (editable + dev tools)

From the repository root:

```bash
python -m pip install -e ".[dev,cli,fred,docs,plot]"
```

## Run tests

```bash
pytest
```

## Lint and format

Check lint:

```bash
ruff check srvar tests
```

Auto-fix import sorting + simple issues:

```bash
ruff check --fix srvar tests
```

Format code:

```bash
black srvar tests
```

## Type checking (optional)

```bash
mypy srvar
```

## Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

