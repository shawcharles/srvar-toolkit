# Installation

## Install from source

```bash
pip install -e .
```

## Optional: plotting dependencies

```bash
pip install -e ".[plot]"
```

## Optional: FRED/ALFRED data access

```bash
pip install -e ".[fred]"
```

## Install docs dependencies

```bash
pip install -e ".[docs]"
```

## Build the docs locally

From the repository root:

```bash
sphinx-build -b html docs docs/_build/html
```
