# Installation

## Reproducible local environment (recommended)

If you're concerned about cross-platform differences (macOS/Windows/Linux), install into a fresh virtual environment.

```bash
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
# .venv\Scripts\Activate.ps1

python -m pip install -U pip

# CLI + YAML + FRED fetch
python -m pip install -e ".[cli,fred]"
```

For development (tests + docs + plotting):

```bash
python -m pip install -e ".[dev,cli,fred,docs,plot]"
```

## Install from source

```bash
pip install -e .
```

## Optional: plotting dependencies

```bash
pip install -e ".[plot]"
```

## Optional: CLI + YAML config support

The config-driven CLI (`srvar validate`, `srvar run`) requires PyYAML.

```bash
pip install -e ".[cli]"
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
