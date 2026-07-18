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

# CLI + YAML + FRED fetch + local Excel benchmark workbooks
python -m pip install -e ".[cli,fred,excel]"
```

For development (tests + docs + plotting):

```bash
python -m pip install -e ".[dev,cli,fred,docs,plot,excel]"
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

## Optional: Excel benchmark workbooks

The local vintage-macro and term-spread/NFCI/Wu-Xia benchmark preparation scripts read tracked
Excel workbooks. Install the Excel extra before running them:

```bash
pip install -e ".[excel]"
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
