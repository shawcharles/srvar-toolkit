# srvar-toolkit

**Shadow-rate VAR toolkit for Bayesian macroeconomic forecasting in pure Python.**

A lightweight, tested implementation of Shadow-Rate Vector Autoregression models with ELB data augmentation, multiple Minnesota-style shrinkage paths, stochastic volatility, structural analysis, and a config-driven CLI workflow with backtesting. The library also includes FRED data fetching for building reproducible macro datasets.

```bash
pip install git+https://github.com/shawcharles/srvar-toolkit.git
```

```{note}
Version `0.3.0` adds explicit legacy/canonical/tempered Minnesota prior modes, missing-data-safe
evaluation/plotting, and a larger benchmark/diagnostic toolkit for comparing Minnesota variants.
```

---

````{grid} 2
:gutter: 3

```{grid-item-card} Getting Started
:link: getting-started/installation
:link-type: doc

Installation instructions and a quickstart tutorial to fit your first Bayesian VAR.

+++
[Installation](getting-started/installation) · [Quickstart](getting-started/quickstart)
```

```{grid-item-card} User Guide
:link: user-guide/concepts
:link-type: doc

Core concepts, configuration options, glossary of terms, and known limitations.

+++
[Concepts](user-guide/concepts) · [Configuration](user-guide/configuration) · [FAQ](FAQ)
```

```{grid-item-card} Theory
:link: theory/shadow-rate-var
:link-type: doc

Statistical methodology: shadow-rate VARs, ELB constraints, stochastic volatility, and MCMC sampling.

+++
[Shadow-Rate VAR](theory/shadow-rate-var) · [Stochastic Volatility](theory/stochastic-volatility)
```

```{grid-item-card} API Reference
:link: reference/index
:link-type: doc

Complete function and class documentation with type signatures and examples.

+++
[Full Reference](reference/index)
```

````

---

## Features

| Component | Description | Status |
|-----------|-------------|--------|
| Conjugate BVAR (NIW) | Closed-form posterior updates | ✅ Supported |
| Legacy Minnesota-style NIW shrinkage | Historical prior construction with lag decay (non-canonical) | ✅ Supported |
| Canonical Minnesota shrinkage | Equation-wise own-vs-cross shrinkage for homoskedastic and diagonal SV models | ✅ Supported |
| Tempered Minnesota bridge | Experimental legacy-to-canonical bridge for diagonal SV | ⚠️ Experimental |
| Shadow-Rate / ELB | Latent shadow-rate sampling | ✅ Supported |
| Stochastic Volatility | Diagonal SV (RW/AR1), triangular covariance SV, and factor SV | ✅ Supported |
| Variable Selection (SSVS) | Spike-and-slab priors | ✅ Supported |
| Bayesian LASSO (BLASSO) | Shrinkage prior for VAR coefficients | ✅ Supported |
| Steady-State VAR (SSP) | Parameterize intercept via steady-state mean `mu` | ✅ Supported |
| Forecasting | Posterior predictive simulation | ✅ Supported |
| Backtesting + Evaluation | Rolling/expanding backtests with metrics, plots, and streaming evaluation | ✅ Supported |
| Conditional forecasts | Scenario forecasts with hard constraints | ✅ Supported |
| Structural IRFs / FEVD / HD | Cholesky and sign-restricted structural analysis from posterior draws | ✅ Supported |

---

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/development
getting-started/releasing
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

user-guide/concepts
user-guide/glossary
user-guide/configuration
user-guide/configuration-reference
user-guide/model-recipes
user-guide/evaluation
user-guide/backtesting
user-guide/limitations
FAQ
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Theory

theory/shadow-rate-var
theory/elb
theory/stochastic-volatility
theory/variable-selection
theory/mcmc
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

reference/index
