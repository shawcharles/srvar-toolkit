# srvar-toolkit

**Shadow-rate VAR toolkit for Bayesian macroeconomic forecasting in pure Python.**

A lightweight, tested implementation of Shadow-Rate Vector Autoregression models with stochastic volatility, Minnesota shrinkage priors, and variable selection.

```bash
pip install git+https://github.com/shawcharles/srvar-toolkit.git
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
| Minnesota Shrinkage | Prior construction with lag decay | ✅ Supported |
| Shadow-Rate / ELB | Latent shadow-rate sampling | ✅ Supported |
| Stochastic Volatility | Diagonal log-variance random-walk | ✅ Supported |
| Variable Selection (SSVS) | Spike-and-slab priors | ✅ Supported |
| Bayesian LASSO (BLASSO) | Shrinkage prior for VAR coefficients | ✅ Supported |
| Forecasting | Posterior predictive simulation | ✅ Supported |

---

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

user-guide/concepts
user-guide/glossary
user-guide/configuration
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
