# Stochastic volatility (SV)

## Overview

Stochastic volatility (SV) allows the variance of VAR residuals to change over time. This is important in macro/financial data where volatility can shift markedly across regimes.

This toolkit implements SV variants commonly used in the Bayesian VAR literature:

- **log-volatility dynamics**: random walk (`SVRW`) or AR(1)
- **residual covariance**: diagonal covariance (independent shocks) or a triangular factorization with time-invariant correlations (CCC-style)

## Model sketch

Let $h_{t,j}$ be the log-variance state for series $j$ at time $t$.

### Random walk (SVRW)

For each series $j$ the log-variance evolves as:

$$
 h_{t,j} = h_{t-1,j} + \eta_{t,j}, \qquad \eta_{t,j} \sim \mathcal{N}(0, \sigma_{\eta,j}^2).
$$

### AR(1)

Optionally, the log-variance follows an AR(1):

$$
 h_{t,j} = \gamma_{0,j} + \phi_j h_{t-1,j} + \eta_{t,j}.
$$

### Residual covariance structure

By default, conditional residual covariance is diagonal:

$$
\Sigma_t = \mathrm{diag}(\exp(h_t)).
$$

With the triangular factorization, the model uses:

$$
\Sigma_t = Q^{-1}\,\mathrm{diag}(\exp(h_t))\,(Q^{-1})',
$$

where $Q$ is upper-triangular with ones on the diagonal. This yields **time-varying variances** with a **time-invariant correlation structure**.

## Inference approach (KSC mixture)

The toolkit uses a standard auxiliary-mixture method (Kim, Shephard and Chib) to sample log-volatilities efficiently by approximating the log-$\chi^2$ distribution with a discrete mixture.

Implementation notes:
- the log-volatility state is sampled with a banded precision representation;
- when `covariance="triangular"`, the triangular factor is updated via a Gaussian prior on off-diagonal elements.

Related:
- {doc}`mcmc`
