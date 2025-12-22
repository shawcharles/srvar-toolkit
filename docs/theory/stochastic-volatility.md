# Stochastic volatility (SVRW)

## Overview

Stochastic volatility (SV) allows the variance of VAR residuals to change over time. This is important in macro/financial data where volatility can shift markedly across regimes.

The paper uses SV ideas common in the Bayesian VAR literature. In this toolkit, SV is implemented as **diagonal** volatility with a **random-walk** evolution (SVRW).

## Model sketch

For each series $j$ the log-variance evolves as a random walk:

$$
 h_{t,j} = h_{t-1,j} + \eta_{t,j}, \qquad \eta_{t,j} \sim \mathcal{N}(0, \sigma_{\eta,j}^2).
$$

Conditional on $h_{t,j}$, residuals are Gaussian with variance $\exp(h_{t,j})$.

## Inference approach (KSC mixture)

The toolkit uses a standard auxiliary-mixture method (Kim, Shephard and Chib) to sample log-volatilities efficiently by approximating the log-$\chi^2$ distribution with a discrete mixture.

Implementation notes:
- volatility is **diagonal** (no time-varying covariances), to keep computation manageable;
- the log-volatility state is sampled with a banded precision representation.

Related:
- {doc}`theory/mcmc`
