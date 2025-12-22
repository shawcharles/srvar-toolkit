# MCMC overview

This page summarises the Markov chain Monte Carlo (MCMC) logic used in the toolkit. The implementation is a pragmatic Gibbs sampler for a reduced-form VAR with optional ELB augmentation and diagonal stochastic volatility.

## BVAR (NIW) without ELB/SV

1. Compute NIW posterior parameters.
2. Sample $(\beta, \Sigma)$ from the matrix-normal inverse-Wishart posterior.

## ELB only (shadow-rate augmentation)

At each iteration:

1. Sample VAR parameters $(\beta, \Sigma)$ conditional on the current latent series.
2. For each ELB-constrained observation, sample a latent shadow value from its conditional distribution subject to the bound.

## SV only (diagonal SVRW)

At each iteration:

1. Sample coefficients $\beta$ conditional on the current log-volatilities $h$.
2. Sample log-volatilities $h$ using the auxiliary-mixture approach.
3. Sample SV hyperparameters (initial log-volatility and innovation variance).

## SV + ELB (combined)

At each iteration:

1. Sample $\beta$ conditional on $h$ and the current latent series.
2. Sample ELB latent values conditional on $\beta$ and $h$.
3. Sample $h$ conditional on residuals.
4. Sample SV hyperparameters.

## Practical notes

- Use `burn_in` and `thin` to control storage and reduce autocorrelation in retained draws.
- For long runs, profile your model and consider multiple shorter chains rather than a single very long chain.

Related:
- {doc}`../getting-started/quickstart`
- {doc}`../reference/api`
