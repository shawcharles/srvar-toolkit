# Effective lower bound (ELB) data augmentation

## What is the ELB constraint?

For some interest-rate series, observations are effectively censored at a lower bound (e.g. 0 or slightly negative). When the ELB binds, the observed series no longer behaves like an unconstrained Gaussian variable.

## The basic modelling idea

In an ELB/shadow-rate setting, the observed rate $i_t$ is treated as a censored version of an underlying latent series $s_t$ (a shadow rate). One simple relationship is:

$$
 i_t = \max\{\mathrm{ELB},\, s_t\}.
$$

## Data augmentation in MCMC

A standard way to fit these models is **data augmentation**:
- treat ELB-bound observations as latent,
- sample latent values conditional on the VAR parameters and the constraint $s_t \le \mathrm{ELB}$.

In practice, this becomes sampling from a (possibly univariate) **truncated normal** conditional distribution for each constrained time point.

## In this toolkit

When ELB support is enabled via `ElbSpec`, `fit()` uses Gibbs steps that alternate between:
- sampling VAR parameters conditional on the current latent series, and
- sampling latent shadow values for ELB observations conditional on the VAR parameters.

With stochastic volatility enabled, the conditional distribution accounts for time-varying variance through the current log-volatility state.

Related:
- {doc}`stochastic-volatility`
- {doc}`mcmc`
