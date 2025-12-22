# Glossary

This page defines the main acronyms and terms used throughout the toolkit.

## ELB

**Effective lower bound**. In this toolkit, ELB refers to a censoring constraint applied to selected observed series (typically short-term policy rates). When the observed series is at or below the bound (up to a tolerance), the model treats the “true” value as a *latent* shadow value subject to the constraint.

## Shadow rate

A **shadow rate** is a latent (unobserved) policy rate intended to represent the stance of monetary policy when observed policy rates are constrained by an ELB.

## VAR

**Vector autoregression**. A multivariate time-series model where each variable is regressed on its own lags and the lags of the other variables.

## BVAR

**Bayesian VAR**. A VAR estimated with Bayesian priors over coefficients and innovation covariance.

## NIW

**Normal-Inverse-Wishart**. A conjugate prior for VAR coefficients and the innovation covariance matrix.

## Minnesota prior

A structured shrinkage prior for VAR coefficients, typically shrinking toward a random-walk / white-noise baseline with lag decay and cross-variable shrinkage.

## SSVS

**Stochastic search variable selection**. A spike-and-slab prior with inclusion indicators that stochastically include/exclude predictor rows.

## SV / SVRW

**Stochastic volatility**. A model for time-varying residual variances.

**SVRW** means the log-variance follows a random walk. In this toolkit, volatility is currently **diagonal** (series-specific variances, no time-varying covariances).

## KSC

**Kim-Shephard-Chib** auxiliary mixture approximation for the log-\(\chi^2\) distribution used in many stochastic volatility samplers.

## Gibbs sampler

An MCMC algorithm that iteratively samples from conditional distributions of each parameter block.
