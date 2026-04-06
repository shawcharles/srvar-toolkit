# Concepts

## Shadow rates

A shadow rate is a latent policy-rate concept used when observed policy rates are constrained by an effective lower bound (ELB).

## ELB (shadow-rate) data augmentation

When ELB constraints are enabled, the model treats ELB-constrained observations as latent and samples shadow values subject to the bound.

## Stochastic volatility (SVRW)

When volatility is enabled, the toolkit supports diagonal stochastic volatility with random-walk
or AR(1) log-variance dynamics, plus triangular and factor covariance variants on the supported
sampler paths.

## Minnesota prior modes

The toolkit now distinguishes three Minnesota-style shrinkage paths:

- `niw_minnesota_legacy`: the historical compatibility path and default reproducibility baseline
- `niw_minnesota_canonical`: the explicit equation-wise canonical path where supported
- `niw_minnesota_tempered`: an experimental diagonal-SV-only bridge between the legacy and canonical variance maps

## Background

This toolkit is inspired by the SRVAR methodology in Grammatikopoulos (2025). For the original MATLAB replication code, see:
https://github.com/MichaelGrammmatikopoulos/MLSRVARs
