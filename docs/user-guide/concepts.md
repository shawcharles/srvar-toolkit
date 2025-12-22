# Concepts

## Shadow rates

A shadow rate is a latent policy-rate concept used when observed policy rates are constrained by an effective lower bound (ELB).

## ELB (shadow-rate) data augmentation

When ELB constraints are enabled, the model treats ELB-constrained observations as latent and samples shadow values subject to the bound.

## Stochastic volatility (SVRW)

When volatility is enabled, each series is modeled with diagonal stochastic volatility following a random-walk evolution.

## Background

This toolkit is inspired by the SRVAR methodology in Grammatikopoulos (2025). For the original MATLAB replication code, see:
https://github.com/MichaelGrammmatikopoulos/MLSRVARs
