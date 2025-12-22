# Variable selection and shrinkage (SSVS)

## Why shrinkage?

Bayesian VARs can include many coefficients (lags \times variables \times equations). Shrinkage priors regularise the model and can improve forecast performance.

## SSVS intuition

Stochastic search variable selection (SSVS) places a mixture prior on coefficients so that, conditional on an inclusion indicator, a coefficient is either:
- heavily shrunk towards zero ("spike"), or
- allowed to vary more freely ("slab").

## In this toolkit

The Python toolkit supports an SSVS-style prior for reduced-form VAR coefficients:
- a latent boolean vector `gamma` controls which coefficient *rows* are in the spike or slab regime,
- Gibbs updates alternate between sampling VAR parameters and updating `gamma`.

This implementation is designed for practical forecasting workflows rather than reproducing every shrinkage variant in the paper.

Related:
- {doc}`../reference/api`
- {doc}`mcmc`
