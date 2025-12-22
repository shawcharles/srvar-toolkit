# Limitations and performance

This project targets transparency and reproducibility and is currently in an **alpha** stage.

## Modeling limitations

- **Diagonal stochastic volatility only**: volatility is modeled per-series; there is no time-varying covariance / full-covariance SV.
- **VAR-only structure**: the high-level API currently focuses on VAR(p) models with optional ELB and SV. More structural parameterizations (e.g. steady-state VARs) are not yet implemented.
- **ELB treatment**: ELB handling is implemented via latent shadow-rate augmentation for selected series.

## Statistical limitations / caveats

- **MCMC diagnostics are your responsibility**: the toolkit returns draws, but does not currently ship full diagnostic tooling (R-hat, ESS, trace diagnostics). You should validate convergence and mixing.
- **Sensitivity to prior settings**: results can change meaningfully with Minnesota hyperparameters, SSVS spike/slab variances, and SV priors.

## Performance considerations

Runtime depends primarily on:

- `T`: number of observations
- `N`: number of variables
- `p`: lag order
- `draws`, `burn_in`, `thin`: sampler configuration
- model features enabled (ELB and SV are more expensive than conjugate NIW)

Rules of thumb:

- Start with small samplers to validate data plumbing and model stability.
- Increase draws only once the model runs end-to-end and outputs look reasonable.

## Numerical considerations

- Some numerical constants/initializations are chosen for stability (e.g., latent ELB initialization uses a small offset below the bound).
- The SV implementation uses an auxiliary mixture approximation (KSC) and banded linear algebra; extreme data scaling can still cause numerical issues.
