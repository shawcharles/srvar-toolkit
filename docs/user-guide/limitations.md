# Limitations and performance

This project targets transparency and reproducibility and is currently in an **alpha** stage.

## Modeling limitations

- **Stochastic volatility is not “fully general”**: the toolkit supports diagonal SV and a triangular factorization with time-invariant correlations, but it does not currently implement fully time-varying correlation / covariance dynamics.
- **Structural analysis coverage is partial**: reduced-form, Cholesky, and sign-restricted IRFs are supported via `srvar.analysis` (and FEVD from structural IRFs), but historical decompositions are not yet first-class workflows.
- **Conditional/scenario forecasts are limited**: `srvar.scenario.conditional_forecast` currently supports homoskedastic VARs. For ELB models, conditioning is applied to the latent (unfloored) process.
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

Backtesting can also be memory-heavy. For long backtests, prefer streaming evaluation:

- `output.save_plots: false`
- `output.store_forecasts_in_memory: false`

## Numerical considerations

- Some numerical constants/initializations are chosen for stability (e.g., latent ELB initialization uses a small offset below the bound).
- The SV implementation uses an auxiliary mixture approximation (KSC) and banded linear algebra; extreme data scaling can still cause numerical issues.
