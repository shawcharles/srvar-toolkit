# Limitations and performance

This project targets transparency and reproducibility and is currently in an **alpha** stage.

## Modeling limitations

- **Stochastic volatility coverage is still evolving**: diagonal SV, triangular (time-invariant correlation) SV, and factor SV (time-varying full covariance) are supported. Factor SV is currently limited to `prior.family: "niw"` with RW dynamics; ELB, steady-state, and robust shocks are supported.
- **Structural analysis coverage is partial**: reduced-form, Cholesky, and sign-restricted IRFs are supported via `srvar.analysis` (plus FEVD and Cholesky historical decompositions), but other workflows (e.g. sign-restricted historical decompositions) are not yet first-class.
- **Conditional/scenario forecasts are limited**: `srvar.scenario.conditional_forecast` currently supports homoskedastic VARs. For ELB models, conditioning is applied to the latent (unfloored) process.
- **ELB treatment**: ELB handling is implemented via latent shadow-rate augmentation for selected series.
- **Robust shocks limitations**: Student‑t and outlier-mixture innovations are supported for homoskedastic VARs and for factor SV. They are not yet supported for diagonal/triangular SV; ELB/steady-state combinations require factor SV.

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
