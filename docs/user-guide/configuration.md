# Configuration guide

This page describes how to configure models in **srvar-toolkit**.

## Core objects

Most workflows use four objects:

- `Dataset`: data container (values + variable names + time index)
- `ModelSpec`: model structure (lag order, intercept, ELB, stochastic volatility)
- `PriorSpec`: prior family and hyperparameters (NIW or SSVS)
- `SamplerConfig`: MCMC controls (draws, burn-in, thinning)

## Choosing the lag order `p`

- Larger `p` increases the number of regressors `K` and typically increases runtime.
- A common starting point in macro data is `p=4` (quarterly) or `p=12` (monthly), but you should validate using forecast performance.

## NIW vs SSVS

### NIW (conjugate)

Use NIW when you want fast, stable inference and do not need variable selection.

- Use `PriorSpec.niw_default(k=..., n=...)` for a simple default prior.
- Use `PriorSpec.niw_minnesota(p=..., y=..., include_intercept=...)` to get Minnesota-style shrinkage.

### SSVS

Use SSVS when you want posterior inclusion probabilities over predictors.

- Use `PriorSpec.from_ssvs(k=..., n=..., include_intercept=...)`.
- The intercept can be forced included (`fix_intercept=True`) when an intercept is present.

## Enabling ELB (shadow-rate augmentation)

ELB is controlled by `ModelSpec(elb=ElbSpec(...))`.

Key parameters:

- `ElbSpec.bound`: the bound level
- `ElbSpec.applies_to`: list of variable names to constrain
- `ElbSpec.tol`: tolerance used to decide if an observation is at the bound

In ELB models, the fitted object may contain:

- `FitResult.latent_dataset`: one latent “shadow” path
- `FitResult.latent_draws`: latent draws across kept MCMC iterations

Forecasting returns both observed and latent predictive draws:

- `ForecastResult.draws`: observed draws (ELB floor applied)
- `ForecastResult.latent_draws`: unconstrained latent draws

## Enabling stochastic volatility (SVRW)

Stochastic volatility is controlled by `ModelSpec(volatility=VolatilitySpec(...))`.

- Volatility is currently **diagonal** (per-series variances).
- The log-variance follows a random walk.

In SV models:

- `FitResult.h_draws` contains log-volatility state draws.

## Sampler configuration

`SamplerConfig(draws=..., burn_in=..., thin=...)` controls MCMC.

Rules of thumb:

- Start with small numbers to smoke-test code (`draws=200`, `burn_in=50`, `thin=2`).
- Increase draws once the model runs and basic diagnostics look stable.

## Reproducibility

All user-facing sampling/forecast functions accept `rng: np.random.Generator`.

- Use a fixed seed for reproducibility.
- Prefer passing a dedicated RNG instance rather than relying on global state.
