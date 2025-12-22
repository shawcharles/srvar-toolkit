# Frequently asked questions

## What is a shadow rate?

A shadow rate is a *latent* (unobserved) policy rate intended to represent the stance of monetary policy when the observed policy rate is constrained by an effective lower bound (ELB).

Intuitively:
- When the policy rate is away from the ELB, the observed rate and the shadow rate should be similar.
- When the policy rate is at (or near) the ELB, the shadow rate can move below the bound, capturing additional accommodation that may come from unconventional policy tools.

Related pages:
- {doc}`user-guide/concepts`
- {doc}`theory/elb`

## I can forecast the shadow rate. Why might that be useful?

If your model produces credible shadow-rate forecasts, they can be useful as a summary of the *future stance of monetary policy*, particularly during ELB episodes. Depending on the strategy and time horizon, that can feed into decisions about:

- **Rates and duration risk**: expected easing/tightening often matters for yield-curve positioning.
- **Cross-market relative value**: differences in the expected stance of policy across economies can matter for FX and relative-rate trades.
- **Scenario design and stress testing**: shadow-rate paths can provide a structured way to explore “easy/tight” regimes in macro scenarios.

Caveats:
- Shadow rates are model-dependent and unobserved.
- Inference can be sensitive to prior settings and volatility modelling.
- Forecast evaluation is essential; avoid treating point forecasts as certainties.

Related pages:
- {doc}`theory/shadow-rate-var`
- {doc}`theory/mcmc`

## How is the shadow rate different from the observed policy rate?

In an ELB model, the observed policy rate is treated as a *censored* version of a latent (unobserved) shadow rate.

Intuitively:
- Away from the bound, observed and shadow rates should be similar.
- At the bound, the observed rate can become “stuck”, while the shadow rate can move below the bound and still transmit policy stance.

In `srvar-toolkit`, forecasts may include both:
- `ForecastResult.draws`: observed predictive draws (ELB applied)
- `ForecastResult.latent_draws`: latent shadow predictive draws (unconstrained)

Related pages:
- {doc}`theory/elb`
- {doc}`theory/shadow-rate-var`

## Why do my interest-rate forecasts look stuck at the ELB?

This can be expected if the observed rate is frequently at the bound and you’re looking at observed predictive draws.

Things to check:
- You are inspecting `ForecastResult.latent_draws` if you want the unconstrained shadow-rate forecast.
- `model.elb.applies_to` matches your rate variable name.
- `model.elb.bound` reflects the effective bound in your data.

## What does “diagonal stochastic volatility” mean?

Diagonal stochastic volatility means each series has its own time-varying variance, but the model does not include time-varying covariances (no stochastic correlation / no full-covariance SV).

Practical implications:
- This captures changing uncertainty series-by-series.
- It is not intended to model evolving cross-series correlation.

Related pages:
- {doc}`theory/stochastic-volatility`
- {doc}`user-guide/limitations`

## How many MCMC draws do I need?

There is no universal answer; it depends on model complexity (ELB/SV are more expensive), data size, and how stable your posterior summaries need to be.

Rules of thumb:
- Smoke test: `draws=200`, `burn_in=50`, `thin=1–2`
- Basic analysis starting point: `draws=2000`, `burn_in=500`, `thin=2`

The toolkit does not currently ship full convergence diagnostics (e.g. ESS/R-hat), so you should validate mixing and convergence yourself.

Related pages:
- {doc}`theory/mcmc`
- {doc}`user-guide/limitations`

## Why do results change when I rerun the same model?

Common causes:
- You didn’t fix a seed (`sampler.seed` in YAML or `rng=` in Python).
- You changed `draws`, `burn_in`, or `thin` (Monte Carlo error changes).
- ELB/SV models can be more sensitive to priors and initial conditions.

For reproducibility:
- Use a fixed seed.
- Keep the saved `config.yml` from the output directory with your results.

## What files does `srvar run config.yml` write?

The run writes outputs to `output.out_dir` (or the `--out` override). Typical artifacts include:

- `config.yml`: exact config used
- `fit_result.npz`: posterior draws and metadata
- `forecast_result.npz`: predictive simulation outputs (if forecasting enabled)
- `shadow_rate_*.png`, `volatility_*.png`, `forecast_fan_*.png`: plots (if plot saving enabled)

## How do I make the CLI quieter / more verbose?

- `--quiet`: suppress console output
- `--verbose`: print more detailed progress output
- `--no-color`: disable ANSI colors (useful for CI logs)

## When should I use `niw_default` vs Minnesota shrinkage?

- `PriorSpec.niw_default(...)` is a simple conjugate baseline and is useful for smoke tests and sanity checks.
- `PriorSpec.niw_minnesota(...)` introduces structured shrinkage that is often a better default for macro forecasting.

When in doubt, start with Minnesota shrinkage and then do sensitivity checks.

## Why does increasing lag order `p` slow things down so much?

The number of regressors grows with `K = (intercept) + N * p`. Increasing `p` increases the size of coefficient objects and the amount of linear algebra per iteration.

This is especially noticeable when ELB and/or SV are enabled.

Related pages:
- {doc}`user-guide/limitations`
