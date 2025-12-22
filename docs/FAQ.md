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
