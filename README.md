# srvar-toolkit
 
Shadow-rate VAR toolkit (Bayesian VAR / SRVAR) in pure Python.

This package implements a small, testable SRVAR workflow:
- Fit conjugate BVARs (NIW) and run forecasts.
- Shadow-rate / ELB data augmentation.
- Diagonal stochastic volatility (SVRW).
- Combined SV + ELB model.
 
## Methodology / capability matrix

| Component | What it does | How to enable | Status |
| --- | --- | --- | --- |
| Conjugate BVAR (NIW) | Closed-form posterior updates and fast sampling for VAR coefficients/covariance | `PriorSpec.niw_default(...)` or `PriorSpec.niw_minnesota(...)` | Supported |
| Minnesota-style shrinkage (NIW) | Minnesota-style shrinkage implemented through an NIW prior construction | `PriorSpec.niw_minnesota(...)` | Supported |
| Variable selection (SSVS) | Spike-and-slab inclusion indicators for regression rows (stochastic search variable selection) | `PriorSpec.ssvs(...)` (with `PriorSpec.family='ssvs'`) | Supported |
| Shadow-rate / ELB augmentation | Latent shadow-rate sampling when an observed rate is at/near an ELB (data augmentation Gibbs) | `ModelSpec(elb=ElbSpec(...))` | Supported |
| Stochastic volatility (SVRW, diagonal) | Diagonal log-volatility random-walk state model (KSC mixture + precision-based sampling) | `ModelSpec(volatility=VolatilitySpec(...))` | Supported |
| Combined ELB + SV | Joint latent shadow-rate augmentation with diagonal SVRW | `ModelSpec(elb=..., volatility=...)` | Supported |
| Forecasting | Posterior predictive simulation + mean/quantiles (with ELB flooring applied to constrained variables) | `srvar.api.forecast(...)` | Supported |
| Steady states | Steady-state VAR / explicit steady-state parameterization | N/A | Not implemented |
| Bayesian LASSO prior | LASSO-type shrinkage prior | N/A | Not implemented |
| Dirichlet-Laplace prior | Dirichlet-Laplace shrinkage prior | N/A | Not implemented |

## Install

Editable install:

```bash
pip install -e .
```
 
## Quickstart
 
Fit a simple BVAR and forecast:
```python
import numpy as np
from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig

ds = Dataset.from_arrays(values=np.random.standard_normal((80, 2)), variables=["y1", "y2"])
model = ModelSpec(p=2, include_intercept=True)
prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

fit_res = fit(ds, model, prior, sampler)
fc = forecast(fit_res, horizons=[1, 4], draws=200)
print(fc.mean)
```
 
## Examples
 
See `examples/` for runnable scripts, including:
- `examples/phase3_elb_fit_forecast.py`
- `examples/phase4_sv_elb_fit_forecast.py`
 
## Development
 
Run tests:
 
```bash
pytest
```

## Credits

If you need the original MATLAB replication code, see: https://github.com/MichaelGrammmatikopoulos/MLSRVARs

Grammatikopoulos, M. 2025. "Forecasting With Machine Learning Shadow-Rate VARs." Journal of Forecasting 1–17. https://doi.org/10.1002/for.70041.
