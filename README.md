# srvar-toolkit
 
Shadow-rate VAR toolkit (Bayesian VAR / SRVAR) in pure Python.

This package implements a small, testable SRVAR workflow:
- Fit conjugate BVARs (NIW) and run forecasts.
- Shadow-rate / ELB data augmentation.
- Diagonal stochastic volatility (SVRW).
- Combined SV + ELB model.

If you need the original MATLAB replication code, see:
https://github.com/MichaelGrammmatikopoulos/MLSRVARs
 
## Status
 
Implemented:
- Phase 2: Conjugate NIW Bayesian VAR (BVAR)
- Phase 3: ELB / shadow-rate data augmentation
- Phase 4: Diagonal stochastic volatility (SVRW) via KSC mixture + precision-based state sampling
- Combined: SV + ELB

## Install

```bash
pip install srvar-toolkit
```

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
