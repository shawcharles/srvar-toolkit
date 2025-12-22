# Quickstart

## Fit a BVAR and forecast

```python
import numpy as np

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig

rng = np.random.default_rng(123)

y = rng.standard_normal((80, 2))
ds = Dataset.from_arrays(values=y, variables=["y1", "y2"])

model = ModelSpec(p=2, include_intercept=True)
prior = PriorSpec.niw_default(k=1 + ds.N * model.p, n=ds.N)
sampler = SamplerConfig(draws=500, burn_in=100, thin=1)

fit_res = fit(ds, model, prior, sampler, rng=rng)
fc = forecast(fit_res, horizons=[1, 4], draws=200, rng=rng)

print(fc.mean)
```

For more runnable scripts, see `examples/`.
