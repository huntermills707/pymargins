---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Accelerated failure time models

Parametric AFTs (Weibull, log-logistic, log-normal,
generalized-gamma, piecewise exponential) report on the *time* scale
rather than the *hazard* scale. The session helper of choice is
`Margins.log_scale` for time ratios, or `Margins.linear_scale` for
expected survival time itself.

```{code-cell} python
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from pymargins import Margins

rng = np.random.default_rng(5)
n = 1500
df = pd.DataFrame({
    "age": rng.normal(60, 10, n),
    "treated": rng.binomial(1, 0.5, n),
})
lp = 2.0 + 0.5 * df["treated"] - 0.01 * (df["age"] - 60)
T = rng.weibull(1.5, n) * np.exp(lp)
df["duration"] = np.minimum(T, 15)
df["event"] = (T < 15).astype(int)

aft = WeibullAFTFitter().fit(df, duration_col="duration", event_col="event")
```

## Time ratio for `treated`

```{code-cell} python
m = Margins.log_scale(aft, at="overall")
m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
).summary()
```

## Marginal effect of age on expected duration

```{code-cell} python
Margins.linear_scale(aft, at="overall").dydx("age").summary()
```

Other AFT families work the same way; swap the fitter:

- `LogLogisticAFTFitter`
- `LogNormalAFTFitter`
- `GeneralizedGammaFitter`
- `PiecewiseExponentialRegressionFitter`
