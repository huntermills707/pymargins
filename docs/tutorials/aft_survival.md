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
rather than the *hazard* scale. Use `scale="log"` for time ratios, or
`scale="identity"` for expected survival time itself.

```{code-cell} python
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

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

Because the model was fit directly on a DataFrame (not via a formula),
we pass the training data explicitly to the adapter:

```{code-cell} python
from pymargins.adapters import LifelinesWeibullAFTAdapter

_adapter = LifelinesWeibullAFTAdapter(aft, training_data=df)
m = GComputation(aft, adapter=_adapter, at="overall", scale="log")
print(m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
).summary())
```

## Predicted median survival time by treatment

On the linear scale, predictions are expected survival times:

```{code-cell} python
print(GComputation(aft, adapter=_adapter, at="overall", scale="identity").predict(
    atexog={"treated": [0, 1]}
).summary())
```

## Marginal effect of age on expected duration

```{code-cell} python
print(GComputation(aft, adapter=_adapter, at="overall", scale="identity").dydx("age").summary())
```

## Plot: predicted median survival time by treatment

```{code-cell} python
import matplotlib.pyplot as plt

res = GComputation(aft, adapter=_adapter, at="overall", scale="identity").predict(
    atexog={"treated": [0, 1]}
)
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(["Control", "Treated"], df_plot["estimate"],
       yerr=[df_plot["estimate"] - df_plot["ci_lower"],
             df_plot["ci_upper"] - df_plot["estimate"]],
       capsize=4, color="teal", edgecolor="black")
ax.set(ylabel="Predicted median survival time")
```

Other AFT families work the same way; swap the fitter:

- `LogLogisticAFTFitter`
- `LogNormalAFTFitter`
- `GeneralizedGammaFitter`
- `PiecewiseExponentialRegressionFitter`