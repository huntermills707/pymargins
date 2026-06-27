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

# Cluster and block bootstrap

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation, steps  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "x": rng.normal(0, 1, n),
    "firm": rng.integers(1, 50, n),
})
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.5 * df["x"]))))

fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
```


For panel / multilevel data, pass `cluster=` through `steps.input`:

```{code-cell} python
m = GComputation(
    steps.input(df, cluster=df["firm"].values),
    outcome=fit,
    method="bootstrap",
    B=2000,
    scale="log",
)
```

Whole clusters are resampled with replacement; within-cluster
dependence is preserved.

For time-series data, pass `block=` through `steps.input` to use
moving-block resampling:

```{code-cell} python
m = GComputation(
    steps.input(df, block=8),
    outcome=fit,
    method="bootstrap",
    B=2000,
    scale="identity",
)
```

`cluster=` and `block=` are mutually exclusive. The block length
should span the dependence horizon: too short under-covers, too long
collapses to fewer effective draws.