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

# Inference — delta, simulation, bootstrap

`pymargins` exposes three inference paths behind one session keyword,
`method=`. Picking the right one is a function of curvature (κ) and
the resampling structure of your data.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({
    "x": rng.normal(0, 1, n),
    "g": rng.integers(0, 50, n),
})
lp = -2.5 + 1.6 * df["x"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
```

## Delta — the default

```{code-cell} python
m = Margins.log_scale(fit, at="overall", method="delta")
m.predict(atexog={"x": [-2, 0, 2]}).summary()
```

## Krinsky–Robb simulation

Useful when a probability sits near 0 or 1 and the symmetric Wald CI
would cross the boundary.

```{code-cell} python
m_sim = Margins.log_scale(fit, at="overall", method="simulation", n_sim=4000)
m_sim.predict(atexog={"x": [-2, 0, 2]}).summary()
```

## Pairs bootstrap

```{code-cell} python
m_boot = Margins.log_scale(
    fit, at="overall", method="bootstrap", n_boot=500
)
m_boot.predict(atexog={"x": [-2, 0, 2]}).summary()
```

## Cluster bootstrap

Pass cluster IDs at session construction to switch from pairs to
cluster resampling — required when within-cluster correlation
matters.

```{code-cell} python
m_clust = Margins.log_scale(
    fit, at="overall", method="bootstrap",
    n_boot=500, cluster=df["g"].values,
)
m_clust.predict(atexog={"x": [-2, 0, 2]}).summary()
```

See [](../explanations/delta_sim_bootstrap.md) for the decision rule.
