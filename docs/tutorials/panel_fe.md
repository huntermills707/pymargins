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

# Panel data — fixed effects
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

`linearmodels.PanelOLS` (entity, time, or two-way fixed effects),
`RandomEffects`, `BetweenOLS`, `AbsorbingLS`, and `FamaMacBeth` are
each handled by a dedicated adapter. The session API is unchanged.

```{code-cell} python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(13)
n_entities, n_periods = 200, 6
entity = np.repeat(np.arange(n_entities), n_periods)
period = np.tile(np.arange(n_periods), n_entities)
df = pd.DataFrame({
    "entity": entity, "period": period,
    "x": rng.normal(0, 1, n_entities * n_periods),
    "alpha_i": np.repeat(rng.normal(0, 1, n_entities), n_periods),
}).set_index(["entity", "period"])
df["y"] = 1.0 + 0.6 * df["x"] + df["alpha_i"] + rng.normal(0, 0.5, len(df))

fit = PanelOLS(df["y"], df[["x"]], entity_effects=True).fit(
    cov_type="clustered", cluster_entity=True
)
```

## AME of `x` with cluster-robust SEs

The fit's clustered covariance is consumed automatically — the
session does not need a separate `vcov=` argument when the model
already carries one.

```{code-cell} python
m = Margins.linear_scale(fit, at="overall")
print(m.dydx("x").summary())
```

## Predicted `y` at representative `x`

Entity and time fixed effects are absorbed during estimation; they do
not appear as explicit regressors in the margin calculations.
Predictions are therefore evaluated at the **population-averaged**
profile (or the representative profile you specify with `at=`).

```{code-cell} python
print(m.predict(atexog={"x": [-1.0, 0.0, 1.0]}).summary())
```

## Plot: predicted outcome over representative `x`

```{code-cell} python
import matplotlib.pyplot as plt

x_vals = np.linspace(-2, 2, 50)
res = m.predict(atexog={"x": x_vals})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_plot["x"], df_plot["estimate"], color="navy")
ax.fill_between(
    df_plot["x"], df_plot["ci_lower"], df_plot["ci_upper"],
    alpha=0.2, color="navy"
)
ax.set(xlabel="x", ylabel="Predicted y")
```

See [](../howto/cluster_block_bootstrap.md) for cluster-resampling
bootstrap in panels where the analytic clustered SE is suspect.