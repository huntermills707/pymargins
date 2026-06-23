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
`pymargins` exposes three inference paths behind one keyword,
`method=`. Picking the right one is a function of curvature (κ) and
the resampling structure of your data.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, steps  # 0.4.0: Margins -> GComputation

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
m = GComputation(fit, at="overall", method="delta", scale="log")
print(m.predict(atexog={"x": [-2, 0, 2]}).summary())
```

## Krinsky–Robb simulation

Useful when a probability sits near 0 or 1 and the symmetric Wald CI
would cross the boundary.

```{code-cell} python
m_sim = GComputation(fit, at="overall", method="simulation", n_sim=2000, scale="log")
print(m_sim.predict(atexog={"x": [-2, 0, 2]}).summary())
```

## Pairs bootstrap

```{code-cell} python
m_boot = GComputation(fit, at="overall", method="bootstrap", B=200, scale="log")
print(m_boot.predict(atexog={"x": [-2, 0, 2]}).summary())
```

## Cluster bootstrap

Pass cluster IDs through `steps.input` to switch from pairs to
cluster resampling — required when within-cluster correlation
matters.

```{code-cell} python
m_clust = GComputation(
    steps.input(df, cluster=df["g"].values),
    outcome=fit,
    at="overall",
    method="bootstrap",
    B=500,
    scale="log",
)
print(m_clust.predict(atexog={"x": [-2, 0, 2]}).summary())
```

## Plot: comparing CI widths across methods

```{code-cell} python
import matplotlib.pyplot as plt

x_grid = [-2, 0, 2]
res_delta = m.predict(atexog={"x": x_grid}).to_frame()
res_sim = m_sim.predict(atexog={"x": x_grid}).to_frame()
res_boot = m_boot.predict(atexog={"x": x_grid}).to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
widths = {
    "delta": res_delta["ci_upper"] - res_delta["ci_lower"],
    "simulation": res_sim["ci_upper"] - res_sim["ci_lower"],
    "bootstrap": res_boot["ci_upper"] - res_boot["ci_lower"],
}
x_pos = np.arange(len(x_grid))
width = 0.25
for i, (label, w) in enumerate(widths.items()):
    ax.bar(x_pos + i * width, w, width, label=label)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([str(v) for v in x_grid])
ax.set(xlabel="x", ylabel="CI width")
ax.legend(title="Method")
```

See [](../explanations/delta_sim_bootstrap.md) for the decision rule.