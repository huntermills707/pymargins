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

# Generalised Estimating Equations (GEE)

`statsmodels.GEE` handles correlated outcomes through a working
correlation structure and sandwich covariance.  `pymargins` consumes the
GEE fit directly — the robust (sandwich) covariance is the default, and
the session API is identical to GLM.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(42)
n = 600
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
    "site": rng.choice(["A", "B", "C", "D"], n),
    "subject": np.repeat(np.arange(60), 10),
})
lp = -1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.gee(
    "y ~ age + female + treated + C(site)",
    groups="subject",
    data=df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Independence(),
).fit()
```

## Average marginal effect of `age`

The default covariance is the GEE sandwich (robust) estimator:

```{code-cell} python
m = Margins.linear_scale(fit, at="overall")
print(m.dydx("age").summary())
```

## Predicted probabilities by treatment arm

```{code-cell} python
print(m.predict(atexog={"treated": [0, 1]}).summary())
```

## Risk difference for treatment

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("treated", [1, 0])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## Plot: predicted probability curve over age

```{code-cell} python
import matplotlib.pyplot as plt

ages = list(range(20, 76, 2))
res = m.predict(atexog={"age": ages, "treated": [0, 1]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for level, sub in df_plot.groupby("treated"):
    label = "Treated" if level == 1 else "Control"
    ax.plot(sub["age"], sub["estimate"], label=label)
    ax.fill_between(
        sub["age"], sub["ci_lower"], sub["ci_upper"], alpha=0.15
    )
ax.set(xlabel="Age", ylabel="P(y=1)")
ax.legend(title="Treatment")
```

## Plot: forest plot of site contrasts

```{code-cell} python
from pymargins import reference

scen, W = reference("site", ["A", "B", "C", "D"], ref_level="A")
res = m.contrasts(scenarios=scen, contrasts=W)
df_forest = res.to_frame()

fig, ax = plt.subplots(figsize=(4, 3))
y = range(len(df_forest))
ax.errorbar(
    df_forest["estimate"], y,
    xerr=[df_forest["estimate"] - df_forest["ci_lower"],
          df_forest["ci_upper"] - df_forest["estimate"]],
    fmt="o", capsize=3,
)
ax.axvline(0, color="grey", lw=0.5)
ax.set_yticks(list(y))
ax.set_yticklabels(df_forest["label"])
ax.set_xlabel("Risk difference vs Site A")
ax.invert_yaxis()
```

## Alternative vcov flavours

GEE fits carry three built-in covariances.  You can switch at session
construction:

```{code-cell} python
# Naive (model-based) covariance
m_naive = Margins.linear_scale(fit, vcov="naive", at="overall")
print(m_naive.dydx("age").summary())

# Bias-corrected robust sandwich (if available)
try:
    m_bc = Margins.linear_scale(fit, vcov="robust_bc", at="overall")
    print(m_bc.dydx("age").summary())
except ValueError as exc:
    print("robust_bc not available:", exc)
```

## Formula vs array interface

If you fit with `sm.GEE(y, X, groups=...)` instead of `smf.gee(...)`,
provide the training data explicitly so the adapter can reconstruct the
design matrix:

```python
from pymargins._adapters.statsmodels_gee import StatsmodelsGEEAdapter

adapter = StatsmodelsGEEAdapter(fit_array, training_data=df)
m = Margins.linear_scale(fit_array, adapter=adapter, at="overall")
```
