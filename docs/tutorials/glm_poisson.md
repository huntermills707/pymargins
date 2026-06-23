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

# GLM — Poisson count
A Poisson regression for count data. The natural reporting scale is
the *rate ratio* (log link → exp); `pymargins` makes that explicit
with `scale="log"`.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(7)
n = 4000
df = pd.DataFrame({
    "exposure": rng.uniform(1, 5, n),
    "treated": rng.binomial(1, 0.4, n),
    "age": rng.integers(20, 75, n),
})
lp = -2.0 + 0.6 * df["treated"] + 0.02 * df["age"]
df["events"] = rng.poisson(df["exposure"] * np.exp(lp))

fit = smf.glm(
    "events ~ treated + age",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["exposure"]),
).fit()
```

## Rate ratio for treatment

The contrast `treated=1` vs `treated=0` on the log scale is a log-RR;
the estimator back-transforms to a rate ratio with an asymmetric CI.

```{code-cell} python
m = GComputation(fit, vcov="HC1", at="overall", scale="log")
print(m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
).summary())
```

## Predicted counts by age

```{code-cell} python
print(m.predict(atexog={"age": [25, 45, 65]}).summary())
```

## Plot: predicted event rate by age

```{code-cell} python
import matplotlib.pyplot as plt

res = m.predict(atexog={"age": [25, 35, 45, 55, 65]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_plot["age"], df_plot["estimate"], marker="o", color="darkgreen")
ax.fill_between(
    df_plot["age"], df_plot["ci_lower"], df_plot["ci_upper"],
    alpha=0.2, color="darkgreen"
)
ax.set(xlabel="Age", ylabel="Predicted event count")
```

## Average marginal effect of `age` on rate

```{code-cell} python
print(GComputation(fit, at="overall", scale="identity").dydx("age").summary())
```