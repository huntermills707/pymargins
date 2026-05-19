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

# Linear Mixed Effects

`statsmodels.MixedLM` fits random-intercept and random-slope models.
`pymargins` uses the fixed-effects coefficients and their covariance
for population-averaged marginal effects and predictions.  Random
effects are integrated out — the reported quantities are unconditional
on any particular cluster.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(42)
n = 400
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
    "school": rng.choice(["A", "B", "C", "D"], n),
    "student": np.repeat(np.arange(40), 10),
})
# Random intercept per student
student_effect = rng.standard_normal(40)[df["student"].values]
df["score"] = (
    50 + 0.3 * df["age"] - 2.0 * df["female"] + 3.0 * df["treated"]
    + student_effect + rng.normal(0, 5, n)
)

fit = smf.mixedlm(
    "score ~ age + female + treated + C(school)",
    groups="student",
    data=df,
).fit()
```

## AME of `age`

```{code-cell} python
m = Margins.linear_scale(fit, at="overall")
print(m.dydx("age").summary())
```

## Predicted score by treatment arm

```{code-cell} python
print(m.predict(atexog={"treated": [0, 1]}).summary())
```

## Treatment contrast

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("treated", [1, 0])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## Plot: predicted score over age by sex

```{code-cell} python
import matplotlib.pyplot as plt

ages = list(range(20, 76, 2))
res = m.predict(atexog={"age": ages, "female": [0, 1]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for level, sub in df_plot.groupby("female"):
    label = "Female" if level == 1 else "Male"
    ax.plot(sub["age"], sub["estimate"], label=label)
    ax.fill_between(
        sub["age"], sub["ci_lower"], sub["ci_upper"], alpha=0.15
    )
ax.set(xlabel="Age", ylabel="Predicted score")
ax.legend(title="Sex")
```

## Plot: forest plot of school contrasts

```{code-cell} python
from pymargins import reference

scen, W = reference("school", ["A", "B", "C", "D"], ref_level="A")
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
ax.set_xlabel("Score difference vs School A")
ax.invert_yaxis()
```

## Array interface

If you fit with `sm.MixedLM(y, X, groups=...)` instead of
`smf.mixedlm(...)`, pass the training data explicitly:

```python
from pymargins.adapters import StatsmodelsMixedLMAdapter

adapter = StatsmodelsMixedLMAdapter(fit_array, training_data=df)
m = Margins.linear_scale(fit_array, adapter=adapter, at="overall")
```

## Notes on bootstrap inference

Mixed-model covariance is already cluster-aware through the random
effects structure.  If you still want a bootstrap, open the session
with `method="bootstrap"` and a modest `n_boot=` — refitting MixedLM
is slower than OLS or GLM, so keep the replicate count low for
exploration and raise it only for final reporting.
