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

# Cox proportional hazards
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

`lifelines.CoxPHFitter` is supported through a dedicated adapter that
exposes hazard ratios on the log-scale (`Margins.log_scale`) and
survival probabilities at user-specified times.

```{code-cell} python
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "age": rng.normal(60, 10, n),
    "treated": rng.binomial(1, 0.5, n),
    "biomarker": rng.normal(0, 1, n),
})
lp = 0.05 * (df["age"] - 60) - 0.6 * df["treated"] + 0.3 * df["biomarker"]
df["duration"] = rng.exponential(np.exp(-lp) * 10)
df["event"] = (df["duration"] < 12).astype(int)
df["duration"] = df["duration"].clip(upper=12)

cph = CoxPHFitter().fit(df, duration_col="duration", event_col="event")
```

## Hazard ratio for `treated`

Because the model was fit directly on a DataFrame (not via a formula),
we pass the training data explicitly to the adapter:

```{code-cell} python
from pymargins.adapters import LifelinesCoxPHAdapter

_adapter = LifelinesCoxPHAdapter(cph, training_data=df)
m = Margins.log_scale(cph, adapter=_adapter, at="overall")
print(m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
).summary())
```

## Marginal HR per unit of `biomarker`

```{code-cell} python
print(Margins.log_scale(cph, adapter=_adapter, at="overall").dydx("biomarker").summary())
```

Because the session is on the **log scale**, `dydx` returns the
change in `log(HR)` per unit of biomarker.  For a Cox model this is
numerically close to the coefficient itself (the difference arises
from the covariate centering lifelines applies internally).  If you
want the change in the raw HR, use `Margins.linear_scale(...)` and
interpret the AME as the absolute change in the partial hazard ratio.

## Restricted mean survival time (RMST)

For survival adapters that support per-scenario `prediction_time`,
`rmst()` integrates the survival function up to a horizon by
trapezoidal rule:

```{code-cell} python
from pymargins.adapters import LifelinesCoxPHSurvivalAdapter

m_surv = Margins(
    cph,
    adapter=LifelinesCoxPHSurvivalAdapter(cph, training_data=df, prediction_time=365),
    at="overall",
    method="bootstrap",
    n_boot=100,
)

# RMST at 3 years (1095 days) under treated and control
rmst_treat   = m_surv.rmst(horizon=1095, atexog={"treated": 1}, n_grid=40)
rmst_control = m_surv.rmst(horizon=1095, atexog={"treated": 0}, n_grid=40)

print(rmst_treat.summary())
print(rmst_control.summary())

# Difference with joint inference
rmst_diff = rmst_treat - rmst_control
print(rmst_diff.summary())
```

The default grid is `n_grid=80`. Increase it for a more accurate
integral (especially when the survival curve changes rapidly) at the
cost of more prediction evaluations.

## Plot: forest plot of hazard ratios

```{code-cell} python
import matplotlib.pyplot as plt
from pymargins import reference

scen, W = reference("treated", [0, 1], ref_level=0)
res = m.contrasts(scenarios=scen, contrasts=W)
df_forest = res.to_frame()

fig, ax = plt.subplots(figsize=(4, 2))
y = range(len(df_forest))
ax.errorbar(
    df_forest["estimate"], y,
    xerr=[df_forest["estimate"] - df_forest["ci_lower"],
          df_forest["ci_upper"] - df_forest["estimate"]],
    fmt="o", capsize=3, color="firebrick"
)
ax.axvline(1, color="grey", lw=0.5)
ax.set_yticks(list(y))
ax.set_yticklabels(df_forest["label"])
ax.set_xlabel("Hazard ratio")
ax.invert_yaxis()
```

See [](aft_survival.md) for parametric AFT models, where the natural
inference scale is the time scale rather than the hazard scale.