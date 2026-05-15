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

`lifelines.CoxPHFitter` is supported through a dedicated adapter that
exposes hazard ratios on the log-scale (`Margins.log_scale`) and
survival probabilities at user-specified times.

```{code-cell} python
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from pymargins import Margins

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
from pymargins._adapters.lifelines_coxph import LifelinesCoxPHAdapter

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

See [](aft_survival.md) for parametric AFT models, where the natural
inference scale is the time scale rather than the hazard scale.
