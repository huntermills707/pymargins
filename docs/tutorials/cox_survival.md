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

```{code-cell} python
m = Margins.log_scale(cph, at="overall")
m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
).summary()
```

## Marginal HR per unit of `biomarker`

```{code-cell} python
m.dydx("biomarker").summary()
```

See [](aft_survival.md) for parametric AFT models, where the natural
inference scale is the time scale rather than the hazard scale.
