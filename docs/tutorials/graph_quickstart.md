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

# Graph quickstart

New in **pymargins 0.4.0**.

This tutorial walks through the three canonical workflows from the design
note (§8.1–8.3).

## 1. Anchor — complete data, hand-fit, delta

This is the v0.3.x workflow, unchanged in spirit:

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pymargins import GComputation

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "age": rng.integers(18, 80, size=n),
    "treat": rng.binomial(1, 0.4, size=n),
    "x1": rng.normal(size=n),
})
logit_p = 1 / (1 + np.exp(-(-1.0 + 0.04 * df["age"] + 0.5 * df["treat"] + 0.3 * df["x1"])))
df["y"] = rng.binomial(1, logit_p)

model = smf.logit("y ~ treat*x1 + age", data=df).fit(disp=0)
est = GComputation(model, at="overall", scale="response", method="delta")

est.predict()                                    # average adjusted prediction
est.predict(atexog={"treat": [0, 1]})            # standardized rates
est.contrasts(
    scenarios=[{"atexog": {"treat": 0}}, {"atexog": {"treat": 1}}],
    contrasts=[-1, 1],
)                                                  # marginal risk difference
est.dydx("age")                                  # AME
```

No graph, no fan: this compiles to the v0.3.x path. Queries share one fit →
joint covariance is available.

## 2. Survival flavor — declared simulation

```{code-cell} python
# Requires: pip install lifelines
from lifelines import CoxPHFitter

rng = np.random.default_rng(7)
n = 800
df_surv = pd.DataFrame({
    "time": rng.exponential(50, size=n),
    "event": rng.binomial(1, 0.8, size=n),
    "treat": rng.binomial(1, 0.4, size=n),
    "age": rng.normal(50, 10, size=n),
})

cph = CoxPHFitter().fit(df_surv, "time", "event")

from pymargins.adapters import LifelinesCoxPHAdapter

est = GComputation(
    cph,
    adapter=LifelinesCoxPHAdapter(cph, training_data=df_surv),
    at="overall",
    scale="response",
    method="simulation",
    n_sim=2000,
    seed=42,
)

est.dydx("age")
```

Declare `method="simulation"` explicitly when you want a non-delta path;
the method is resolved once at compile and never flips later.

## 3. Matching → ATT, hand-fit preserved

```{code-cell} python
import logging

from pymargins import steps, PysmatchClient
from pysmatch.Matcher import Matcher

# pysmatch logs ~15 INFO/WARNING lines (to the root logger) on every match,
# and the bootstrap below re-matches on each of the B replicates. Raise the
# root log level so that per-replicate spam stays out of the rendered output
# and the cached notebook.
logging.getLogger().setLevel(logging.ERROR)

test = df[df["treat"] == 1].copy()
control = df[df["treat"] == 0].copy()
matcher = Matcher(test, control, yvar="treat", exclude=["y"])
matcher.fit_scores(balance=True, model_type="linear")
matcher.predict_scores()
matcher.match(method="min", nmatches=1, threshold=0.001)

matched = matcher.matched_data
model = smf.logit("y ~ treat + x1 + age", data=matched).fit(disp=0)

est = GComputation(
    steps.match(steps.input(matched), PysmatchClient(matcher, treatment_col="treat")),
    outcome=model,
    at="overall",
    scale="response",
    method="bootstrap",
    B=999,
    seed=123,
)
est.contrasts(
    scenarios=[{"atexog": {"treat": 0}}, {"atexog": {"treat": 1}}],
    contrasts=[-1, 1],
)
```

Matching has no `influence()` → analytic vcov is structurally unavailable;
bootstrap re-matches per replicate mandatorily.

## Plan inspection

Every estimator carries an immutable plan:

```{code-cell} python
print(est.plan.hash)
print(est.plan.describe())
```

The plan hash is printed in every result summary footer, making
pre-registration literal.
