# Graph quickstart

New in **pymargins 0.4.0**.

This tutorial walks through the three canonical workflows from the design
note (§8.1–8.3).

## 1. Anchor — complete data, hand-fit, delta

```python
import statsmodels.formula.api as smf
from pymargins import GComputation

model = smf.logit("y ~ treat*x1 + x2 + age", data=df).fit()
est = GComputation(model, at="overall", scale="response", method="delta")

est.evaluate(steps.at_levels("treat", [0, 1]))   # standardized rates
est.contrasts("treat")                           # marginal risk difference
est.dydx("age")                                  # AME
```

No graph, no fan: compiles to the v0.3.x path, byte-identical. κ is checked
once at compile and recorded in the plan.

## 2. Survival flavor — declared simulation

```python
from lifelines import CoxPHFitter
from pymargins import GComputation

cph = CoxPHFitter().fit(df_surv, "time", "event")
est = GComputation(cph, at="overall", scale="response",
                   method="simulation", n_sim=8000)
est.rmst("treat", tau=365)
```

`method="auto"` would have resolved to simulation at compile (κ pre-pass) and
recorded why; it never flips later.

## 3. Matching → ATT, hand-fit preserved

```python
from pymargins import steps, GComputation

prep = steps.input(df)
prep = steps.match(prep, PysmatchClient(model="knn", k=1))

matched = prep.collect()
model = smf.logit("y ~ treat + x1 + x2", data=matched).fit()

est = GComputation(prep, outcome=model, at="treated",
                   scale="response", method="bootstrap", B=1999)
est.contrasts("treat")
```

Matching has no `influence()` → analytic vcov structurally unavailable;
bootstrap re-matches per replicate mandatorily.

## Plan inspection

Every estimator carries an immutable plan:

```python
est.plan.hash       # 'a7f3c21@1'
est.plan.describe() # human-readable summary
```

The plan hash is printed in every result summary footer, making
pre-registration literal.
