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

# Effects by subgroup with `over=`
`over=` partitions the sample and computes the estimand *within* each
group, returning one row per group with a shared covariance. It works
on `predict` and `dydx` (the aggregating estimands); for discrete
contrasts by subgroup, pin the group in the scenario instead (last
section).

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(42)
n = 4000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "treated": rng.binomial(1, 0.5, n),
    "region": rng.choice(["N", "S", "E", "W"], n),
})
lp = -3 + 0.04 * df["age"] + 0.9 * df["treated"] + 0.05 * df["age"] * df["treated"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age * treated + C(region)", data=df,
              family=sm.families.Binomial()).fit()
m = GComputation(fit, at="overall", scale="identity")
```

## AME within each subgroup

Pass the grouping column name. Each row is the average marginal effect
computed over the rows in that group only:

```{code-cell} python
print(m.dydx("age", over="treated").summary())
```

The slope of `age` differs between the treated and untreated groups —
here because the model contains an `age × treated` interaction.

## Predicted values within each subgroup

`predict` takes `over=` the same way. Combine it with `atexog` to sweep
a variable *inside* each group:

```{code-cell} python
res = m.predict(atexog={"treated": [0, 1]}, over="region")
print(res.to_frame()[["region", "treated", "estimate", "std_error"]])
```

Other covariates are averaged within each region, so the rows are
region-specific adjusted predictions rather than a single pooled
profile.

## Crossing several grouping variables

`over=` accepts a list. The sample is partitioned by the full cross of
the listed columns, one row per non-empty cell:

```{code-cell} python
print(m.dydx("age", over=["treated", "region"]).summary())
```

## Nonlinear models give heterogeneity *for free*

In a linear model with no interaction, every subgroup AME is identical
— `over=` only changes the *baseline*, not the *effect*. In a nonlinear
model the marginal effect depends on where each subgroup sits on the
response curve, so subgroup AMEs differ **even without an interaction
term**:

```{code-cell} python
# Drop the interaction: age enters additively, treated only shifts the level.
fit_add = smf.glm("y ~ age + treated + C(region)", data=df,
                  family=sm.families.Binomial()).fit()
m_add = GComputation(fit_add, at="overall", scale="identity")
print(m_add.dydx("age", over="treated").summary())
```

The two slopes still differ: the treated group sits higher on the
logistic curve, where the same change in the linear predictor maps to a
different change in probability. This is genuine link-driven
heterogeneity, not a modelling artefact — but note it is a property of
the *probability* scale. On a `scale="log"` or `scale="logit"` estimator
the subgroup effects of an additive model would instead coincide.

## Discrete contrasts by subgroup

`contrasts` does not take `over=` — a contrast is a single linear
combination, not an aggregation. To get a contrast *within* each
subgroup, pin the group in the scenarios (the grouping variable must be
a model regressor) and request all groups' contrasts at once via a
named-contrast `dict`, which gives them a joint covariance:

```{code-cell} python
from itertools import product

regions = ["N", "S", "E", "W"]
scenarios = []
for r in regions:
    scenarios.append({"atexog": {"treated": 1, "region": r}, "label": f"{r}:treated"})
    scenarios.append({"atexog": {"treated": 0, "region": r}, "label": f"{r}:control"})

weights = {}
for i, r in enumerate(regions):
    w = [0] * len(scenarios)
    w[2 * i], w[2 * i + 1] = 1, -1
    weights[f"effect[{r}]"] = w

print(m.contrasts(scenarios=scenarios, contrasts=weights).summary())
```

Because the contrasts share a covariance, you can follow up with a
joint test that the subgroup effects are equal, or add a
difference-of-contrasts row to read a specific between-group gap
directly — see [](contrasts.md) and the
[union-premium demo](../demos/wage_heterogeneity.md).

## Where to next

- [](../demos/wage_heterogeneity.md) — a full worked subgroup analysis
  (union premium by education) on real panel data.
- [](contrasts.md) — the contrast primitive used for per-subgroup
  effects and their differences.
- [](grid_predictions.md) — sweeping a covariate grid, which composes
  with `over=` for subgroup-by-grid surfaces.