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

# Union premium heterogeneity — effects by subgroup
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

An average marginal effect answers "what is the effect on average?".
Often the research question is sharper: *for whom* is the effect large,
and *for whom* is it small? The union wage premium is the textbook
case — unions are widely held to *compress* wages, raising pay most at
the bottom of the skill distribution and least at the top. That is a
statement about heterogeneity, and it is invisible in a single AME.

This demo uses the `linearmodels` wage panel (545 young men, 1980–1987)
to estimate the union premium **by education band**, using the `over=`
subgroup argument and a contrast that lets the premium differ across
groups. It also makes the central modelling point explicit: *a subgroup
effect can only differ if the model lets it* — through an interaction,
or through a nonlinear link.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.datasets import wage_panel

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

df = wage_panel.load().reset_index()
df["educ_band"] = pd.cut(
    df["educ"], [0, 11, 12, 20], labels=["<HS", "HS", "college"]
).astype(str)
print(df[["nr", "year", "lwage", "union", "married", "exper", "educ",
          "educ_band"]].head())
print("\nObservations per education band:")
print(df["educ_band"].value_counts())
```

## 1. A model that *allows* heterogeneity

If we regress log wage on `union` plus controls **without an
interaction**, the union coefficient is a single number and the
implied premium is identical in every subgroup — `over=` would just
return the same value three times. To let the premium vary by
education we interact `union` with `educ_band` (and, so that the
return to experience can vary too, `exper` with `educ_band`):

```{code-cell} python
fit = smf.ols(
    "lwage ~ union * educ_band + exper * educ_band + expersq + married + C(year)",
    data=df,
).fit()
m = Margins.linear_scale(
    fit,
    vcov={"type": "cluster", "groups": df["nr"]},  # cluster by worker
    at="overall",
)
```

We cluster the standard errors by worker (`nr`) because the same man is
observed in up to eight years — the panel analogue of the survey-design
adjustment in the [California API demo](california_api_survey.md).

## 2. Predicted wages by union status, within each band

`over="educ_band"` averages the prediction within each education band
rather than over the whole sample. Toggling `union` across `[0, 1]`
gives the union and non-union wage in each band:

```{code-cell} python
profiles = m.predict(atexog={"union": [0, 1]}, over="educ_band")
print(profiles.summary())
```

Each row is a band × union cell, with the other covariates averaged
*within* that band — so the comparison holds composition fixed at each
education level rather than at the pooled mean.

## 3. The premium, allowed to differ across bands

The premium within a band is the union-minus-non-union contrast, taken
*inside* that band by pinning `educ_band` in the scenario. A `dict` of
named contrasts returns all three premiums in one result with a joint
covariance — which is what lets us then test whether they differ.

```{code-cell} python
bands = ["<HS", "HS", "college"]
scenarios = []
for b in bands:
    scenarios.append({"atexog": {"union": 1, "educ_band": b}, "label": f"{b}:union"})
    scenarios.append({"atexog": {"union": 0, "educ_band": b}, "label": f"{b}:nonunion"})

# premium[b] = (union cell) − (non-union cell) for band b
weights = {}
for i, b in enumerate(bands):
    w = [0] * len(scenarios)
    w[2 * i] = 1
    w[2 * i + 1] = -1
    weights[f"premium[{b}]"] = w

# add a difference-of-premiums so we can read its CI directly
w = [0] * len(scenarios)
w[0], w[1] = -1, 1          # subtract <HS premium
w[4], w[5] = 1, -1          # add college premium
weights["college − <HS"] = w

premiums = m.contrasts(scenarios=scenarios, contrasts=weights)
print(premiums.summary())
```

The premium is largest for high-school and below-high-school workers
and roughly half the size for college graduates — the wage-compression
story, made quantitative. The final row, `college − <HS`, is the
heterogeneity itself: a single contrast with its own interval, so you
can report *whether* the gap in premiums is distinguishable from noise
rather than eyeballing three overlapping confidence intervals.

## 4. Continuous-variable heterogeneity with `over=`

For a *continuous* regressor the subgroup slope comes straight from
`dydx(..., over=...)` — no scenario bookkeeping. Here is the return to
labour-market experience by education band, which the `exper × educ_band`
interaction lets vary:

```{code-cell} python
ret = m.dydx("exper", over="educ_band")
print(ret.summary())
```

`over=` accepts a list to cross several grouping variables at once
(`over=["union", "educ_band"]`), giving one slope per cell — the same
machinery scales from one subgroup dimension to several.

## The modelling caveat, stated plainly

Subgroup effects are only as rich as the model that produces them:

- **Linear model, no interaction** → every subgroup AME is identical.
  `over=` is descriptive (different *baselines*) but the *effect* is
  constant by construction.
- **Linear model, with interaction** (this demo) → effects differ by
  exactly what the interaction encodes.
- **Nonlinear link** (logit, Poisson, Cox) → effects differ across
  subgroups *even with no interaction*, because the marginal effect
  depends on where each subgroup sits on the response curve. The
  [subgroup how-to](../howto/subgroups.md) demonstrates this case.

`over=` reports the heterogeneity your specification implies; it does
not manufacture heterogeneity the model has no room for. Deciding which
interactions belong in the model is a modelling choice that precedes
the margins call.

## Where to next

- [](../howto/subgroups.md) — the `over=` argument in full, including
  the nonlinear-link case and crossing multiple grouping variables.
- [](wage_panel.md) — the same data with an entity-fixed-effects
  specification for the pooled union premium.
- [](../howto/contrasts.md) — the contrast primitive used here for
  per-subgroup premiums and their differences.