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

# California API — stratified survey design
The California Academic Performance Index (API) measures school achievement.
The California Department of Education publishes the full population (`apipop`,
6 194 schools) and several survey samples drawn from it. Here we analyze
**`apistrat`**, a stratified sample of 200 schools where strata are school type
(`stype`: Elementary / Middle / High) and sampling weights (`pw`) correct for
unequal selection probabilities within each stratum.

This demo shows:

1. Loading the stratified sample and its design variables.
2. Fitting a **weighted** model for 2000 API scores (Stata `svy: reg` style).
3. Computing **survey-adjusted** average marginal effects and standard errors
   via Taylor linearization.
4. A design-based bootstrap cross-check.
5. A policy-relevant contrast: the expected API gap between low- and
   high-poverty schools.
6. Validating the survey estimate against the *known* population truth.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, SurveyDesign, steps  # 0.4.0: Margins -> GComputation

# Load the stratified sample generated from R's survey package
apistrat = pd.read_csv("data/apistrat.csv")
print(apistrat.shape)
print(apistrat[["stype", "api00", "api99", "meals", "ell", "avg.ed",
                "mobility", "pw", "fpc"]].head())
```

## 1. Explore the survey design

```{code-cell} python
print("Schools per stratum:")
print(apistrat["stype"].value_counts().sort_index())
print("\nWeight range: {:.1f} – {:.1f}".format(apistrat["pw"].min(),
                                                apistrat["pw"].max()))
print("FPC range:    {:.1f} – {:.1f}".format(apistrat["fpc"].min(),
                                                apistrat["fpc"].max()))
```

The weights differ by roughly a factor of three because elementary, middle, and
high schools were sampled at different rates within their strata.

## 2. Fit a weighted model

We model the 2000 API score as a function of:
- `meals` — percent of students eligible for subsidized meals (poverty proxy)
- `ell` — percent of English-language learners
- `avg.ed` — average parent education (in years)
- `mobility` — percent of students who move in/out during the year

Because the sample is stratified with unequal probabilities, we fit with
`sampling weights` so the coefficients generalize to the population:

```{code-cell} python
fit = smf.glm(
    "api00 ~ meals + ell + Q(\"avg.ed\") + mobility",
    data=apistrat,
    freq_weights=apistrat["pw"].values,
).fit()
print(fit.summary().tables[1])
```

Note that, conditional on parent education and poverty, the English-learner
share (`ell`) carries almost no independent signal — its coefficient is small
and statistically indistinguishable from zero. Poverty (`meals`) is the
dominant predictor, so that is the marginal effect we focus on below.

## 3. Declare the survey design and compute AMEs

`SurveyDesign` receives the weights, the stratum labels, and the
finite-population correction. Here the schools are drawn directly from strata
without an intermediate cluster stage, so we omit `psu`. The `fpc` column holds
each stratum's population size, which tightens the SE to reflect that we
sampled a non-trivial fraction of every stratum:

```{code-cell} python
# stype is 'E', 'H', 'M' — encode as integers for SurveyDesign
stratum_codes = apistrat["stype"].astype("category").cat.codes

survey = SurveyDesign(
    weights=apistrat["pw"].values,
    strata=stratum_codes.values,
    fpc=apistrat["fpc"].values,
)

m = GComputation(
    steps.input(apistrat, design=survey),
    outcome=fit,
    weights=apistrat["pw"].values,
    at="overall",
)
print(m.dydx("meals").summary())
```

The point estimate is a **population-weighted** AME and the standard error is
**survey-adjusted** — it accounts for the stratified sampling design via Taylor
linearization.

## 4. Design-based bootstrap cross-check

A stratified bootstrap resamples schools *within* each stratum. With enough
replicates it should give a standard error close to the linearization one:

```{code-cell} python
m_boot = GComputation(
    steps.input(apistrat, design=survey),
    outcome=fit,
    weights=apistrat["pw"].values,
    at="overall",
    method="bootstrap",
    B=500,
    seed=42,
)
print(m_boot.dydx("meals").summary())
```

## 5. Side-by-side comparison

```{code-cell} python
results = pd.DataFrame({
    "approach": ["survey linearization", "survey bootstrap"],
    "estimate": [
        float(m.dydx("meals").estimate),
        float(m_boot.dydx("meals").estimate),
    ],
    "std_error": [
        float(m.dydx("meals").std_error),
        float(m_boot.dydx("meals").std_error),
    ],
})
print(results.round(4))
```

Both approaches agree: each additional percentage point of students on
subsidized meals is associated with roughly **1.8 fewer API points**, with a
standard error around **0.4**.

## 6. Policy contrast: low- vs high-poverty schools

What is the expected API gap between a low-poverty school (20 % of students on
subsidized meals) and a high-poverty school (80 %), holding the other
covariates at their (population-weighted) observed values?

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("meals", [20, 80])
res = m.contrasts(scenarios=scen, contrasts=w)
print(res.summary())
```

The contrast is a **difference in expected API scores** — the low-poverty
prediction minus the high-poverty one — not a slope. It is averaged over the
weighted distribution of the *other* covariates, and its standard error is
propagated through both the regression and the survey design. A 60-point swing
in the poverty share maps to roughly a **110-point** API gap.

## 7. Validate against the population truth

Because the full population `apipop` is published, we can check whether the
survey estimate actually recovers the population marginal effect. We fit the
same model on all 6 194 schools (no weights needed — this *is* the population)
and compare:

```{code-cell} python
apipop = pd.read_csv("data/apipop.csv")

truth_fit = smf.glm(
    "api00 ~ meals + ell + Q(\"avg.ed\") + mobility",
    data=apipop.dropna(subset=["api00", "meals", "ell", "avg.ed", "mobility"]),
).fit()
m_truth = GComputation(truth_fit, at="overall")

ame_survey = m.dydx("meals")
truth_value = float(m_truth.dydx("meals").estimate)
lo, hi = ame_survey.conf_int()

print(f"population truth AME(meals): {truth_value:7.3f}")
print(f"survey estimate:            {float(ame_survey.estimate):7.3f}")
print(f"95% survey CI:              [{float(lo):.3f}, {float(hi):.3f}]")
print(f"truth inside CI:            {float(lo) <= truth_value <= float(hi)}")
```

The 200-school survey estimate lands close to the population value, and the
design-based confidence interval covers the truth — exactly what a correctly
specified survey analysis should deliver.

## Where to next

- [](../tutorials/survey_design.md) — the underlying tutorial, which uses this
  same California API data to walk through weights, strata, and clusters one at
  a time.
- [](../howto/robust_clustered_ses.md) — robust and clustered SEs for
  non-survey settings.