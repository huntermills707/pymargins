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

The California Academic Performance Index (API) measures school
achievement. The California Department of Education publishes the full
population (`apipop`, 6,194 schools) and several survey samples drawn
from it. Here we use **`apistrat`**, a stratified sample of 200 schools
where strata are school type (`stype`: Elementary / Middle / High) and
sampling weights (`pw`) correct for unequal selection probabilities.

This demo shows:

1. Loading the stratified sample and its design variables.
2. Fitting a **weighted** model for 2000 API scores (Stata `svy: reg`
   style).
3. Computing **survey-adjusted** average marginal effects and standard
   errors via Taylor linearization.
4. A design-based bootstrap cross-check.
5. A policy-relevant contrast: the expected API gap between schools with
   high and low shares of English-language learners.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, SurveyDesign

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

The weights differ by roughly a factor of two because elementary,
middle, and high schools were sampled at different rates within their
strata.

## 2. Fit a weighted model

We model the 2000 API score as a function of:
- `meals` — percent of students eligible for subsidized meals (poverty
  proxy)
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

## 3. Declare the survey design and compute AMEs

`SurveyDesign` receives the weights, the stratum labels, and (optionally)
the PSU identifiers. Here the schools are drawn directly from strata
without an intermediate cluster stage, so we omit `psu`:

```{code-cell} python
# stype is 'E', 'H', 'M' — encode as integers for SurveyDesign
stratum_codes = apistrat["stype"].astype("category").cat.codes

survey = SurveyDesign(
    weights=apistrat["pw"].values,
    strata=stratum_codes.values,
)

m = Margins(fit, survey_design=survey,
            weights=apistrat["pw"].values, at="overall")
print(m.dydx("meals").summary())
```

The point estimate is a **population-weighted** AME and the standard
error is **survey-adjusted** — it accounts for the stratified sampling
design via Taylor linearization.

## 4. Design-based bootstrap cross-check

A stratified bootstrap resamples schools *within* each stratum. With
enough replicates it should give a standard error close to the
linearization one:

```{code-cell} python
m_boot = Margins(
    fit,
    survey_design=survey,
    weights=apistrat["pw"].values,
    at="overall",
    method="bootstrap",
    n_boot=500,
    rng_seed=42,
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
subsidized meals is associated with roughly **1.8 fewer API points**,
with a standard error around **0.4–0.5**.

## 6. Policy contrast: high-ELL vs low-ELL schools

What is the expected API gap between a school with 5 % English-language
learners and one with 35 %, holding other covariates at their
(population-weighted) observed values?

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("ell", [5, 35])
res = m.contrasts(scenarios=scen, contrasts=w)
print(res.summary())
```

The contrast is a **difference in expected API scores**, not a slope. It
is averaged over the weighted distribution of the *other* covariates,
and its standard error is propagated through both the regression link
and the survey design.

## Where to next

- [](../tutorials/survey_design.md) — the underlying tutorial with
  simulated data, showing both weighted and unweighted fits.
- [](../howto/robust_clustered_ses.md) — robust and clustered SEs for
  non-survey settings.
