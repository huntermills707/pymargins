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

# Survey designs — weights, strata, and clusters
Complex surveys sample with **unequal probabilities**, **stratification**, and
**clustering**. Ignore them and you get two distinct failures:

- *Wrong point estimate* — when the sampling weights are informative, the
  sample no longer looks like the population, so an unweighted average is
  biased.
- *Wrong standard error* — clustering makes observations within a cluster
  correlated, so the naïve SE (which assumes independent draws) is far too
  small.

This tutorial fixes each problem with `pymargins`, using a dataset where we can
**check our answers against the truth**.

## The data: California schools

We use the California Academic Performance Index (API) data — the canonical
teaching example from R's `survey` package. Its advantage over a simulation is
that the *entire population* is available, so every design-based estimate has a
known target:

- **`apipop`** — all 6 194 California schools (the population; ground truth).
- **`apistrat`** — a *stratified* sample of 200 schools. Strata are school
  type (`stype`: Elementary / Middle / High); the three types are sampled at
  different rates, so weights `pw` vary.
- **`apiclus1`** — a single-stage *cluster* sample: 15 school **districts**
  (`dnum`) were drawn, then every school in them was taken (183 schools).

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, SurveyDesign, steps  # 0.4.0: Margins -> GComputation

DATA = "../demos/data"
pop = pd.read_csv(f"{DATA}/apipop.csv")
strat = pd.read_csv(f"{DATA}/apistrat.csv")
clus = pd.read_csv(f"{DATA}/apiclus1.csv")

print(f"population: {len(pop):>5} schools")
print(f"stratified: {len(strat):>5} schools (strata = school type)")
print(f"cluster:    {len(clus):>5} schools in "
      f"{clus['dnum'].nunique()} districts")
```

## The target: population truth

Because we have the whole population, we can compute the answers a perfect
sample would reproduce. We model the 2000 API score (`api00`) as a function of
the share of students on subsidized meals (`meals`, a poverty proxy) and the
share of English-language learners (`ell`):

```{code-cell} python
TRUTH_MEAN = pop["api00"].mean()
truth_fit = smf.ols("api00 ~ meals + ell", data=pop).fit()
TRUTH_AME = truth_fit.params["meals"]

print(f"population mean api00:      {TRUTH_MEAN:7.2f}")
print(f"population slope on meals:  {TRUTH_AME:7.3f}")
```

Keep these two numbers in mind: a good survey estimate should land near them.

## 1. Weights correct bias

The stratified sample deliberately **oversamples** the rarer school types so
that Middle and High schools are well represented. Compare the population mix
to the sample mix:

```{code-cell} python
mix = pd.DataFrame({
    "population %": 100 * pop["stype"].value_counts(normalize=True),
    "sample %": 100 * strat["stype"].value_counts(normalize=True),
    "weight pw": strat.groupby("stype")["pw"].first(),
}).round(1)
print(mix)
```

Elementary schools are 71 % of the population but only 50 % of the sample, so
each sampled elementary school stands in for many more schools (a larger
`pw`). An unweighted average over-counts the oversampled Middle/High schools
and is biased; weighting by `pw` undoes the distortion:

```{code-cell} python
unweighted = strat["api00"].mean()
weighted = np.average(strat["api00"], weights=strat["pw"])

print(f"unweighted sample mean: {unweighted:7.2f}")
print(f"weighted sample mean:   {weighted:7.2f}")
print(f"population truth:        {TRUTH_MEAN:7.2f}")
```

The weighted mean is almost exactly the population value; the unweighted one is
biased downward by ~12 points. The same logic carries into a model: fitting
with `freq_weights` makes the coefficients — and therefore every
`GComputation` estimand built from them — generalize to the population.

```{code-cell} python
fit_w = smf.glm(
    "api00 ~ meals + ell",
    data=strat,
    freq_weights=strat["pw"].values,
).fit()

m_weighted = GComputation(fit_w, at="overall", weights=strat["pw"].values)
print(f"weighted population-mean prediction: "
      f"{float(m_weighted.predict().estimate):.2f}")
```

## 2. Strata give design-based standard errors

A weighted fit gets the *point estimate* right, but its SEs still assume simple
random sampling. To get **design-based** standard errors, declare the design
with `SurveyDesign`. For `apistrat` the design is: weights `pw`, strata
`stype`, and a finite-population correction `fpc` (each row carries the
stratum's population size, so the SE shrinks to reflect that we sampled a
non-trivial fraction of each stratum):

```{code-cell} python
# SurveyDesign wants integer-coded strata
stratum_codes = strat["stype"].astype("category").cat.codes.values

design = SurveyDesign(
    weights=strat["pw"].values,
    strata=stratum_codes,
    fpc=strat["fpc"].values,
)

m_survey = GComputation(
    steps.input(strat, design=design),
    outcome=fit_w,
    weights=strat["pw"].values,
    at="overall",
)
print(m_survey.dydx("meals").summary())
```

`pymargins` uses Taylor linearization (the survey sandwich). Because the model
was fit *with* the same weights, the adapter avoids double-counting them. The
AME of `meals` (~ −1.8 API points per percentage point on meals) brackets the
population truth, and the SE is honest about the stratified design.

## 3. Clustering inflates standard errors

Stratification *reduces* variance; clustering *increases* it. The cluster
sample `apiclus1` makes this dramatic: its 183 schools come from only **15
districts**, and schools in the same district are alike. Here the weights are
constant (every district had the same selection probability), so weighting does
*not* change the point estimate — clustering is purely a variance phenomenon.

We fit, then compare the naïve SE (pretend the 183 schools are independent) to
the design-based SE (`psu = dnum`):

```{code-cell} python
fit_c = smf.glm(
    "api00 ~ meals + ell",
    data=clus,
    freq_weights=clus["pw"].values,
).fit()

m_naive = GComputation(fit_c, at="overall", weights=clus["pw"].values)

design_c = SurveyDesign(
    weights=clus["pw"].values,
    psu=clus["dnum"].values,     # primary sampling unit = school district
    fpc=clus["fpc"].values,
)
m_cluster = GComputation(
    steps.input(clus, design=design_c),
    outcome=fit_c,
    weights=clus["pw"].values,
    at="overall",
)

se_naive = float(m_naive.dydx("meals").std_error)
se_cluster = float(m_cluster.dydx("meals").std_error)
print(f"naïve SE (ignores clustering): {se_naive:.3f}")
print(f"cluster-robust survey SE:      {se_cluster:.3f}")
print(f"design effect (variance ratio): "
      f"{(se_cluster / se_naive) ** 2:.1f}x")
```

The naïve SE is roughly **nine times too small** in standard-deviation terms —
an enormous design effect. Reporting the naïve interval here would massively
overstate precision; the survey SE is the one to trust.

## 4. What if you can't fit with weights?

Some estimators don't accept sampling weights. `pymargins` supports the
post-hoc route: fit **unweighted**, then attach the design through
`steps.input(..., design=...)`. You still get a design-based SE; the
point estimate just comes from the unweighted coefficients.

```{code-cell} python
fit_u = smf.glm("api00 ~ meals + ell", data=strat).fit()

m_posthoc = GComputation(
    steps.input(strat, design=design),
    outcome=fit_u,
    weights=strat["pw"].values,
    at="overall",
)
print(m_posthoc.dydx("meals").summary())
```

The estimate differs from the weighted-fit version (different coefficients) but
the SE is still survey-adjusted.

## 5. Bootstrap cross-check

A design-based bootstrap resamples PSUs **within** strata, preserving the
survey structure. With enough replicates it should track the linearization SE —
a useful independent check, and your only option for estimators that expose no
score (linearization needs one; the bootstrap does not).

```{code-cell} python
m_boot = GComputation(
    steps.input(clus, design=design_c),
    outcome=fit_c,
    weights=clus["pw"].values,
    at="overall",
    method="bootstrap",
    B=400,
    seed=0,
)
print(f"linearization SE: {se_cluster:.3f}")
print(f"bootstrap SE:     {float(m_boot.dydx('meals').std_error):.3f}")
```

## 6. Everything side by side

```{code-cell} python
def row(label, result):
    r = result.dydx("meals")
    return {"approach": label,
            "estimate": float(r.estimate),
            "std_error": float(r.std_error)}

summary = pd.DataFrame([
    row("apistrat: weighted (SRS SE)", m_weighted),
    row("apistrat: survey linearization", m_survey),
    row("apistrat: post-hoc unweighted", m_posthoc),
    row("apiclus1: naïve (ignores clusters)", m_naive),
    row("apiclus1: survey linearization", m_cluster),
    row("apiclus1: survey bootstrap", m_boot),
])
print(summary.round(3).to_string(index=False))
print(f"\npopulation truth slope on meals: {TRUTH_AME:.3f}")
```

What to read off the table:

- **Estimates** that use weights (or, for `apiclus1`, any method, since the
  weights are constant) cluster around the population truth.
- **SEs** within `apistrat` are similar — stratification only modestly changes
  precision here. Within `apiclus1` the naïve SE is far too small; the survey
  linearization and bootstrap agree and are the honest ones.

## Where to next

- [](../demos/california_api_survey.md) — a full end-to-end analysis on the
  stratified sample, including a policy-relevant contrast.
- [](../howto/robust_clustered_ses.md) — robust and clustered standard errors
  for non-survey settings.