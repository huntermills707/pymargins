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

Complex surveys sample with unequal probabilities, stratification, and
clustering. A naïve analysis ignores all three and produces biased point
estimates (when weights are informative) and wrong standard errors
(when design features are ignored). This tutorial walks through the
`pymargins` tools for each problem.

The example uses **simulated data** so we know the ground truth:
- Three strata (small / medium / large cities)
- Three PSUs (cities) sampled per stratum
- Unequal sampling weights by stratum
- A binary outcome driven by age, income, and city size

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, SurveyDesign

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Simulate a stratified two-stage sample
# ---------------------------------------------------------------------------
# Population: 20 cities per stratum. We sample 3 cities per stratum
# (PSU = city) and then 100 people per sampled city.
strata_names = ["small", "medium", "large"]
n_strata = len(strata_names)
n_psu_per_stratum = 3
n_per_psu = 100

rows = []
for h, stratum in enumerate(strata_names):
    # City-level intercept (PSU effect)
    city_effects = rng.normal(0, 0.3, 20)
    sampled_cities = rng.choice(20, size=n_psu_per_stratum, replace=False)
    for psu_idx, city in enumerate(sampled_cities):
        # Individual-level data
        age = rng.integers(25, 70, n_per_psu)
        income = rng.normal(50 + 5 * h, 15, n_per_psu)
        # True DGP: older, richer, and larger-city people are more likely
        lp = (-2.0
              + 0.04 * age
              + 0.02 * income
              + 0.3 * h
              + city_effects[city])
        prob = 1 / (1 + np.exp(-lp))
        y = rng.binomial(1, prob)
        # Sampling weight varies by stratum (oversampling small cities)
        weight = (20 / 3) * (500 / 100) * (1 + 0.3 * h)
        rows.append(pd.DataFrame({
            "y": y,
            "age": age,
            "income": income,
            "stratum": stratum,
            "psu": f"{stratum}_{psu_idx}",
            "weight": weight,
        }))

df = pd.concat(rows, ignore_index=True)
print(df.head())
print(f"\nN = {len(df)}, strata = {df['stratum'].unique().tolist()}")
print(f"PSUs per stratum = {df.groupby('stratum')['psu'].nunique().tolist()}")
```

## 1. Fit the model

We fit a logit **with sampling weights** so the coefficients generalize
to the population. `freq_weights=` tells statsmodels to maximize the
weighted log-likelihood:

```{code-cell} python
fit = smf.glm(
    "y ~ age + income",
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df["weight"].values,
).fit(disp=False)
print(fit.summary().tables[1])
```

## 2. Unweighted vs weighted predictions

Even though the model was fit with weights, `Margins` defaults to an
unweighted aggregation unless you pass `weights=`. Compare the two:

```{code-cell} python
m_naive = Margins(fit, at="overall")
m_weighted = Margins(fit, at="overall", weights=df["weight"].values)

print("Naive  P(y=1 | age=40):",
      float(m_naive.predict(atexog={"age": [40]}).estimate))
print("Weighted P(y=1 | age=40):",
      float(m_weighted.predict(atexog={"age": [40]}).estimate))
```

Because the sampling weights are informative (larger cities are
under-sampled relative to their population size), the weighted
prediction is slightly higher — the population contains more large-city
residents, who have a higher baseline probability.

## 3. Unweighted vs weighted AMEs

The same logic applies to average marginal effects. The weighted AME
averages over the population distribution, not the sample distribution:

```{code-cell} python
print("Naive  AME of age:")
print(m_naive.dydx("age").summary())
print("\nWeighted AME of age:")
print(m_weighted.dydx("age").summary())
```

## 4. Survey-adjusted standard errors

Now we declare the full design. `SurveyDesign` holds the weights, PSU
identifiers, and stratum identifiers. `pymargins` uses Taylor
linearization (the survey sandwich) to compute design-based standard
errors. Because the model was fit with weights, the adapter
automatically avoids double-counting:

```{code-cell} python
# Map stratum names to integer codes for SurveyDesign
stratum_codes = df["stratum"].astype("category").cat.codes

survey = SurveyDesign(
    weights=df["weight"].values,
    psu=df["psu"].values,
    strata=stratum_codes.values,
)

m_survey = Margins(fit, survey_design=survey, weights=df["weight"].values)
print(m_survey.dydx("age").summary())
```

The **point estimate** is the same as the weighted AME (because we passed
`weights=`), but the **standard error** is larger than the naïve one.
The design-based SE accounts for the fact that observations are
clustered within cities, not independent.

## 5. What if you fit unweighted?

`pymargins` also supports the post-hoc approach: fit unweighted and
declare weights only via `survey_design`. This is useful when your
modelling package does not support weighted fits:

```{code-cell} python
fit_unweighted = smf.glm(
    "y ~ age + income",
    data=df,
    family=sm.families.Binomial(),
).fit(disp=False)

m_posthoc = Margins(
    fit_unweighted,
    survey_design=survey,
    weights=df["weight"].values,
)
print(m_posthoc.dydx("age").summary())
```

The point estimate will differ from the weighted-fit approach (because
the coefficients are different), but the standard error is still
survey-adjusted.

## 6. Stratified bootstrap (optional cross-check)

A design-based bootstrap resamples PSUs *within* strata, preserving the
survey structure. With enough replicates it should give a similar SE to
the linearization estimator:

```{code-cell} python
m_boot = Margins(
    fit,
    survey_design=survey,
    weights=df["weight"].values,
    method="bootstrap",
    n_boot=300,
    rng_seed=42,
)
print(m_boot.dydx("age").summary())
```

## 7. Compare all approaches side-by-side

```{code-cell} python
results = pd.DataFrame({
    "approach": ["naïve", "weighted only", "survey linearization",
                 "survey bootstrap", "post-hoc unweighted"],
    "estimate": [
        float(m_naive.dydx("age").estimate),
        float(m_weighted.dydx("age").estimate),
        float(m_survey.dydx("age").estimate),
        float(m_boot.dydx("age").estimate),
        float(m_posthoc.dydx("age").estimate),
    ],
    "std_error": [
        float(m_naive.dydx("age").std_error),
        float(m_weighted.dydx("age").std_error),
        float(m_survey.dydx("age").std_error),
        float(m_boot.dydx("age").std_error),
        float(m_posthoc.dydx("age").std_error),
    ],
})
print(results.round(5))
```

The pattern to expect:
- **Estimate:** naïve may differ from weighted/survey (bias from
  informative weights). Weighted, linearization, and bootstrap agree on
  the point estimate because they all use the same aggregation weights.
  The post-hoc unweighted estimate differs because the coefficients are
  different.
- **SE:** naïve and weighted-only are too optimistic because they ignore
  clustering. Survey linearization and bootstrap are larger and similar
  to each other.

## Where to next

- [](../demos/california_api_survey.md) — a real-world e2e analysis
  with California school data, using weighted fits.
- [](../howto/robust_clustered_ses.md) — robust and clustered
  standard errors for non-survey settings.
