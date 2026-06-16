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

# Fair (1978) — Extramarital affairs, two ways
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

The Fair affairs survey records the *number* of extramarital
encounters reported by 6,366 respondents, alongside age, years
married, religiousness, education, and a marriage-rating scale. The
same dataset supports two distinct analyses:

- a **logit** on `had_affair = affairs > 0` — the binary question, "is
  this respondent at risk of any affair at all?"
- a **Poisson** on `affairs` itself — the count question, "how many
  encounters do we expect from a respondent at this profile?"

Both are useful. The point of this demo is to run them through one
`pymargins` session each and contrast how they answer the same
substantive questions on different scales.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, pairwise  # 0.4.0: Margins -> GComputation

raw = sm.datasets.fair.load_pandas().data.copy()
raw["had_affair"] = (raw["affairs"] > 0).astype(int)
print(raw[["had_affair", "affairs", "age", "yrs_married",
           "religious", "rate_marriage"]].describe().round(2))
```

## 1. Logit on the binary outcome

```{code-cell} python
logit = smf.glm(
    "had_affair ~ age + yrs_married + religious + rate_marriage + educ",
    data=raw,
    family=sm.families.Binomial(),
).fit()

m_logit = Margins.linear_scale(logit, vcov="HC3", at="overall")
print(m_logit.dydx(["age", "yrs_married", "religious",
                    "rate_marriage", "educ"]).summary())
```

The AMEs are reported as changes in the *probability of any affair*
per unit of each covariate, averaged over the observed sample. The
signs match intuition: longer marriages and lower marriage ratings
raise the probability, religiousness lowers it.

## 2. Poisson on the count

```{code-cell} python
pois = smf.glm(
    "affairs ~ age + yrs_married + religious + rate_marriage + educ",
    data=raw,
    family=sm.families.Poisson(),
).fit()

m_pois = Margins.linear_scale(pois, vcov="HC3", at="overall")
print(m_pois.dydx(["age", "yrs_married", "religious",
                   "rate_marriage", "educ"]).summary())
```

These AMEs are the change in the *expected count* per unit. The
ranking of effects is similar to the logit; the magnitudes are not
comparable because they live on different response scales.

## 3. Marriage rating as a discrete contrast

`rate_marriage` is recorded on a 1–5 ordinal scale. The natural
policy question is "what's the gap between a 'very happy' marriage
(5) and a 'very unhappy' marriage (1)?" — a contrast, not a slope:

```{code-cell} python
scen, w = pairwise("rate_marriage", [5, 1])
print("Logit (probability scale):")
print(m_logit.contrasts(scenarios=scen, contrasts=w).summary())
print("\nPoisson (count scale):")
print(m_pois.contrasts(scenarios=scen, contrasts=w).summary())
```

The logit contrast says: averaging across the sample, a respondent
rating their marriage 5 has *X percentage points* lower probability
of any affair than the same respondent at rating 1. The Poisson
contrast gives the corresponding gap in expected count.

## 4. Scale matters — log vs linear inference for the Poisson

The Poisson canonical link is log. Opening the session on the log
scale gives a *rate ratio* interpretation with an asymmetric CI that
is multiplicative on the count scale, which is usually what readers
expect for count models:

```{code-cell} python
m_pois_log = Margins.log_scale(pois, vcov="HC3", at="overall")
rr = m_pois_log.contrasts(scenarios=scen, contrasts=w)
print(rr.summary())
```

The point estimate here is *log(rate ratio)*; the back-transformed
CI is the rate ratio with an asymmetric interval. See
[](../explanations/inference_scale.md) for the menu of scales.

The κ diagnostic checks whether the estimand surface is well
linearized on the chosen scale:

```{code-cell} python
print(m_pois_log.diagnose().summary())
```

A small κ confirms the delta method is safe on the log scale; a
large κ would trigger an automatic fallback to simulation.

## 5. Predicted profile over years married

```{code-cell} python
import matplotlib.pyplot as plt

years = list(range(0, 25))
res_l = m_logit.predict(atexog={"yrs_married": years})
res_p = m_pois.predict(atexog={"yrs_married": years})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, res, ylabel in [
    (axes[0], res_l, "P(any affair)"),
    (axes[1], res_p, "E(affairs)"),
]:
    d = res.to_frame()
    ax.plot(d["yrs_married"], d["estimate"], color="firebrick")
    ax.fill_between(
        d["yrs_married"], d["ci_lower"], d["ci_upper"],
        alpha=0.2, color="firebrick",
    )
    ax.set(xlabel="Years married", ylabel=ylabel)
fig.suptitle("Same dataset, two estimands")
fig.tight_layout()
```

The shapes are similar but not identical — the logit saturates as
`yrs_married` grows, the Poisson does not. Which curve to report
depends on whether the audience cares about *who* has affairs or
*how many*.

## Where to next

- [](../tutorials/glm_poisson.md) — Poisson tutorial with offsets.
- [](../explanations/inference_scale.md) — log vs linear vs logit
  scale and what each one buys you.
- [](mroz_lfp.md) — same model family, focused on the AME-vs-MEM
  question.