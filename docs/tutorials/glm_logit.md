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

# GLM — Binomial logit
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

A logit with factor variables and an interaction, every Stata
`margins` analysis side-by-side with the `pymargins` call that
produces it.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(42)
n = 5000
female = rng.binomial(1, 0.52, n)
black = rng.binomial(1, 0.11, n)
age = rng.integers(20, 75, n)
agegrp = pd.cut(age, bins=[19, 29, 39, 49, 59, 69, 100],
                labels=[1, 2, 3, 4, 5, 6]).astype(int)
bmi = np.clip(22 + 0.15 * age + 1.5 * female + rng.normal(0, 4, n), 15, 50)
lp = (-4.0 + 0.55 * black + 0.10 * female + 0.06 * age + 0.03 * bmi
      + 0.5 * (agegrp == 2) + 0.9 * (agegrp == 3)
      + 1.4 * (agegrp == 4) + 2.0 * (agegrp == 5) + 2.6 * (agegrp == 6))
df = pd.DataFrame({
    "diabetes": rng.binomial(1, 1 / (1 + np.exp(-lp))),
    "black": black, "female": female, "age": age, "agegrp": agegrp, "bmi": bmi,
})

fit = smf.glm(
    "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
    data=df,
    family=sm.families.Binomial(),
).fit()
```

## APM — predictions at the typical profile

Stata: `margins agegrp, atmeans`.

For predicted probabilities the natural reporting scale is the
probability scale itself, so `linear_scale` is the default that
matches Stata:

```{code-cell} python
print(Margins.linear_scale(fit, at="typical").predict(
    atexog={"agegrp": list(range(1, 7))}
).summary())
```

## AAP — averaged over the sample

Stata: `margins agegrp`.

```{code-cell} python
print(Margins.linear_scale(fit, at="overall").predict(
    atexog={"agegrp": list(range(1, 7))}
).summary())
```

## APR — at representative values

Stata: `margins, at(age=(20 50 70)) atmeans`.

```{code-cell} python
print(Margins.linear_scale(fit, at="typical").predict(
    atexog={"age": [20, 50, 70]}
).summary())
```

## MEM and AME

Stata: `margins, dydx(age) atmeans` and `margins, dydx(age)`.

```{code-cell} python
Margins.linear_scale(fit, at="typical").dydx("age").summary()
print(Margins.linear_scale(fit, at="overall").dydx("age").summary())
```

## Discrete change

For binary regressors, `dydx` is a derivative — the discrete change
on the response scale is a contrast across the two levels:

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("black", [1, 0])
print(Margins.linear_scale(fit, at="overall").contrasts(
    scenarios=scen, contrasts=w
).summary())
```

## Plot: predicted probability by age group

```{code-cell} python
import matplotlib.pyplot as plt

res = Margins.linear_scale(fit, at="overall").predict(
    atexog={"agegrp": list(range(1, 7))}
)
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df_plot["agegrp"].astype(str), df_plot["estimate"],
       yerr=[df_plot["estimate"] - df_plot["ci_lower"],
             df_plot["ci_upper"] - df_plot["estimate"]],
       capsize=4, color="steelblue", edgecolor="black")
ax.set(xlabel="Age group", ylabel="P(diabetes=1)")
```

## Robust SEs

```{code-cell} python
print(Margins.linear_scale(fit, vcov="HC3", at="overall").dydx("age").summary())
```