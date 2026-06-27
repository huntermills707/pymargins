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

# Getting started
This tutorial fits a small logit model, builds a `GComputation`
estimator, and walks through the three orthogonal axes: estimand,
aggregation, and inference. By the end you should be able to map every
common Stata-style `margins` invocation to its `pymargins` equivalent.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(0)
n = 4000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
})
lp = -1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
```

## 1. Fit a model

```{code-cell} python
fit = smf.glm(
    "y ~ age + female + treated",
    data=df,
    family=sm.families.Binomial(),
).fit()
print(fit.summary())
```

## 2. Build an estimator

A `GComputation` estimator commits to an inference scale, a vcov
estimator, a confidence level, an aggregation default (`at=`), and an
inference method. Once constructed, every call inherits these
commitments.

```{code-cell} python
m = GComputation(fit, scale="log", vcov="HC3", level=0.95, at="overall")
```

`scale="log"` is shorthand for the back-transform
`jnp.exp` / `jnp.log`.  We use it here because we will compute a
risk-ratio contrast below; for predicted probabilities
`scale="response"` (identity) or `scale="logit"` are also valid
choices.  See [](../explanations/inference_scale.md) for the full
scale menu.

## 3. Adjusted predictions

Predictions on the response scale, averaged over the observed
distribution of the *other* covariates (`at="overall"`, the AAP):

```{code-cell} python
print(m.predict(atexog={"treated": [0, 1]}).summary())
```

Because the estimator is on the log scale, the CI is asymmetric on
the probability scale (multiplicative around the point estimate).  For
a probability that can approach 1, `scale="logit"` keeps the CI inside
(0, 1); `scale="response"` gives a symmetric CI on the probability
scale itself.

The same predictions at the typical covariate profile (the APM —
`at="typical"` uses median for continuous and mode for discrete):

```{code-cell} python
print(GComputation(fit, scale="log", vcov="HC3", at="typical").predict(
    atexog={"treated": [0, 1]}
).summary())
```

## 4. Marginal effects

Average marginal effect of `age` on the response scale:

```{code-cell} python
print(m.dydx("age").summary())
```

A subgroup AME — same call, with `female` fixed at each level:

```{code-cell} python
print(m.dydx("age", atexog={"female": [0, 1]}).summary())
```

## 5. Contrasts

A risk-ratio contrast (`treated=1` vs `treated=0`):

```{code-cell} python
rr = m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
)
print(rr.summary())
```

Because the estimator is on the log scale, the back-transform turns a
log-RR into an RR with an asymmetric CI. See
[](../explanations/inference_scale.md).

## Plot: prediction curve over a continuous variable

```{code-cell} python
import matplotlib.pyplot as plt

ages = list(range(20, 76, 2))
res = m.predict(atexog={"age": ages, "treated": [0, 1]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for level, sub in df_plot.groupby("treated"):
    label = "Treated" if level == 1 else "Control"
    ax.plot(sub["age"], sub["estimate"], label=label)
    ax.fill_between(
        sub["age"], sub["ci_lower"], sub["ci_upper"], alpha=0.15
    )
ax.set(xlabel="Age", ylabel="P(y=1)")
ax.legend(title="Treatment")
```

## Where to next

- [](glm_logit.md) — a deeper logit walkthrough with factor variables.
- [](contrasts_and_did.md) — pairwise contrasts and 2×2 DiD.
- [](inference_methods.md) — delta vs simulation vs bootstrap.