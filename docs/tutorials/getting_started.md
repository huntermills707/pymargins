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

This tutorial fits a small logit model, opens a `Margins` session, and
walks through the three orthogonal axes: estimand, aggregation, and
inference. By the end you should be able to map every common
Stata-style `margins` invocation to its `pymargins` equivalent.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins

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
fit.summary()
```

## 2. Open a session

A session commits to an inference scale, a vcov estimator, a
confidence level, an aggregation default (`at=`), and an inference
method. Once constructed, every call inherits these commitments.

```{code-cell} python
m = Margins.log_scale(fit, vcov="HC3", level=0.95, at="overall")
print(m.summary())
```

`Margins.log_scale(...)` is shorthand for
`Margins(..., phi=jnp.exp, phi_inv=jnp.log)`. See
[](../explanations/inference_scale.md) for why scale is a session-level
commitment.

## 3. Adjusted predictions

Predictions on the response scale, averaged over the observed
distribution of the *other* covariates (`at="overall"`, the AAP):

```{code-cell} python
m.predict(atexog={"treated": [0, 1]}).summary()
```

The same predictions at the typical covariate profile (the APM —
`at="typical"` uses median for continuous and mode for discrete):

```{code-cell} python
Margins.log_scale(fit, vcov="HC3", at="typical").predict(
    atexog={"treated": [0, 1]}
).summary()
```

## 4. Marginal effects

Average marginal effect of `age` on the response scale:

```{code-cell} python
m.dydx("age").summary()
```

A subgroup AME — same call, with `female` fixed at each level:

```{code-cell} python
m.dydx("age", atexog={"female": [0, 1]}).summary()
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
rr.summary()
```

Because the session is on the log scale, the back-transform turns a
log-RR into an RR with an asymmetric CI. See
[](../explanations/inference_scale.md).

## 6. Pre-flight diagnostic

```{code-cell} python
m.diagnose().summary()
```

`diagnose()` computes the κ curvature diagnostic on a sample of the
estimand surface. When κ is small the delta method is reliable; when
κ is large `pymargins` will *auto-fall-back to simulation* on the
next call. See [](../explanations/kappa_diagnostic.md).

## Where to next

- [](glm_logit.md) — a deeper logit walkthrough with factor variables.
- [](contrasts_and_did.md) — pairwise contrasts and 2×2 DiD.
- [](inference_methods.md) — delta vs simulation vs bootstrap.
