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

# Reading and controlling κ
> **Migration note (0.4.0):** κ is a per-query diagnostic in `GComputation`. The
> `method="auto"` resolution and `result.kappa` values shown below work as of
> v0.4.0; user-controlled borderline overrides are not yet exposed.

In **pymargins 0.4.0** κ is a *compile-time diagnostic*, not a runtime
fallback trigger. The estimator decides once at construction whether the
delta-method linearization is trustworthy for the declared estimand; it never
flips method later.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
})
lp = -1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female + treated", data=df,
              family=sm.families.Binomial()).fit(disp=0)
```

## Decide-once at compile

When `method="auto"`, pymargins builds a posture estimand (a prediction over
the base data), probes JAX differentiability, and computes worst-case κ on
the inference scale. If κ ≤ `KAPPA_BORDERLINE` (0.3 by default) it resolves to
`"delta"`; otherwise it resolves to `"simulation"` and records the reason in
the plan.

```{code-cell} python
est = GComputation(fit, at="overall", scale="response", method="auto", n_sim=4000)
print(est.plan.describe())
```

## Explicit method always wins

If you declare `method="delta"`, the delta method runs regardless of κ. κ is
still computed per query and stored in `result.kappa` for transparency, but it
does not change the method.

```{code-cell} python
est = GComputation(fit, at="overall", scale="response", method="delta")
res = est.predict(atexog={"age": [25, 45, 65]})
print(res.kappa)
```

## Pre-flight: read the plan

Use `est.plan.describe()` instead of a separate `diagnose()` call. It reports
the resolved method, the resolution reason (when auto), and the declared
analysis parameters.

```{code-cell} python
print(est.plan.describe())
```

## What to do when κ is high

A high κ means the delta-method linearization is a poor approximation for
your estimand. Three strategies, in order of preference:

1. **Change the inference scale.** Often the estimand is nearly linear on a
   different scale (log instead of identity, logit instead of probability).
   See [](../tutorials/scales_and_kappa.md).
2. **Accept the simulation resolution.** If the scale is already the most
   natural one, Krinsky–Robb simulation is a robust alternative. Make sure
   `n_sim` is large enough (≥ 4000) for stable tail quantiles.
3. **Switch to bootstrap.** Bootstrap does not rely on local linearity at all;
   it is the safest choice when both curvature and model misspecification are
   concerns. See [](bootstrap.md).

## Tuning the borderline

The default κ borderline is 0.3. If you need a stricter or looser threshold,
the current release requires choosing the inference method explicitly:

```{code-cell} python
# Always use delta, regardless of κ
est_delta = GComputation(fit, at="overall", scale="response", method="delta")

# Always use simulation / Krinsky–Robb
est_sim = GComputation(fit, at="overall", scale="response", method="simulation", n_sim=4000)
```

User-controlled borderline overrides are planned for a future release.
