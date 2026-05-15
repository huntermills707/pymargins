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

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import Margins

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
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```

# Reading and controlling the κ fallback

Every result records the inference path actually used. When the
session's κ threshold is exceeded, delta auto-falls-back to
simulation, and the summary annotates the fallback reason.

```{code-cell} python
m = Margins.log_scale(fit, kappa_threshold=0.3, method="delta", n_sim=4000)
res = m.predict(atexog={"age": [25, 45, 65]})
print(res.summary())     # notes 'fallback: simulation (κ=0.42 > 0.30)'
```

## Pre-flight diagnostic

```{code-cell} python
print(m.diagnose().summary())
```

`diagnose()` samples the estimand surface and reports the κ
distribution; if it is uniformly large, change the inference scale
(see [](../tutorials/scales_and_kappa.md)) before paying for
simulation on every call.

## What to do when κ is high

A high κ means the delta-method linearization is a poor approximation
for your estimand.  Three strategies, in order of preference:

1. **Change the inference scale.**  Often the estimand is nearly
   linear on a different scale (log instead of identity, logit instead
   of probability).  See [](../tutorials/scales_and_kappa.md).
2. **Accept the simulation fallback.**  If the scale is already the
   most natural one, Krinsky–Robb simulation is a robust alternative.
   Just make sure `n_sim` is large enough (≥ 4000) for stable tail
   quantiles.
3. **Switch to bootstrap.**  Bootstrap does not rely on local
   linearity at all; it is the safest choice when both curvature and
   model misspecification are concerns.  See [](bootstrap.md).

## Disabling the fallback

Set `kappa_threshold=float("inf")` to force the chosen `method=` to
run regardless of curvature.

```{code-cell} python
m = Margins.log_scale(fit, kappa_threshold=float("inf"))
```

## Disabling diagnostics altogether

In tight loops, set `diagnostics=False` to skip the κ computation:

```{code-cell} python
m = Margins.log_scale(fit, diagnostics=False)
```
