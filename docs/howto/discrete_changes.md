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

# Discrete changes for binary / categorical regressors
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.


```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
    "region": rng.choice(["N", "S", "E", "W"], n),
})
lp = (-1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
      + 0.2 * (df["region"] == "S") + 0.4 * (df["region"] == "E")
      - 0.1 * (df["region"] == "W"))
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female + treated + C(region)", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```


`dydx` on a binary regressor is a derivative — it evaluates the slope
at the midpoint of the 0→1 jump (or averages midpoints over the
sample).  That is almost never the quantity you want.  The correct
quantity is the **discrete change**: the predicted difference between
the two levels, holding everything else constant.

## Binary regressor (0 → 1)

Use `contrasts` with the `pairwise` helper:

```{code-cell} python
from pymargins import GComputation, pairwise  # 0.4.0: Margins -> GComputation

m = Margins.linear_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## Multi-level factor (each level vs baseline)

```{code-cell} python
from pymargins import reference

scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
print(m.contrasts(scenarios=scen, contrasts=W).summary())
```

## All-pairs comparisons

All-pairs are returned in one result with a joint covariance, enabling
post-hoc combination and testing:

```{code-cell} python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)
print(res.summary())
```