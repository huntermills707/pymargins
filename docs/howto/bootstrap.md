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

# Bootstrap inference

Switch the session's inference method to `"bootstrap"` and pick the
number of replicates. The default scheme is *pairs* — rows resampled
IID with replacement.

```{code-cell} python
m = Margins.log_scale(fit, method="bootstrap", n_boot=2000, vcov="HC3")
print(m.dydx("age").summary())
```

Parallelism uses thread pools; BLAS threads are pinned to 1 per worker
to avoid oversubscription:

```{code-cell} python
m = Margins.log_scale(fit, method="bootstrap", n_boot=2000, n_jobs=-1)
```

## Point estimates under bootstrap

The point estimate stays the analytic `g(β̂)` — `pymargins` does not
report the bootstrap mean as the estimate, matching Stata's
convention.  The bootstrap is used *only* for the standard error and
the empirical quantiles of the CI.  This keeps the estimator
consistent even when the bootstrap distribution is biased (e.g. in
small samples).

## Failed refits

Failed refits are caught and counted; a `RuntimeWarning` fires when
the failure rate exceeds 5%.  If you see this warning, inspect the
model specification — non-convergence on 5% of bootstrap samples
usually indicates separation, perfect multicollinearity, or a
misspecified link function.

For correlated data use [](cluster_block_bootstrap.md).
