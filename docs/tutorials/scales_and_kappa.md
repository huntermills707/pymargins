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

# Inference scales and the κ diagnostic
The estimator-level `phi` / `phi_inv` pair picks the *inference scale*:
the scale on which the delta method is computed, and the scale whose
CI endpoints get back-transformed to the report.

Common scale choices:

| Constructor keyword                     | Back-transform | When to use                            |
|-----------------------------------------|----------------|----------------------------------------|
| `scale="identity"`                      | identity       | additive contrasts; AME on response    |
| `scale="log"`                           | `exp`          | rate ratios, risk ratios, hazard ratios |
| `scale="logit"`                         | `expit`        | odds ratios, probabilities             |
| `scale=(jnp.tanh, jnp.arctanh)`         | `tanh`         | Fisher-z transformed correlations      |

The rule of thumb: pick the scale on which the *contrast* is most
nearly linear in `β`. That keeps the symmetric Wald CI honest and the
back-transformed reporting CI asymmetric in the right direction.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({"x": rng.normal(0, 1, n)})
lp = -3.0 + 1.8 * df["x"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
```

## Comparing scales on the same estimand

```{code-cell} python
print(GComputation(fit, at="overall", scale="identity").dydx("x").summary())
print(GComputation(fit, at="overall", scale="log").dydx("x").summary())
print(GComputation(fit, at="overall", scale="logit").dydx("x").summary())
```

The point estimate and CI width change with the scale. Choose the
scale that makes the contrast most interpretable for your audience,
and switch to `method="simulation"` or `method="bootstrap"` if you
suspect the delta-method linearization is poor.

See [](../explanations/kappa_diagnostic.md) for the curvature math.