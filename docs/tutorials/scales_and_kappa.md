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

The session-level `phi` / `phi_inv` pair picks the *inference scale*:
the scale on which the delta method and the κ diagnostic are
computed, and the scale whose CI endpoints get back-transformed to
the report.

Common scale helpers:

| Helper                          | `phi`       | When to use                            |
|---------------------------------|-------------|----------------------------------------|
| `Margins.linear_scale(...)`     | identity    | additive contrasts; AME on response    |
| `Margins.log_scale(...)`        | `exp`       | rate ratios, risk ratios, hazard ratios |
| `Margins.logit_scale(...)`      | `expit`     | odds ratios, probabilities             |
| `Margins.correlation_scale(...)` | `tanh`     | Fisher-z transformed correlations      |

The rule of thumb: pick the scale on which the *contrast* is most
nearly linear in `β`. That keeps κ small, the symmetric Wald CI
honest, and the back-transformed reporting CI asymmetric in the right
direction.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(0)
n = 1500
df = pd.DataFrame({"x": rng.normal(0, 1, n)})
lp = -3.0 + 1.8 * df["x"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
```

## κ as a pre-flight diagnostic

```{code-cell} python
print(Margins.linear_scale(fit, at="overall").diagnose().summary())
print(Margins.log_scale(fit, at="overall").diagnose().summary())
print(Margins.logit_scale(fit, at="overall").diagnose().summary())
```

When κ exceeds the session threshold (`kappa_threshold=0.3` by
default), the next call auto-falls-back to simulation. The summary
on every result tells you which inference path was actually used.

See [](../explanations/kappa_diagnostic.md) for the math.
