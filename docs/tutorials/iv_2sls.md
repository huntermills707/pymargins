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

# Instrumental variables (2SLS / LIML / GMM)

`linearmodels.IV2SLS`, `IVLIML`, `IVGMM`, and friends are supported
through the linearmodels IV adapter. The fitted second-stage
coefficients enter `pymargins` just like an OLS fit, with the IV
covariance carried through to delta-method SEs.

```{code-cell} python
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from pymargins import Margins

rng = np.random.default_rng(2)
n = 2000
z = rng.normal(0, 1, n)
u = rng.normal(0, 1, n)
x = 0.5 * z + 0.3 * u + rng.normal(0, 1, n)
y = 1.0 + 0.8 * x + 0.5 * rng.normal(0, 1, n) + 0.7 * u

df = pd.DataFrame({"y": y, "x": x, "z": z})
df["const"] = 1.0

fit = IV2SLS(df["y"], df[["const"]], df[["x"]], df[["z"]]).fit()
```

## AME of the endogenous regressor

```{code-cell} python
m = Margins.linear_scale(fit, at="overall")
print(m.dydx("x").summary())
```

## Predictions at counterfactual `x`

```{code-cell} python
print(m.predict(atexog={"x": [-1.0, 0.0, 1.0]}).summary())
```

## Contrast: endogenous regressor +1 vs baseline

```{code-cell} python
print(m.contrasts(
    scenarios=[
        {"atexog": {"x": 1.0}, "label": "x=1"},
        {"atexog": {"x": 0.0}, "label": "x=0"},
    ],
    contrasts=[+1, -1],
).summary())
```
