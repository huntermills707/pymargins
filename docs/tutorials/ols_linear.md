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

# OLS — linear regression

For OLS the response scale and the linear predictor scale coincide;
the natural session is `Margins.linear_scale(...)`.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(11)
n = 3000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.5, n),
    "education": rng.choice(["hs", "college", "grad"], size=n, p=[.5, .35, .15]),
})
df["wage"] = (
    10 + 0.3 * df["age"] - 2.0 * df["female"]
    + 4 * (df["education"] == "college") + 8 * (df["education"] == "grad")
    + rng.normal(0, 5, n)
)

fit = smf.ols("wage ~ age + C(female) + C(education) + age:C(female)",
              data=df).fit()
```

## AME of `age` overall

```{code-cell} python
m = Margins.linear_scale(fit, vcov="HC2", at="overall")
print(m.dydx("age").summary())
```

## AME of `age` by sex

```{code-cell} python
print(m.dydx("age", atexog={"female": [0, 1]}).summary())
```

## Predicted wage at representative education levels

```{code-cell} python
print(m.predict(atexog={"education": ["hs", "college", "grad"]}).summary())
```

## A pairwise wage gap

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("education", ["grad", "hs"])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```
