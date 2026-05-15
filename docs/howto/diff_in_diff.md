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
n = 4000
df = pd.DataFrame({
    "group": rng.choice(["A", "B"], n),
    "preexist": rng.binomial(1, 0.4, n),
    "age": rng.integers(20, 80, n),
})
lp = (-1.5 + 0.6 * (df["group"] == "B") + 0.4 * df["preexist"]
      + 0.5 * ((df["group"] == "B") & (df["preexist"] == 1))
      + 0.02 * df["age"])
df["condX"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("condX ~ C(group) * C(preexist) + age", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.linear_scale(fit, at="overall")
```

# Difference-in-differences on the response scale

For a 2×2 DiD on a nonlinear model, evaluate the four cells on the
response scale and difference them. The interaction coefficient
itself is on the link scale and does *not* answer the question
(Ai & Norton, 2003).

```{code-cell} python
from pymargins import Margins, did

m = Margins.linear_scale(fit, vcov="HC3", at="overall")

scen, w = did(
    "group", "preexist",
    treated_level="B", control_level="A",
    post_level=1, pre_level=0,
)
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

At a single representative patient profile:

```{code-cell} python
print(m.contrasts(
    scenarios=did(
        "group", "preexist",
        treated_level="B", control_level="A",
        post_level=1, pre_level=0,
        age=60,
    )[0],
    contrasts=[+1, -1, -1, +1],
).summary())
```

The four cell predictions and the two simple effects share the same
joint covariance, so the DiD's standard error is exact under the
delta method.
