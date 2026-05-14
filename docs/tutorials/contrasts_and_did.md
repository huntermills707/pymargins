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

# Contrasts and difference-in-differences

Pairwise contrasts, reference-level contrasts, and 2×2 DiD all reduce
to a list of scenarios plus a contrast weight vector.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, pairwise, reference, did

rng = np.random.default_rng(8)
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

fit = smf.glm(
    "condX ~ C(group) * C(preexist) + age",
    data=df,
    family=sm.families.Binomial(),
).fit()
m = Margins.linear_scale(fit, at="overall")
```

## Pairwise risk difference

```{code-cell} python
scen, w = pairwise("group", ["B", "A"])
m.contrasts(scenarios=scen, contrasts=w).summary()
```

## Reference-level contrasts for a factor

```{code-cell} python
scen, w_mat = reference(
    "group", levels=["A", "B"], ref_level="A"
)
m.contrasts(scenarios=scen, contrasts=w_mat).summary()
```

## 2×2 Difference-in-differences (Ai & Norton, 2003)

The `did` helper builds four scenarios and a `[+1, -1, -1, +1]`
contrast weight, evaluating the DiD on the response scale where the
clinical / economic question lives — not on the log-odds scale of
the interaction coefficient.

```{code-cell} python
scen, w = did("group", "preexist", group_levels=["A", "B"],
              condition_levels=[0, 1])
m.contrasts(scenarios=scen, contrasts=w).summary()
```

See [](../math.rst) for why DiD must be evaluated on the response
scale for nonlinear models.
