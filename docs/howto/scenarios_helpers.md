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

# Scenario helpers

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
    "group": rng.choice(["A", "B"], n),
    "preexist": rng.binomial(1, 0.4, n),
})
lp = (-1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
      + 0.2 * (df["region"] == "S") + 0.4 * (df["region"] == "E")
      - 0.1 * (df["region"] == "W")
      + 0.6 * (df["group"] == "B") + 0.4 * df["preexist"])
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female + treated + C(region) + C(group) + preexist", data=df,
              family=sm.families.Binomial()).fit()
m = GComputation(fit, at="overall", scale="log")
```


The `pymargins.scenarios` module ships factories for common contrast
patterns so you don't have to hand-build dicts and weight vectors.

```{code-cell} python
from pymargins import pairwise, reference, at_levels, grid, did, diff, all_pairwise
```

## `pairwise(var, [a, b])` — two scenarios, `[+1, -1]` contrast

```{code-cell} python
scen, w = pairwise("treated", [1, 0])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## `reference(var, levels, ref_level=...)` — each level vs baseline

```{code-cell} python
scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
print(m.contrasts(scenarios=scen, contrasts=W).summary())
```

## `all_pairwise(var, levels)` — every level pair

```{code-cell} python
scen, W = all_pairwise("region", ["N", "S", "E", "W"])
```

## `at_levels(var, levels=[...])` — predict at each level

```{code-cell} python
print(m.predict(atexog={"region": ["N", "S", "E", "W"]}).summary())
```

## `grid(**vars)` — Cartesian product of counterfactual values

```{code-cell} python
print(m.predict(atexog={"age": [25, 45, 65], "treated": [0, 1]}).summary())
```

## `did(g, c, ...)` — 2×2 DiD

```{code-cell} python
scen, w = did("group", "preexist",
              treated_level="B", control_level="A",
              post_level=1, pre_level=0)
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## `diff(n)` — `[-1, 0, …, 0, +1]` for an ordered grid

```{code-cell} python
scen = at_levels("age", levels=[25, 45, 65])
print(m.contrasts(scenarios=scen, contrasts=diff(3)).summary())   # age=65 minus age=25
```