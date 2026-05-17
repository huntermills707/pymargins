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

# Grid predictions


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
    "treatment": rng.binomial(1, 0.40, n),
})
lp = -1.5 + 0.04 * df["age"] + 0.8 * df["treatment"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + treatment", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```


For a Cartesian product of counterfactual values, use `grid` (or pass
a list to `atexog`):

```{code-cell} python
from pymargins import grid

# grid() builds scenario dicts for use in contrasts/evaluate
print(grid(age=[25, 45, 65], treatment=[0, 1]))

# For predict, pass the grid directly as atexog
print(m.predict(atexog={"age": [25, 45, 65], "treatment": [0, 1]}).summary())
```

## Plot: grid predictions as a heatmap

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

ages = np.arange(25, 66, 10)
treatments = [0, 1]
res = m.predict(atexog={"age": ages, "treatment": treatments})
df_grid = res.to_frame().pivot(index="treatment", columns="age", values="estimate")

fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(df_grid.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(ages)))
ax.set_xticklabels(ages)
ax.set_yticks(range(len(treatments)))
ax.set_yticklabels(["Control", "Treated"])
ax.set(xlabel="Age", ylabel="Treatment")
fig.colorbar(im, ax=ax, label="P(y=1)")
```

Variables not mentioned in `atexog` / `grid` follow the session's
`at=` rule (`"overall"` averages over the sample, `"typical"` /
`"mean"` hold them at a representative profile).

## Memory and large grids

Every grid point materialises a full copy of the design matrix.  For
a 10-point grid over a 1M-row dataset that is 10M rows.  Strategies:

- Use a smaller representative sample at session construction
  (`at="typical"`).
- Pass explicit `data=` overrides on the call to override the source
  rows with a smaller representative sample.
- Call `result.materialize()` promptly on results you intend to keep
  long-term; this drops the heavy machinery (gradients, design
  matrices, session refs).
