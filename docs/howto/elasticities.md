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

# Elasticities and semi-elasticities

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "x": rng.normal(50, 10, n),
    "female": rng.binomial(1, 0.52, n),
})
lp = -1.5 + 0.03 * df["x"] - 0.3 * df["female"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ x + female", data=df,
              family=sm.families.Binomial()).fit()
m = GComputation(fit, at="overall", scale="identity")
```


`dydx` returns the level derivative. The three elasticity / semi-elasticity
flavours are available directly as estimator methods:

| Stata `margins`    | Quantity                            | pymargins method       |
|--------------------|-------------------------------------|------------------------|
| `dydx(x)`          | level change                        | `m.dydx("x")`          |
| `dyex(x)`          | `dy/d(ln x)`                        | `m.dyex("x")`          |
| `eyex(x)`          | full elasticity `(dy/dx) (x/y)`     | `m.eyex("x")`          |
| `eydx(x)`          | `d(ln y)/dx`                        | `m.eydx("x")`          |

Each method computes the slope and prediction internally and composes them
with the correct transform, carrying the joint gradient through the delta
method so standard errors and confidence intervals are valid.

## Basic usage

```{code-cell} python
# Full elasticity at the estimator's `at` setting
print(m.eyex("x").summary())

# Semi-elasticities
print(m.eydx("x").summary())
print(m.dyex("x").summary())
```

These methods honour `atexog` and `over` just like `dydx`:

```{code-cell} python
# Elasticity of x within each level of female
print(m.eyex("x", over="female").summary())
```

## Under the hood: manual composition with `.scaled()`

If you need a custom scaling factor (e.g. a subgroup mean that is not the
overall mean, or a theoretical value), the underlying recipe is a ratio of the
slope and prediction.  The convenience methods above are exactly this pattern
wrapped for you.

For reference, `eyex` is equivalent to scaling the level derivative by the
ratio of the predictor mean to the predicted response mean:

```python
import numpy as np

slope_x = m.dydx("x")
pred    = m.predict()
x_bar   = float(df["x"].mean())
y_bar   = float(pred.estimate)

elasticity = slope_x.scaled(by=x_bar / y_bar, units="eyex(x)")
```

The `.scaled(by=...)` helper offers a lighter-weight alternative when the
scaling factor is a simple scalar:

```{code-cell} python
x_bar = df["x"].mean()
y_bar = m.predict().estimate.item()

# eyex via scaled()
print(m.dydx("x").scaled(by=x_bar / y_bar).summary())
```

`scaled` is a deterministic transform — it propagates SE, CI, and
covariance correctly under the delta method.

## Subgroup elasticities

Pass `over=` to compute a separate elasticity for each level of a discrete
covariate.  `eyex` uses the group-specific means of `x` and the group-specific
average prediction, and the joint covariance is carried through the delta
method, so you can test differences between groups with `result.contrast()`:

```{code-cell} python
import numpy as np

res = m.eyex("x", over="female")
print(res.summary())

# Difference in elasticities with a proper SE
diff = res.contrast(np.array([[1, -1]]), labels=["female=1 - female=0"])
print(diff.summary())
```

## Plot: comparing elasticities across subgroups

```{code-cell} python
import matplotlib.pyplot as plt

df_res = res.to_frame()

labels = ["female=0", "female=1"]
estimates = df_res["estimate"].tolist()
ci_lower = df_res["ci_lower"].tolist()
ci_upper = df_res["ci_upper"].tolist()

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(labels, estimates,
       yerr=[np.array(estimates) - np.array(ci_lower),
             np.array(ci_upper) - np.array(estimates)],
       capsize=4, color="seagreen", edgecolor="black")
ax.set(ylabel="Elasticity of x")
```

## Pitfalls

* **Division by near-zero predictions.** Elasticities blow up when the
  predicted response is close to zero. The convenience methods clip near-zero
  denominators at `1e-12` by default; for a more principled solution, consider
  a log-scale estimator (`GComputation(..., scale="log")`) which linearises the problem.

* **Discrete inputs.** Elasticities are only defined for continuous variables —
  `pymargins` raises on discrete inputs.