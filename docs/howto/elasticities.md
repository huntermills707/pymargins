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
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.


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
m = Margins.linear_scale(fit, at="overall")
```


`dydx` returns the level derivative. The three elasticity / semi-elasticity
flavours are available directly as session methods:

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
# Full elasticity at the session's `at` setting
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
overall mean, or a theoretical value), the underlying recipe is a
`compose_results` of the slope and prediction.  The convenience methods above
are exactly this pattern wrapped for you.

For reference, `eyex` is equivalent to:

```python
from pymargins._result._margins import compose_results
import jax.numpy as jnp

slope_x = m.dydx("x")
pred    = m.predict()
x_bar   = float(df["x"].mean())

elasticity = compose_results(
    [slope_x, pred],
    fn=lambda t: t[0] * x_bar / t[1],
    label="eyex(x)",
)
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

Because `.scaled()` propagates the joint covariance, you can compute
elasticities for several subgroups and test differences between them:

```{code-cell} python
# Subgroup means for scaling
x_bar_0 = df.loc[df["female"] == 0, "x"].mean()
y_bar_0 = m.predict(atexog={"female": 0}).estimate.item()
x_bar_1 = df.loc[df["female"] == 1, "x"].mean()
y_bar_1 = m.predict(atexog={"female": 1}).estimate.item()

# Elasticity of x for female=0 and female=1
res_0 = m.dydx("x", atexog={"female": 0}).scaled(by=x_bar_0 / y_bar_0)
res_1 = m.dydx("x", atexog={"female": 1}).scaled(by=x_bar_1 / y_bar_1)

# Difference in elasticities with a proper SE
diff = res_1 - res_0
print(diff.summary())
```

## Plot: comparing elasticities across subgroups

```{code-cell} python
import matplotlib.pyplot as plt

df_0 = res_0.to_frame()
df_1 = res_1.to_frame()

labels = ["female=0", "female=1"]
estimates = [df_0["estimate"].iloc[0], df_1["estimate"].iloc[0]]
ci_lower = [df_0["ci_lower"].iloc[0], df_1["ci_lower"].iloc[0]]
ci_upper = [df_0["ci_upper"].iloc[0], df_1["ci_upper"].iloc[0]]

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
  a log-scale session (`Margins.log_scale(...)`) which linearises the problem.

* **Discrete inputs.** Elasticities are only defined for continuous variables —
  `pymargins` raises on discrete inputs.