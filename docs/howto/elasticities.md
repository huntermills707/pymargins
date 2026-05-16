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
n = 2000
df = pd.DataFrame({
    "x": rng.normal(50, 10, n),
    "female": rng.binomial(1, 0.52, n),
})
lp = -1.5 + 0.03 * df["x"] - 0.3 * df["female"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ x + female", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```

# Elasticities and semi-elasticities

`dydx` returns the level derivative. The other three elasticity flavors
correspond to the standard Stata `dyex` / `eyex` / `eydx` methods —
build them as compositions through `evaluate`, or scale the AME by
hand:

| Stata `margins`    | Quantity                            | Build                                        |
|--------------------|-------------------------------------|----------------------------------------------|
| `dydx(x)`          | level change                        | `m.dydx("x")`                                |
| `dyex(x)`          | `dy/d(ln x)`                        | `m.dydx("x").scaled(by=x_bar)`               |
| `eyex(x)`          | full elasticity `(dy/dx) (x/y)`     | `m.dydx("x").scaled(by=x_bar / y_bar)`       |
| `eydx(x)`          | `d(ln y)/dx`                        | `m.dydx("x").scaled(by=1 / y_bar)`           |

`scaled` is a deterministic transform — it propagates SE, CI, and
covariance correctly under the delta method.

## Computing `x_bar` and `y_bar`

For an **average** elasticity (`at="overall"`), `x_bar` is the sample
mean of `x` and `y_bar` is the average predicted response at the
observed covariate profiles.  Both are easy to recover from the
session:

```{code-cell} python
x_bar = df["x"].mean()                       # or median, depending on theory
y_bar = m.predict().estimate.item()          # AAP on the response scale

# eyex: full elasticity at the mean
print(m.dydx("x").scaled(by=x_bar / y_bar).summary())
```

For an elasticity **at a representative profile** (`at="typical"` or
with `atexog`), use the values at that profile:

```{code-cell} python
profile_x = 5.0
y_at = m.predict(atexog={"x": profile_x}).estimate.item()
print(m.dydx("x", atexog={"x": profile_x}).scaled(by=profile_x / y_at).summary())
```

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

Elasticities are only defined for continuous variables — `pymargins`
raises on discrete inputs.
