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

# Multinomial logit

Multi-outcome models return one estimate per outcome category. The
session API is unchanged; results carry an outcome axis you can slice
with `result.outcome("category")`.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymargins import Margins

rng = np.random.default_rng(3)
n = 4000
df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": rng.uniform(20, 200, n),
})
# Three-category outcome: bus / car / bike
util_car = 0.05 * df["age"] + 0.01 * df["income"]
util_bike = 0.02 * (60 - df["age"]) + rng.normal(0, 1, n)
util_bus = np.zeros(n)
util = np.column_stack([util_bus, util_car, util_bike])
util += rng.gumbel(size=util.shape)
df["mode"] = util.argmax(1)  # 0=bus, 1=car, 2=bike

# Note: pymargins uses integer outcome indices for MNLogit.

fit = smf.mnlogit("mode ~ age + income", data=df).fit()
```

## Predicted probability by mode at representative ages

```{code-cell} python
m = Margins.linear_scale(fit, at="overall")
preds = m.predict(atexog={"age": [25, 45, 65]})
print(preds.summary())
```

The response scale for multinomial predictions is the probability scale
itself, so `linear_scale` is the natural default.  If you need CIs that
are guaranteed to stay inside \[0, 1\] for a particular category, you
could open a `logit_scale` session and then use `outcome="car"`, but
`linear_scale` is the standard choice for tables and plots.

## Plot: predicted probability curves by mode

For multinomial predictions the result arrays are 2-D
(scenarios × outcomes), so we build the plot DataFrame by hand:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

ages = list(range(18, 71, 2))
res = m.predict(atexog={"age": ages})

est = np.asarray(res.estimate)
se = np.asarray(res.std_error)
ci = np.asarray(res.conf_int())
n_scen, n_out = est.shape

rows = []
for i in range(n_scen):
    for j in range(n_out):
        rows.append({
            "age": ages[i],
            "outcome": j,
            "estimate": est[i, j],
            "ci_lower": ci[0, i, j],
            "ci_upper": ci[1, i, j],
        })
df_plot = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 4))
for outcome, sub in df_plot.groupby("outcome"):
    ax.plot(sub["age"], sub["estimate"], label=f"Mode {outcome}")
    ax.fill_between(
        sub["age"], sub["ci_lower"], sub["ci_upper"], alpha=0.1
    )
ax.set(xlabel="Age", ylabel="P(mode)")
ax.legend(title="Outcome", loc="upper right")
```

## AME of income on the probability of `car`

```{code-cell} python
ame = m.dydx("income")
print(ame.outcome(1).summary())
```
