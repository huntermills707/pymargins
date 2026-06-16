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

# ANES96 — Multinomial logit on party identification
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

The 1996 American National Election Study records party
identification (`PID`) on a 7-point scale from strong Democrat (0)
to strong Republican (6), along with demographics and a 7-point
self-placed liberal-conservative scale (`selfLR`). Multinomial
logit on this kind of outcome is famously easy to *fit* and famously
easy to *misreport* — the coefficient table gives one log-odds
contrast per category against the baseline, and almost nobody
reading the paper will read it correctly.

`pymargins` handles MNL the same way it handles any other model:
ask for AMEs or predicted probabilities, and the result carries an
outcome axis you can slice.

This demo walks through:

1. Fit the MNL.
2. Predicted-probability profile across the ideology scale.
3. AMEs of ideology for each outcome category — the right way to
   report MNL effects.
4. A category-collapsed contrast (Democrat-leaning vs
   Republican-leaning) at a representative voter.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

raw = sm.datasets.anes96.load_pandas().data.copy()
# 7-point PID coded 0..6. Keep covariates compact for the demo.
df = raw[["PID", "TVnews", "selfLR", "age", "educ", "income"]].dropna().copy()
df["PID"] = df["PID"].astype(int)
print(df.describe().round(2))
print("\nOutcome distribution:")
print(df["PID"].value_counts().sort_index())
```

## 1. Fit the MNL

```{code-cell} python
fit = smf.mnlogit(
    "PID ~ TVnews + selfLR + age + educ + income",
    data=df,
).fit(disp=False)
print(fit.summary().tables[1])
```

The coefficient table reports log-odds of each PID category against
the baseline (PID=0, strong Democrat). A coefficient of 0.5 on
`selfLR` for "PID=6" does not mean "0.5 percentage points more
likely to be a strong Republican" — it's a log-odds contrast at a
covariate combination that may or may not exist in the data. This
is exactly why marginal-effects machinery exists.

## 2. Predicted probability profile across ideology

```{code-cell} python
import matplotlib.pyplot as plt

m = Margins.linear_scale(fit, at="overall")
lr_grid = list(range(1, 8))  # selfLR scale 1..7
res = m.predict(atexog={"selfLR": lr_grid})

est = np.asarray(res.estimate)        # (n_lr, n_outcomes)
ci = np.asarray(res.conf_int())       # (2, n_lr, n_outcomes)
n_lr, n_out = est.shape

pid_labels = [
    "Strong Dem", "Weak Dem", "Ind-Dem",
    "Independent",
    "Ind-Rep", "Weak Rep", "Strong Rep",
]
# Blue = Democrat, red = Republican, grey = independent — standard US
# political color convention.
colors = ["#1f4e96", "#5588c8", "#a6c8e8",
          "#7f7f7f",
          "#e8a6a6", "#c85555", "#961f1f"]

fig, ax = plt.subplots(figsize=(7, 4))
for j in range(n_out):
    ax.plot(lr_grid, est[:, j], color=colors[j], label=pid_labels[j], lw=2)
    ax.fill_between(
        lr_grid, ci[0, :, j], ci[1, :, j], alpha=0.15, color=colors[j]
    )
ax.set(xlabel="Self-placed liberal–conservative (1=liberal, 7=conservative)",
       ylabel="P(PID category)",
       title="Predicted party-ID by ideology, sample-averaged")
ax.legend(loc="upper center", ncol=4, fontsize=8)
```

The crossover pattern — strong-Democrat probability falling, strong-
Republican rising, with the middle categories peaking near the
ideology center — is the kind of finding that becomes a figure in
the paper. The coefficient table cannot give you this picture.

## 3. AME of ideology, per outcome category

The slope of `selfLR` is different for each PID category, because
the probabilities sum to 1 and a positive AME for one category
forces negative AMEs elsewhere:

```{code-cell} python
ame_lr = m.dydx("selfLR")
print(ame_lr.summary())
```

Each row of the result is the average per-unit change in the
probability of *that* category, averaged across the sample. The
columns sum to zero by construction (modulo numerical noise) — a
sanity check that the AMEs are coherent.

The same for `educ`:

```{code-cell} python
print(m.dydx("educ").summary())
```

## 4. Collapsed contrast — Democrat-leaning vs Republican-leaning

For a "headline number" you often want to collapse PID categories
into two camps. The cleanest way is to build a custom estimand
that wraps the predicted-probability vector:

```{code-cell} python
import jax.numpy as jnp

def dem_minus_rep(p):
    # p: (n_scenarios, n_outcomes=7) — per-scenario category probabilities.
    # Returns one gap per scenario.
    dem = p[..., 0:3].sum(axis=-1)   # strong Dem, weak Dem, ind-Dem
    rep = p[..., 4:7].sum(axis=-1)   # ind-Rep, weak Rep, strong Rep
    return dem - rep

# One scenario per ideology level; compose collapses the outcome axis.
scenarios = [{"atexog": {"selfLR": v}, "label": f"selfLR={v}"}
             for v in lr_grid]
res_diff = m.evaluate(scenarios=scenarios, compose=dem_minus_rep)
print(res_diff.summary())
```

The result is the gap (P(Democrat-leaning) − P(Republican-leaning))
at each ideology level, with delta-method standard errors propagated
through both the MNL link and the category sum. That single number
is what most write-ups actually need.

## Where to next

- [](../tutorials/mnlogit.md) — the underlying multinomial-logit
  tutorial.
- [](../howto/evaluate.md) — `evaluate` for custom estimands.
- [](../howto/contrasts.md) — built-in contrast helpers for the
  common pairwise / reference / pooled patterns.