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

# Wage panel — union premium with entity fixed effects

The `wage_panel` data from `linearmodels` follows 545 young men
across 1980–1987 with annual observations on log wages, education,
experience, union membership, and marital status. The substantive
question is the union wage premium — but ordinary OLS confounds it
with unobserved worker quality. An entity-FE specification absorbs
that.

This demo walks through:

1. An entity-FE specification with cluster-robust SEs.
2. The marginal union premium, with the FE-corrected interval.
3. Profile of predicted wages over experience, by union status.
4. A Krinsky–Robb simulation cross-check on the analytic clustered
   SE.

```{code-cell} python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.datasets import wage_panel

from pymargins import Margins, pairwise

raw = wage_panel.load()
df = raw.set_index(["nr", "year"]).copy()
print(df[["lwage", "educ", "exper", "expersq",
          "union", "married"]].describe().round(2))
```

## 1. Entity-FE specification

```{code-cell} python
fe = PanelOLS(
    df["lwage"],
    df[["exper", "expersq", "union", "married"]],
    entity_effects=True,
).fit(cov_type="clustered", cluster_entity=True)
print(fe.summary.tables[1])
```

Within-worker variation only. The union coefficient now identifies
the wage change *for the same person* when they enter or leave a
union — that's the policy quantity.

(`educ` is absorbed because it does not vary over time within a
worker.)

## 2. Marginal union premium on the wage scale

`lwage` is log wages and `union` is binary, so the natural estimand
is a *contrast* (union vs non-union) rather than a slope. On a
linear-scale session that contrast is the gap in log wages — i.e.,
the union premium expressed as a log-point gap:

```{code-cell} python
m = Margins.linear_scale(fe, at="overall")
scen, w = pairwise("union", [1, 0])
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

For policy reporting it's clearer to express this as a percentage
gap. Switch to a log session and ask for the same contrast — the
back-transform turns the log-difference into a multiplicative
premium:

```{code-cell} python
m_log = Margins.log_scale(fe, at="overall")
print(m_log.contrasts(scenarios=scen, contrasts=w).summary())
```

The point estimate on the log scale is the log-wage gap; the
back-transformed interval is the multiplicative ratio (1.X means
"X% premium").

## 3. Experience profile by union status

For the predicted-curve plot, switch to `at="typical"` so each grid
point evaluates at one representative profile instead of averaging
over thousands of rows — the question is "what is the predicted log
wage at *this* experience level," not a sample-averaged AAP:

```{code-cell} python
import matplotlib.pyplot as plt

m_typ = Margins.linear_scale(fe, at="typical")
exper_grid = np.arange(0, 18, 2)
res = m_typ.predict(atexog={
    "exper": exper_grid,
    "expersq": exper_grid ** 2,
    "union": [0, 1],
})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for union_val, sub in df_plot.groupby("union"):
    label = "Union" if union_val == 1 else "Non-union"
    ax.plot(sub["exper"], sub["estimate"], label=label)
    ax.fill_between(
        sub["exper"], sub["ci_lower"], sub["ci_upper"], alpha=0.2
    )
ax.set(xlabel="Years of experience", ylabel="Predicted log wage")
ax.legend()
```

Pair `exper` and `expersq` together in the same `atexog` so the
quadratic is internally consistent at every grid point — a common
source of nonsense plots when people forget.

## 4. Krinsky–Robb simulation as a sanity check

The analytic cluster-robust SE assumes the within-cluster dependence
structure is well-approximated by the sandwich formula. With
moderate cluster counts (here, 545 workers) that's usually fine —
but it's worth a cross-check. The cluster-block bootstrap is the
most rigorous check; at the time of writing it is not yet wired up
for `linearmodels` panel adapters (re-fitting on resampled panels
needs special handling), so the practical alternative is Krinsky–Robb
simulation, which draws coefficient vectors from the fitted MVN and
re-evaluates the estimand:

```{code-cell} python
m_sim = Margins.log_scale(
    fe, at="overall",
    method="simulation", n_sim=2000, rng_seed=0,
)
print(m_sim.contrasts(scenarios=scen, contrasts=w).summary())
```

If the simulation CI is materially wider than the analytic CI on the
log scale, the response surface is curved enough that the delta
method's linearization is starting to bite — the [κ
diagnostic](../explanations/kappa_diagnostic.md) will usually have
flagged this already. For a within-worker linear model on `lwage` the
two methods agree closely.

## Where to next

- [](../tutorials/panel_fe.md) — the underlying panel tutorial.
- [](../howto/cluster_block_bootstrap.md) — when and why to prefer
  the cluster-resampled bootstrap over the analytic clustered SE.
- [](../howto/robust_clustered_ses.md) — the menu of clustered
  covariance estimators.
