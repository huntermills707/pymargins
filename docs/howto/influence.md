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

# Per-observation influence
`GraphResult.influence()` returns, for each training observation, how much
that observation contributes to the estimate — the leave-one-out *deletion
influence* `θ̂ − θ_{(-i)}`. Unlike statsmodels' built-in influence diagnostics
(which describe the coefficients `β`), this works for the **estimand you
actually report**: an average marginal effect, a contrast, an elasticity, an
RMST. A large positive value means dropping that observation would pull the
estimate down.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation  # 0.4.0: Margins -> GComputation

rng = np.random.default_rng(1)
n = 180
df = pd.DataFrame({
    "x": rng.normal(0, 1, n),
    "z": rng.normal(0, 1, n),
})
lp = -0.5 + 0.8 * df["x"] - 0.4 * df["z"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ x + z", data=df, family=sm.families.Binomial()).fit()
m = GComputation(fit, at="overall", scale="identity")

ame = m.dydx("x")
print(ame.summary())
```

## Basic usage

```{code-cell} python
infl = ame.influence()
infl.shape          # (n_obs,) for a scalar estimand
```

The values are signed and (to first order) sum to zero, since the score
contributions cancel at the fitted estimate:

```{code-cell} python
print("sum of influence ≈", float(infl.sum()))
```

## Finding the most influential observations

Sort by absolute influence to see which rows move the marginal effect most —
a Cook's-distance analogue, but for the AME rather than the coefficients.

```{code-cell} python
order = np.argsort(-np.abs(infl))
top = df.iloc[order[:5]].copy()
top["influence"] = infl[order[:5]]
print(top)
```

## Reconstructing the variance

Summing the squared influence reconstructs the **robust (HC0) sandwich
variance** of the estimand — because `Σ_i score_i score_iᵀ` is exactly the
sandwich "meat":

```{code-cell} python
se_from_influence = np.sqrt(np.sum(infl ** 2))

m_hc0 = GComputation(fit, at="overall", vcov="HC0", scale="identity")
se_hc0 = float(np.atleast_1d(m_hc0.dydx("x").std_error)[0])

print(f"sqrt(Σ influence²) = {se_from_influence:.6f}")
print(f"HC0 robust SE      = {se_hc0:.6f}")
```

The two agree to numerical precision. (The default model-based SE differs
because it assumes the information equality holds in-sample.) Summing influence
*within clusters* before squaring gives the cluster-robust variance — the same
construction at a coarser grouping.

## Two routes, same quantity

The example above uses the **delta method**: the analytical influence function
`∇h(β̂)ᵀ Σ̂ score_i(β̂)`, where `Σ̂ score_i` is the one-step (infinitesimal
jackknife) approximation to `β̂ − β̂_{(-i)}`.

A **BCa bootstrap** result returns the *exact* leave-one-out refits that were
already computed for the acceleration constant — no extra work:

```{code-cell} python
m_boot = GComputation(fit, at="overall", method="bootstrap",
    B=200, seed=0,
    ci="bca", scale="identity")
infl_jack = m_boot.dydx("x").influence()

corr = np.corrcoef(infl, infl_jack)[0, 1]
print(f"correlation(delta, jackknife) = {corr:.4f}")
```

The two are strongly correlated. They are not identical: the delta route is a
*one-step* approximation, so it differs from the exact refit by the leverage
factor `1/(1 − h_ii)` and, for a nonlinear estimand like a logit AME, by
higher-order curvature. The delta route is far cheaper (no refits) and has no
sample-size cap; the jackknife is exact but limited (see pitfalls).

## Vector estimands

For a length-`k` vector result (e.g. a prediction over several scenarios),
`influence()` returns an `(n_obs, k)` matrix — one influence column per
component:

```{code-cell} python
r_vec = m.predict(atexog={"x": [-1.0, 0.0, 1.0]})
infl_vec = r_vec.influence()
infl_vec.shape
```

## Plot: an influence index plot

```{code-cell} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.stem(np.arange(n), infl, markerfmt=" ", basefmt="C7-")
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="observation index", ylabel="influence on AME(x)")
top_i = order[0]
ax.annotate(f"obs {top_i}", (top_i, infl[top_i]),
            textcoords="offset points", xytext=(5, 5), color="C3")
```

## Which models support it

The delta-method route needs per-observation scores via `adapter.score_obs()`.
This is implemented for the statsmodels **GLM** (all families/links), **OLS**,
**Logit/Probit**, and **Poisson/NegativeBinomial** adapters. Adapters without
`score_obs()` raise `NotImplementedError` on the delta route — but the BCa
bootstrap route works for any bootstrappable model.

## Pitfalls

* **Keep the estimator alive.** The delta route reads scores from the live
  adapter, so the `GComputation` estimator must still be in scope:
  `m.dydx("x").influence()` works, but `GComputation(fit, scale="identity").dydx("x").influence()`
  may fail if the estimator is garbage-collected first. Materialized results
  (which drop the estimator reference) cannot use the delta route.

* **Jackknife size cap.** The BCa leave-one-out refits are skipped when
  `n_obs` (or the number of clusters) exceeds 200, and for block bootstrap
  where leave-one-out is not well defined. In those cases use the delta route,
  which has no size limit.

* **`vcov` changes the meaning.** Influence is formed with the estimator's frozen
  `Σ̂`. Under a robust or cluster `vcov` it becomes the corresponding sandwich
  influence rather than the model-based one — usually what you want, but worth
  stating in a write-up.