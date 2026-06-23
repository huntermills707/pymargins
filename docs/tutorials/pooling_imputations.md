# Pooling precomputed imputations (``pool_imputations``)
``pool_imputations`` combines **M finished analyses** — one per completed
dataset — into a single result using **Rubin's rules**.  It is the
artifacts-side counterpart to [``reimpute``](mi_via_reimpute.md): where
``reimpute`` cooks imputation into a bootstrap loop and needs a re-runnable
imputer, ``pool_imputations`` takes the M results you already have and performs
the analytical fan-in.

The package contributes the *combination*, not the imputation.  You generate
the M completed datasets however you like — ``sklearn.IterativeImputer``,
R ``mice`` exported to CSV, your own sampler — fit your model M times, compute
the **same estimand** on each, then hand the M ``GraphResult`` objects to
``pool_imputations``.

## When to use this

- You hold **M completed datasets** (imputer *outputs*), or M finished
  ``GraphResult`` objects — not a re-runnable imputer callable.  If you have
  a callable and want bootstrap inference, use [``reimpute``](mi_via_reimpute.md)
  instead.
- You want a **pooled point estimate _and_ interval** — Rubin's ``Q̄`` with a
  Student-*t* CI on the Rubin degrees of freedom.  (``reimpute`` keeps the
  single-imputation point estimate and only widens the CI.)
- You want the **MI diagnostics**: fraction of missing information (FMI),
  relative efficiency, and the Barnard–Rubin degrees of freedom.

## Quick example

Five posterior-draw imputations of a frame with 30% missingness in ``x1``, an
OLS fit on each, the average marginal effect of ``x1`` from each, pooled:

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from pymargins import GComputation, pool_imputations  # 0.4.0: Margins -> GComputation

# 1. Data with MAR missingness in x1
rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(scale=0.5, size=n)
df_nan = df.copy()
df_nan.loc[rng.uniform(size=n) < 0.30, "x1"] = np.nan

# 2. M completed datasets -> M fits -> the SAME estimand on each
per_imputation = []
for seed in range(5):
    imp = IterativeImputer(sample_posterior=True, random_state=seed, max_iter=10)
    completed = pd.DataFrame(imp.fit_transform(df_nan), columns=df_nan.columns)
    fit = smf.ols("y ~ x1 + x2", data=completed).fit()
    per_imputation.append(GComputation(fit, scale="identity").dydx("x1"))

# 3. Pool via Rubin's rules
pooled = pool_imputations(per_imputation)
print(pooled.summary())
```

```text
=======================================================
          GComputation Result (pooled, level=0.95)
=======================================================
    estimate  std err        t  P>|t|  [95% Conf. Int.]
-------------------------------------------------------
x1    0.6120   0.0257  23.8155  0.000    0.5616, 0.6624
=======================================================

n = 400
MI pooled (Rubin): M=5, FMI 0.058, df 1249.9, rel. eff. 0.989
```

The footer line is the MI audit trail: ``M`` imputations, the (max) FMI, the
(min) Rubin df, and the (min) relative efficiency.

## What pooling computes

Per estimand component, with within-imputation variances ``U_m = se_m²`` and
point estimates ``Q_m``, all on the **inference scale**:

```
Q̄  = mean(Q_m)                  # pooled point estimate
W   = mean(U_m)                  # within-imputation variance
B   = var(Q_m, ddof=1)           # between-imputation variance
T   = W + (1 + 1/M) · B          # total variance
SE  = sqrt(T)
```

``B`` is the only extra term multiple imputation introduces — the spread of the
M point estimates over the missing-data distribution.  It inflates the pooled
SE above the average within-imputation SE:

```python
within = np.mean([float(r.std_error) for r in per_imputation])
print(f"mean within-imputation SE : {within:.4f}")   # 0.0249
print(f"pooled SE                 : {float(pooled.std_error):.4f}")  # 0.0257
```

The confidence interval uses Student-*t* on the **Rubin degrees of freedom**
(not the package's usual *z*), so small *M* with real between-imputation
spread widens the interval correctly.  The diagnostics live on
``imputation_diagnostic``:

```python
diag = pooled.imputation_diagnostic
print(f"FMI                 : {float(diag.fmi):.3f}")   # 0.058
print(f"relative efficiency : {float(diag.relative_efficiency):.3f}")  # 0.989
print(f"Rubin df            : {float(diag.df):.1f}")    # 1249.9
```

For a vector estimand every field is per-component (``diag.fmi.shape ==
pooled.estimate.shape``); for a scalar estimand they are plain floats.

## Key rules

### 1. Same estimand, shared posture

All M results must describe the **same estimand**: identical labels, ``kind``,
``at``-grid, ``scenarios``, confidence ``level``, and inference scale.  Build
the M estimators identically and call the same estimand method on each.
``pool_imputations`` validates this up front and raises on the first mismatch:

```python
# Pooling a slope against a prediction -> ValueError
pool_imputations([per_imputation[0], some_prediction_result])
# pool_imputations: result 1 has labels=[...], expected [...].
# All results must be the same estimand computed under a shared posture ...
```

This is a cross-imputation invariant: the *posture* (scale, scenario grid,
``at``) must be fixed across the M fits.  Only the *data* — the completed
frame — changes from one imputation to the next.

### 2. Pooling happens on the inference scale

Rubin's rules assume approximate normality across imputations, so pooling is
done on the **inference scale** (log, logit, …) and mapped back through ``φ``
at the end — consistent with how the package handles non-identity scales
everywhere.  Use the matching ``GComputation(..., scale=...)`` constructor on each fit and
``pool_imputations`` does the rest:

```python
# Logit fits -> pool on the logit scale -> report on the probability scale
per_imp_logit = []
for seed in range(5):
    imp = IterativeImputer(sample_posterior=True, random_state=seed, max_iter=10)
    completed = df_bin.copy()
    completed[["x1", "x2"]] = imp.fit_transform(df_bin[["x1", "x2"]])
    fit = smf.logit("bin ~ x1 + x2", data=completed).fit(disp=False)
    per_imp_logit.append(
        GComputation(fit, scale="logit").predict(atexog={"x1": 0.0, "x2": 0.0})
    )

pooled_logit = pool_imputations(per_imp_logit)
print(pooled_logit.summary())
```

```text
===================================================================
                GComputation Result (pooled, level=0.95)
===================================================================
                estimate  std err        t  P>|t|  [95% Conf. Int.]
-------------------------------------------------------------------
x1=0.0, x2=0.0    0.4977   0.1204  -0.0755  0.940    0.4390, 0.5566
===================================================================

n = 400
Note: std err is on the inference scale; estimate and CI are on the reporting scale.
MI pooled (Rubin): M=5, FMI 0.072, df 813.3, rel. eff. 0.986
```

The pooled point estimate is the *geometric*-style mean implied by the scale
(arithmetic on the logit, mapped back to a probability), not a naive average of
the reported probabilities.

### 3. Inference-method-agnostic

Pooling reads only each branch's ``(estimate, std_error)``.  Each of the M
analyses may have used delta, simulation, **or** bootstrap independently —
``W_m = se_m²`` is just whatever variance that branch produced — and they pool
together fine.  This is why ``pool_imputations`` does **not** reuse the
single-estimator contrast machinery (``GraphResult.contrast``): the M results
come from M different estimators with M different ``Σ̂``.

### 4. Small-sample degrees of freedom (``complete_df``)

By default the Rubin (1987) degrees of freedom are used.  If you know the
complete-data residual df ``ν_com``, pass it for the Barnard–Rubin (1999)
small-sample correction, which shrinks the df toward ``ν_com``:

```python
pooled = pool_imputations(per_imputation, complete_df=n - 3)
```

### 5. A pooled result is a leaf

The M results came from M different estimators, so a pooled result carries no
Plan, gradient, or draws — it is a **terminal reduction**.  It still
re-levels and tests itself from the stored Rubin df:

```python
lo, hi = pooled.conf_int()  # level is fixed at construction; build a new GComputation to change it
tr = pooled.test(value=0.0, null_scale="inference")  # pooled t-statistic
```

Because it is a leaf, it cannot be composed further (``pooled - other`` raises
the ordinary same-estimator error).  Pool as the **last** step.

## ``pool_imputations`` vs ``reimpute``

Both handle missing data; they sit at different levels and answer to different
inputs.

| | ``pool_imputations`` | [``reimpute``](mi_via_reimpute.md) |
|---|---|---|
| You bring | M completed frames / M results | a re-runnable imputer callable |
| Mechanism | Rubin fan-in (analysis level) | impute-per-replicate (bootstrap loop) |
| Inference | any (delta / sim / bootstrap, per branch) | bootstrap only |
| Point estimate | pooled ``Q̄`` | single-imputation |
| Variance | within + between (Rubin) | one bootstrap distribution |
| Diagnostics | FMI, rel. eff., df | none (bootstrap byproduct) |
| Congeniality | your responsibility | largely sidestepped |

Rule of thumb: **precomputed frames → ``pool_imputations``; a live imputer you
are willing to bootstrap → ``reimpute``.**  For delta/simulation inference a
re-runnable imputer buys nothing that M precomputed frames do not, so pool.

## Limitations

- **At least two results.** ``B`` is undefined for ``M < 2``.
- **Each branch needs a positive SE.** A degenerate analysis (``se_m = 0`` for
  every branch) cannot be pooled and raises.
- **Congeniality is on you.** Rubin's validity is a property of the *pairing*
  of imputation model and analysis model; nothing here checks it.
- **No simultaneous CIs yet.** ``conf_int(simultaneous=True)`` on a pooled
  result raises ``NotImplementedError`` — a leaf has no draws to build a
  sup-*t* band from.
- **R ``mice`` frames work too.** Export each completed dataset to CSV, fit and
  compute your estimand in Python, and pool — you do not need a Python imputer,
  only the M completed frames.