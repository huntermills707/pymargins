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

# Transform pipelines — honest CIs for data-dependent preprocessing
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

Most applied work runs some preprocessing *before* the model: imputing a
missing covariate, trimming implausible values, dropping outliers by a
data-driven rule. The estimate then carries the fingerprints of that
preprocessing — but the standard inference pretends the cleaned data were
handed down from on high. If the imputation model, the trim bounds, or the
outlier flags would have come out differently on a different sample, your
confidence interval is too narrow.

A `pymargins` **transform pipeline** closes that gap. You pass
`transforms=[...]` to a bootstrap session; each stage is a `frame → frame`
transform that the bootstrap **re-derives on every replicate**, exactly the
way [`matching`](../howto/matching.md) re-matches. The imputer re-fits, the
trim bounds recompute, the outlier rule re-runs — so the resample
distribution absorbs the variability of the preprocessing itself.

This demo runs three stages end-to-end on the Wage panel:

1. `reimpute` — bootstrap-then-impute multiple imputation, and how its CI
   widens relative to a naive single-imputation bootstrap.
2. Composition — chaining `reimpute` with a `trim` stage in one pipeline.
3. `drop_outliers` — a data-driven outlier rule re-derived per replicate.

It closes with the **structural guards**: the combinations the session
rejects at construction, and why.

```{code-cell} python
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.datasets import wage_panel
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from pymargins import GComputation, drop_outliers, reimpute, trim  # 0.4.0: Margins -> GComputation

cols = ["lwage", "exper", "educ", "married", "union"]
df = wage_panel.load().reset_index(drop=True)[cols].copy()
print(df.describe().round(2))
```

## 1. `reimpute` — multiple imputation under the bootstrap

We inject MAR missingness into `educ`: workers with more experience are
more likely to have an unrecorded education, so the missingness depends on
an observed covariate (missing-at-random, not completely-at-random).

```{code-cell} python
rng = np.random.default_rng(7)
p_miss = 1 / (1 + np.exp(-(df["exper"] - df["exper"].mean()) / 2))
miss = rng.uniform(size=len(df)) < 0.30 * p_miss

df_nan = df.copy()
df_nan.loc[miss, "educ"] = np.nan
print(f"missing educ: {int(miss.sum())} rows ({miss.mean():.1%})")
```

The **point estimate** comes from a single, cheap mean-fill imputation —
this matches how the package already treats bootstrap (the estimate is from
the original fit, the CI from the resample distribution).

```{code-cell} python
df_init = df_nan.fillna(df_nan.mean(numeric_only=True))
fit = smf.ols("lwage ~ exper + educ + married + union", data=df_init).fit()
print(fit.params.round(4))
```

For the bootstrap we need a **stochastic, runnable** imputer: a callable
that takes a DataFrame and returns a fit-and-imputed DataFrame. We use
sklearn's `IterativeImputer` with `sample_posterior=True` so each draw adds
residual noise — a deterministic imputer (mean-fill) would leave the CIs too
narrow, and `reimpute` warns you if it detects one.

```{code-cell} python
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)


def imputer(frame):
    return pd.DataFrame(imp.fit_transform(frame), columns=frame.columns)
```

### Naive single-imputation bootstrap

First the wrong-but-common approach: bootstrap the *already* mean-filled
frame. Every replicate resamples the same frozen imputed values, so the CI
sees only sampling variability — not the uncertainty about what the missing
education values actually were.

```{code-cell} python
m_naive = Margins.linear_scale(fit, method="bootstrap", n_boot=500, rng_seed=3)
print(m_naive.dydx("educ").summary())
```

### Bootstrap-then-impute

Now the pipeline. `reimpute(imputer, incomplete=df_nan)` tells the bootstrap
to resample the *incomplete* frame and re-impute it fresh on every
replicate. The session per-replicate seed flows into the imputer's
`random_state`, so the run is reproducible.

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # silence the imputer's convergence chatter
    m_mi = Margins.linear_scale(
        fit,
        transforms=[reimpute(imputer, incomplete=df_nan)],
        method="bootstrap",
        n_boot=500,
        rng_seed=3,
    )
print(m_mi.dydx("educ").summary())
```

The point estimate is identical — both sessions report the slope from the
same `fit`. But the **standard error is larger** and the interval is wider:
that extra width is the imputation-model uncertainty the naive bootstrap
threw away. Because the missingness lands on `educ`, the `educ` slope is
exactly the coefficient that should pay for it; a covariate with no
missingness in its predictors would be essentially unchanged.

For the full contract — why the imputer must be stochastic, how seeding
works, the BCa restriction — see the
[`reimpute` tutorial](../tutorials/mi_via_reimpute.md).

## 2. Composing stages

Stages compose in the order you list them; each one sees the frame the
previous stage produced. Here we re-impute, then `trim` away any imputed
`educ` below 2 (years of schooling that low are almost certainly a bad draw,
not a real observation). Both steps re-run on every replicate.

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m_compose = Margins.linear_scale(
        fit,
        transforms=[
            reimpute(imputer, incomplete=df_nan),
            trim(lower=2.0, columns=["educ"]),
        ],
        method="bootstrap",
        n_boot=400,
        rng_seed=3,
    )
print(m_compose.dydx("educ").summary())
```

`trim` sets `alters_rows=True`: it changes the row set, so the bootstrap
refits with `index=None` rather than carrying the resample index through. The
engine reads that flag off the stage's declared contract — you do not have to
manage index bookkeeping yourself.

## 3. `drop_outliers` — a re-derived detection rule

`drop_outliers(rule)` takes a callable returning a boolean mask of rows to
drop. The rule is data-dependent here — it flags log-wages more than five
median-absolute-deviations below the median — so it genuinely should be
recomputed on each resample. We drop back to the complete data for this part.

```{code-cell} python
def far_below(frame):
    med = frame["lwage"].median()
    mad = (frame["lwage"] - med).abs().median()
    return frame["lwage"] < med - 5 * mad


print(f"flagged on the full sample: {int(far_below(df).sum())} rows")

df_clean = df[~far_below(df)].reset_index(drop=True)
fit_clean = smf.ols("lwage ~ exper + educ + married + union", data=df_clean).fit()

m_out = Margins.linear_scale(
    fit_clean,
    transforms=[drop_outliers(far_below)],
    method="bootstrap",
    n_boot=500,
    rng_seed=0,
)
print(m_out.dydx("educ").summary())
```

Each replicate recomputes the median and MAD on its *own* resample and
re-applies the threshold, so the interval reflects the fact that the set of
flagged rows is itself a random quantity. When the flagged rows barely move
the coefficient — as for the `educ` slope here — the pipeline CI tracks the
plain bootstrap closely; when they do move it, the pipeline is the honest
interval.

## 4. Structural guards

The pipeline rejects combinations that would silently produce wrong numbers,
and it does so at session construction — not three minutes into a bootstrap.

A stage that declares `requires_resampling=True` (like `reimpute`) is
bootstrap-only; there is no resample distribution under the delta method to
re-derive it over:

```{code-cell} python
try:
    Margins.linear_scale(
        fit,
        transforms=[reimpute(imputer, incomplete=df_nan, warn_on_deterministic=False)],
        method="delta",
    )
except ValueError as exc:
    print(exc)
```

A row-altering stage (`drop_outliers`, `trim`) cannot be combined with
session `weights=`: the stage thins the rows but the weight vector is not
thinned with it, so the weighted aggregation would misalign.

```{code-cell} python
try:
    Margins.linear_scale(
        fit,
        transforms=[drop_outliers(far_below)],
        weights=np.ones(len(df)),
        method="bootstrap",
    )
except ValueError as exc:
    print(exc)
```

The same spirit covers the rest of the contract: `survey_design` rejects
row-altering and source-overriding stages (a fixed survey design cannot have
its rows or source frame changed underneath it), `matching=` and
`transforms=` are mutually exclusive, and `ci_method="bca"` is rejected
because the BCa jackknife would operate on the raw frame without running the
pipeline.

## Where to next

- [](../tutorials/mi_via_reimpute.md) — the full `reimpute` contract:
  stochastic-imputer requirement, seeding, and limitations.
- [](../howto/matching.md) — the re-derive-per-replicate pattern that the
  pipeline generalizes.
- [](../explanations/session_precommitment.md) — why the session freezes
  inference parameters once the bootstrap bank is built.