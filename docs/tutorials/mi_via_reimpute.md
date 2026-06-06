# Multiple imputation via ``reimpute``

The ``reimpute`` stage implements bootstrap-then-impute: one imputation per
bootstrap replicate, with the imputer re-fit from scratch each time.  This
injects imputation-model parameter uncertainty into the bootstrap
distribution, producing valid confidence intervals without a Rubin combinator.

## When to use this

- Your data has missing values that you do not want to listwise-delete.
- You have a **runnable imputer** (a Python callable that takes a DataFrame
  and returns an imputed DataFrame).  Frozen, pre-computed imputed frames are
  not supported — the stage must be able to re-impute a resampled draw.
- You are willing to use **bootstrap inference** (``method="bootstrap"``).
  ``reimpute`` is invalid under delta-method or simulation inference.

## Quick example

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from pymargins import Margins, reimpute

# 1. Build data with MAR missingness
rng = np.random.default_rng(42)
n = 400
df = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
})
df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(scale=0.5, size=n)
missing = rng.uniform(size=n) < 0.25
df_nan = df.copy()
df_nan.loc[missing, "x1"] = np.nan

# 2. Fit the model on a single initial imputation
#    (the point estimate comes from this fit)
df_init = df_nan.fillna(df_nan.mean())
fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

# 3. Wrap the imputer so it returns a DataFrame
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)

def imputer(frame):
    arr = imp.fit_transform(frame)
    return pd.DataFrame(arr, columns=frame.columns)

# 4. Run bootstrap-then-impute
m = Margins(
    fit,
    transforms=[reimpute(imputer, incomplete=df_nan)],
    method="bootstrap",
    n_boot=1000,
    rng_seed=7,
)
r = m.predict(atexog={"x1": 0, "x2": 0})
```

## Key rules

### 1. The imputer must be stochastic

A deterministic imputer (e.g. ``SimpleImputer(strategy="mean")`` or
``IterativeImputer(sample_posterior=False)``) fills the conditional mean with
no residual draw.  The bootstrap still re-fits each replicate, so the filled
values track each resample's mean, but the *missing residual variance* means
your CIs will be too narrow.

``reimpute`` runs a cheap construction-time guard: it calls the imputer twice
on the same small sample, and if the output is byte-identical it warns you.
You can suppress this with ``warn_on_deterministic=False``.

### 2. Seed the imputer for reproducibility

The session ``rng_seed`` controls the bootstrap resample indices, but it does
**not** automatically seed your imputer.  For reproducible draws you must set
``random_state`` on the imputer object itself:

```python
imp = IterativeImputer(random_state=42, sample_posterior=True)
```

If the imputer exposes ``random_state=None``, ``reimpute`` warns at
construction.

**Important:** create a fresh imputer object for each ``Margins`` session.
Sharing a single imputer instance across sessions can leave internal state from
the first session and break reproducibility, even with a fixed
``random_state``.

### 3. The point estimate is single-imputation

The reported ``estimate`` comes from the model you passed to ``Margins``
(fitted on the initial single imputation).  The bootstrap supplies
imputation-aware **CIs**, not a pooled point estimate.  This matches how the
package already treats bootstrap: the estimate is from the original fit, the
CI from the resample distribution.

### 4. Structural columns must be complete

Columns that define the inference design — ``cluster=``, ``survey_design``
PSU/strata, and ``weights=`` — must not contain missing values.  Only
*substantive* columns may be imputed.

### 5. Bootstrap only

``reimpute`` sets ``requires_resampling=True``, so ``method="delta"`` or
``method="simulation"`` raises at construction.

## Comparison to plain bootstrap

Bootstrap-then-impute widens the CI for coefficients affected by missingness
because each replicate draws a different imputation.  Coefficients with no
missingness in their predictors are essentially unchanged.

## Limitations

- **Frozen frames are not supported on this path.** ``reimpute`` needs a
  re-runnable imputer because it re-imputes every bootstrap replicate. If you
  already hold M *precomputed* completed frames (e.g. from R ``mice`` exported
  to CSV), pool them with Rubin's rules via
  [``pool_imputations``](pooling_imputations.md) instead — no Python imputer
  required, just the M completed frames.
- **BCa CIs are rejected.** The BCa jackknife operates on the raw incomplete
  frame without re-imputing, producing an inconsistent acceleration parameter.
  Use ``ci_method="percentile"`` (default) or ``"studentized"``.
- **Survey designs are incompatible.** ``survey_design`` + ``reimpute`` raises
  because the fixed-design assumption breaks when the source frame is
  incomplete.
