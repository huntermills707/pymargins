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

# scikit-learn models

```{code-cell} python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from pymargins import GComputation  # 0.4.0: Margins -> GComputation
from pymargins.adapters import SklearnBootstrapAdapter

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "age": rng.normal(50, 10, size=n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
})
df["age_sq"] = df["age"] ** 2
df["y"] = (
    10.0
    + 0.2 * df["age"]
    + 0.001 * df["age_sq"]
    - 0.5 * df["female"]
    + 0.8 * df["treated"]
    + rng.normal(size=n)
)
```


`pymargins` supports scikit-learn estimators through a bootstrap-only
adapter.  Because sklearn models do not expose a parametric coefficient
vector or covariance matrix, **only bootstrap inference is available**.
Delta-method and Krinsky–Robb simulation are not supported.

:::{important}
For linear models (OLS, GLM, etc.) prefer statsmodels, linearmodels, or
another parametric backend.  scikit-learn support is intended for
non-parametric estimators — random forests, gradient boosting, support
vector machines — where bootstrap is already the natural inference
strategy.
:::

## Basic workflow

Fit the sklearn estimator as usual, wrap it in
`SklearnBootstrapAdapter`, and build a `GComputation` estimator with
`method="bootstrap"`:

```{code-cell} python
X = df[["age", "female", "treated", "age_sq"]]
y = df["y"]
model = LinearRegression()
model.fit(X, y)

adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
m = GComputation(model, adapter=adapter, method="bootstrap", B=200, seed=42)
print(m.predict(atexog={"treated": [0, 1]}).summary())
```

## Slopes with derived terms

If your model includes interactions, polynomials, splines, or other
transformations, pass a `formula=` and `data=` so that `dydx()` can
re-evaluate derived terms when it perturbs a column.

The formula-built design matrix must match the columns the model was
trained on.  Here we fit without an intercept and suppress the intercept
in the formula so the column counts align:

```{code-cell} python
model_poly = LinearRegression(fit_intercept=False)
X_poly = pd.DataFrame({
    "age": df["age"],
    "female": df["female"],
    "treated": df["treated"],
    "age_sq": df["age_sq"],
})
model_poly.fit(X_poly, df["y"])

adapter_poly = SklearnBootstrapAdapter(
    model_poly,
    formula="y ~ 0 + age + female + treated + I(age**2)",
    data=df,
    target_name="y",
)
m_poly = GComputation(
    model_poly, adapter=adapter_poly,
    method="bootstrap", B=200, seed=42,
)
slope = m_poly.dydx("age")
print(slope.summary())
```

Without `formula=`, the adapter falls back to simple column selection.
If derived terms are present this raises a clear error rather than
silently returning an incorrect slope.

## Non-parametric estimators

The same pattern works for tree-based models.  Point estimates come from
the fitted model; uncertainty comes from resampling:

```{code-cell} python
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbr.fit(X, y)

adapter_gbr = SklearnBootstrapAdapter(gbr, X_train=X, y_train=y)
m_gbr = GComputation(gbr, adapter=adapter_gbr, method="bootstrap", B=100, seed=42)
print(m_gbr.predict(atexog={"treated": [0, 1]}).summary())
```

## Limitations

- **No delta / simulation** — the adapter returns dummy coefficients and
  covariance; all uncertainty is bootstrap-based.
- **No custom `vcov=`** — bootstrap does not use a parametric covariance
  matrix, so passing `vcov=` raises an error.
- **Formula verification** — when `formula=` is provided, the adapter
  verifies that the formula-built design matrix reproduces the model's
  predictions.  Mismatches (e.g. an unexpected intercept) raise
  `ValueError` with a diagnostic message.
- **Parallel bootstrap** — `n_jobs > 1` is supported via thread pools.
  Set `n_jobs=1` if the underlying sklearn estimator is not thread-safe.