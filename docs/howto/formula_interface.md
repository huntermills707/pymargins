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

# Formula interface for array-fit models
statsmodels formula API (`smf.glm(...)`) handles interactions, polynomials,
and splines automatically — the fitted result carries a `design_info`
object that `pymargins` uses to rebuild the design matrix for
perturbations and counterfactuals.

Array-fit models (`sm.GLM(y, X)`) and linearmodels estimators do **not**
carry this metadata.  Without it, `dydx()` on a variable involved in an
interaction or polynomial silently produces the wrong slope because the
adapter cannot re-evaluate the derived term.

`pymargins` solves this by letting you pass the model formula as the
`outcome=` argument after `steps.input(data)`.  The formula is fit on the
input data and the resulting specification is frozen into the Plan.

## When do you need it?

| Model construction | Needs formula `outcome=`? |
|--------------------|---------------------------|
| `smf.glm("y ~ x1 * x2", data=df)` | No — native formula support |
| `sm.GLM(y, X)` where `X` contains `x1`, `x2`, `x1:x2` | **Yes** |
| `sm.OLS(y, X)` with `I(age**2)` in `X` | **Yes** |
| `linearmodels.PanelOLS.from_formula(...)` | No — native formula support |
| `linearmodels.PanelOLS(..., exog=X)` with derived terms | **Yes** |
| Any sklearn estimator with derived terms | **Yes** |

## Passing a formula as `outcome=`

```python
import statsmodels.api as sm
from pymargins import GComputation, steps  # 0.4.0: Margins -> GComputation

# Array-fit OLS with a quadratic
X = pd.DataFrame({
    "Intercept": 1.0,
    "age": df["age"],
    "treat": df["treat"],
    "I(age ** 2)": df["age"] ** 2,
})
fit = sm.OLS(df["y"], X).fit()

# Pass the formula so dydx() knows age appears inside I(age**2)
m = GComputation(
    steps.input(df),
    outcome="y ~ age + treat + I(age**2)",
    scale="identity",
)
print(m.dydx("age").summary())
```

The `outcome=` string must exactly match the model specification,
including the intercept.  If it does not, `pymargins` raises a
verification error showing the max difference in linear predictor.

## Stateful transforms

The formula step freezes transform parameters from the training data, so
counterfactuals and perturbations use the *training-time* state rather
than recomputing it on the new data:

```python
# Centering: the new data is centered by the training mean, not its own mean
m = GComputation(
    steps.input(df),
    outcome="y ~ center(age) + treat",
    scale="identity",
)
```

## Categorical levels

The full training level set is preserved, even when the evaluation data
lacks some levels:

```python
m = GComputation(
    steps.input(df),   # df has regions North, South, East, West
    outcome="y ~ C(region) + age",
    scale="identity",
)
# Prediction on a subset that only contains North and South still
# produces the same design matrix columns (West dummy is all zeros).
print(m.predict(atexog={"region": ["North", "South"]}).summary())
```

## What happens without a formula?

If the adapter cannot recover derived terms from the fitted model, it
falls back to column selection (`df.reindex(columns=exog_names)`).  When
derived terms are detected in the column names, `pymargins` emits a
warning:

```
UserWarning: column-selection fallback cannot reproduce derived terms ...
```

If no derived terms are present, the fallback is exact and no warning is
raised.

For sklearn models without a formula, the adapter **raises** an error
when derived terms are present, because silently incorrect slopes are
especially dangerous for black-box models where the user cannot easily
audit the design matrix by hand.
