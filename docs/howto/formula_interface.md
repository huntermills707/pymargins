# Formula interface for array-fit models
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

statsmodels formula API (`smf.glm(...)`) handles interactions, polynomials,
and splines automatically — the fitted result carries a `design_info`
object that `pymargins` uses to rebuild the design matrix for
perturbations and counterfactuals.

Array-fit models (`sm.GLM(y, X)`) and linearmodels estimators do **not**
carry this metadata.  Without it, `dydx()` on a variable involved in an
interaction or polynomial silently produces the wrong slope because the
adapter cannot re-evaluate the derived term.

`pymargins` solves this with an explicit `formula=` (and `data=`)
argument that builds a frozen `FormulaSpec` at session construction.

## When do you need it?

| Model construction | Needs `formula=`? |
|--------------------|-------------------|
| `smf.glm("y ~ x1 * x2", data=df)` | No — native formula support |
| `sm.GLM(y, X)` where `X` contains `x1`, `x2`, `x1:x2` | **Yes** |
| `sm.OLS(y, X)` with `I(age**2)` in `X` | **Yes** |
| `linearmodels.PanelOLS.from_formula(...)` | No — native formula support |
| `linearmodels.PanelOLS(..., exog=X)` with derived terms | **Yes** |
| Any sklearn estimator with derived terms | **Yes** |

## Passing `formula=` to `Margins`

```python
import statsmodels.api as sm
from pymargins import GComputation  # 0.4.0: Margins -> GComputation

# Array-fit OLS with a quadratic
X = pd.DataFrame({
    "Intercept": 1.0,
    "age": df["age"],
    "treat": df["treat"],
    "I(age ** 2)": df["age"] ** 2,
})
fit = sm.OLS(df["y"], X).fit()

# Pass the formula so dydx() knows age appears inside I(age**2)
m = Margins.linear_scale(
    fit,
    formula="y ~ age + treat + I(age**2)",
    data=df,
)
print(m.dydx("age").summary())
```

The `formula=` string must exactly match the model specification,
including the intercept.  If it does not, `pymargins` raises a
verification error showing the max difference in linear predictor.

## `Margins.from_formula` shorthand

For convenience, there is a classmethod that forwards `formula=` and
`data=` directly:

```python
m = Margins.from_formula(
    fit,
    formula="y ~ age + treat + I(age**2)",
    data=df,
)
```

## Stateful transforms

`FormulaSpec` freezes transform parameters from the training data, so
counterfactuals and perturbations use the *training-time* state rather
than recomputing it on the new data:

```python
# Centering: the new data is centered by the training mean, not its own mean
m = Margins.linear_scale(
    fit,
    formula="y ~ center(age) + treat",
    data=df,
)
```

## Categorical levels

The full training level set is preserved, even when the evaluation data
lacks some levels:

```python
m = Margins.linear_scale(
    fit,
    formula="y ~ C(region) + age",
    data=df,   # df has regions North, South, East, West
)
# Prediction on a subset that only contains North and South still
# produces the same design matrix columns (West dummy is all zeros).
print(m.predict(atexog={"region": ["North", "South"]}).summary())
```

## What happens without `formula=`?

If you pass `data=` but not `formula=`, the adapter falls back to column
selection (`df.reindex(columns=exog_names)`).  When derived terms are
detected in the column names, `pymargins` emits a warning:

```
UserWarning: column-selection fallback cannot reproduce derived terms ...
```

If no derived terms are present, the fallback is exact and no warning is
raised.

For sklearn models without `formula=`, the adapter **raises** an error
when derived terms are present, because silently incorrect slopes are
especially dangerous for black-box models where the user cannot easily
audit the design matrix by hand.