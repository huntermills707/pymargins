# Writing a custom adapter
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

Most users will not need this — `pymargins` ships adapters for
statsmodels, linearmodels, and lifelines. If you have a model class
none of those cover, the adapter interface is small.

The four adapter base classes, in increasing order of work:

| Base class                 | When to use                                     |
|----------------------------|-------------------------------------------------|
| `LinearPredictionAdapter`  | `μ = X β` exactly (OLS-like)                    |
| `GLMAdapter`               | `μ = f(X β)` with an analytic `f'`              |
| `WrappedFDAdapter`         | black-box predict, but `η = X β` is accessible  |
| `BootstrapOnlyAdapter`     | refit-and-resample is the only viable path      |

## Minimal `GLMAdapter` example

A `GLMAdapter` subclass must implement four methods:

- `detect(model)` — return `True` if this adapter handles the fitted object.
- `variable_info(model)` — return a list of `VariableInfo` objects describing
covariates (names and whether they are `"continuous"` or `"categorical"`).
- `design_matrix(model, data)` — build the design matrix from a
`pd.DataFrame`.
- `link_inverse(eta)` and `link_inverse_deriv(eta)` — the mean function
and its derivative, written with `jax.numpy` operations.

```python
from pymargins import GLMAdapter, VariableInfo, register_adapter
import jax.numpy as jnp

class MyGLMAdapter(GLMAdapter):
    def detect(self, model):
        return isinstance(model, MyModel)

    def variable_info(self, model):
        return [VariableInfo(name=n, kind="continuous") for n in model.feature_names_]

    def design_matrix(self, model, data):
        return model.build_design(data)

    def link_inverse(self, eta):
        return jnp.exp(eta) / (1 + jnp.exp(eta))

    def link_inverse_deriv(self, eta):
        p = self.link_inverse(eta)
        return p * (1 - p)

register_adapter(MyGLMAdapter())
```

After registration, `Margins(...)` will auto-detect your model class.
Adapters are tried in registration order; user-registered adapters are
tried *before* built-ins, so you can override default detection if your
fitted object subclasses one of the supported types but has different
`predict` semantics.

## `WrappedFDAdapter` — when you have a black-box predict

If your model already has a `predict(data)` method but you cannot
reimplement it in JAX, subclass `WrappedFDAdapter`.  You only need to
expose the linear predictor `η = X β`:

```python
from pymargins import WrappedFDAdapter, VariableInfo, register_adapter

class MyWrappedAdapter(WrappedFDAdapter):
    def detect(self, model):
        return isinstance(model, MyBlackBoxModel)

    def variable_info(self, model):
        return [VariableInfo(name=n, kind="continuous") for n in model.cols]

    def design_matrix(self, model, data):
        return data[model.cols].values

    def predict(self, model, data):
        # Black-box: can use numpy, sklearn, etc.
        return model.predict(data)

register_adapter(MyWrappedAdapter())
```

The engine wraps `predict` in a custom-JVP finite-difference primitive
and differentiates through it.  Gradient quality is good; Hessian
quality is acceptable for κ.  If you need exact Hessians, consider
reimplementing the predict path in JAX and upgrading to `GLMAdapter`.

## Testing your adapter

Before registering, test the adapter standalone:

```python
adapter = MyGLMAdapter()
assert adapter.detect(my_fitted_model)

X = adapter.design_matrix(my_fitted_model, df[:5])
assert X.shape == (5, len(adapter.variable_info(my_fitted_model)))

# Prediction round-trip
beta = adapter.coefficients()
preds = adapter.predict(beta, X)
assert preds.shape == (5,)
```

See [](../explanations/adapter_pattern.md) for the full contract and
the rationale behind the four-base-class split.