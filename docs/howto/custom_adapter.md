# Writing a custom adapter

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

See [](../explanations/adapter_pattern.md) for the full contract and
the rationale behind the four-base-class split.
