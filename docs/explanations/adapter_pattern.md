# The adapter pattern

`pymargins` separates "what does the model do" from "what statistic
do I want". The adapter is the seam: it answers, for one model
class, the four questions the inference engine needs.

## The four questions

1. **Detect** — does this adapter handle the fitted result?
2. **Predict** — given `β` and a design matrix, produce the linear
   predictor (and, for GLMs, the response).
3. **Differentiate** — what gradient backend should the engine use?
4. **Vary** — what counterfactual values are valid for each variable?

The answers live on a `ModelAdapter` subclass; auto-detection
iterates the registered adapters in order and uses the first one
whose `detect()` returns true.

## Four base classes by gradient backend

| Base                       | Gradient backend           | Use when                                       |
|----------------------------|----------------------------|------------------------------------------------|
| `LinearPredictionAdapter`  | `autodiff` (identity link) | `μ = X β` exactly (OLS, WLS, GLS, IV)          |
| `GLMAdapter`               | `autodiff` via `f'(η)`     | `μ = f(X β)` with analytic link derivative    |
| `WrappedFDAdapter`         | `wrapped_fd`               | black-box predict, but `η = X β` accessible    |
| `BootstrapOnlyAdapter`     | `fd` + bootstrap-only      | refit-and-resample is the only viable path     |

The class hierarchy reflects a tradeoff between adapter author effort
and downstream gradient quality. Pick the most specialized base your
model admits.

## What's included

Available from `pymargins.adapters` (auto-detected by `Margins(model)`;
import explicitly only to override detection or select a non-default
scale):

- **statsmodels**: GLM, OLS, MNLogit, ordered, discrete binary /
  count, Poisson, GEE (gaussian / nominal / ordinal), MixedLM,
  PHReg, QuantReg, RLM, zero-inflated.
- **linearmodels**: IV (2SLS / LIML / GMM), panel (FE / RE / BE),
  AbsorbingLS, FamaMacBeth.
- **lifelines**: CoxPH, CoxTimeVarying, AAF, CRC splines, Weibull
  AFT, LogLogistic AFT, LogNormal AFT, GeneralizedGamma AFT,
  PiecewiseExponential.

## Registration

```python
from pymargins import register_adapter

register_adapter(MyAdapter())
```

Adapters are tried in registration order; user-registered adapters
are tried *before* the built-ins, so you can override default
detection if your fitted object subclasses one of the supported
types but has different `predict` semantics.

See [](../howto/custom_adapter.md) for a step-by-step example.
