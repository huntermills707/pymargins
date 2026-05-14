# Nonlinear estimands with `evaluate`

`Margins.evaluate` is the escape hatch for estimands that cannot be written
as a weighted sum of scenario predictions.  Use it for ratios, numbers
needed to treat (NNT), custom utility functions, or any other
JAX-differentiable composition.

For linear combinations (risk differences, log-rate ratios, etc.) prefer
`contrasts` — it is clearer, uses a faster code path, and the weight
vector is explicitly visible in the audit trail.

## How `evaluate` works

`evaluate` takes a list of **scenarios** and a **compose** function.
The engine:

1. Computes the response-scale prediction for each scenario.
2. Stacks them into a JAX array and passes them to `compose`.
3. Applies `phi_inv` to lift the result onto the inference scale.
4. Runs delta-method inference (or simulation/bootstrap if curvature is
   high or `compose` is not JAX-differentiable).
5. Back-transforms CI endpoints with `phi` for reporting.

Mathematically:

```
result = φ( φ⁻¹( compose(p₁, p₂, …, p_k) ) )
```

where `pᵢ` is the aggregated response-scale prediction for scenario `i`.

## Ratio of two predictions

A ratio contrast (e.g. `p_treated / p_control`) is nonlinear, so it
must go through `evaluate`:

```python
from pymargins import Margins
import jax.numpy as jnp

m = Margins.log_scale(fit, at="overall")

scenarios = [
    {"atexog": {"treatment": 1}, "label": "treated"},
    {"atexog": {"treatment": 0}, "label": "control"},
]

# On a log-scale session the ratio is a difference on the inference
# scale, so the delta method is exact for log(ratio).  The reported
# point estimate is back-transformed to the raw ratio.
res = m.evaluate(
    scenarios=scenarios,
    compose=lambda p: p[0] / p[1],   # treated / control
)
res.summary()
```

## Number needed to treat (NNT)

NNT is the reciprocal of the absolute risk reduction:

```python
m = Margins.linear_scale(fit, at="overall")

scenarios = [
    {"atexog": {"treatment": 1}, "label": "treated"},
    {"atexog": {"treatment": 0}, "label": "control"},
]

res = m.evaluate(
    scenarios=scenarios,
    compose=lambda p: 1.0 / (p[0] - p[1]),
)
res.summary()
```

If the risk difference crosses zero, the denominator can change sign
and κ will be large.  In that case `pymargins` auto-falls back to
simulation, which is the safe thing to do for a reciprocal.

## Custom utility / welfare function

Suppose you have a utility function `u(p) = p**0.5` (a concave
transformation of a predicted probability) and you want the
expected utility difference between two policy regimes:

```python
m = Margins.linear_scale(fit, at="overall")

scenarios = [
    {"atexog": {"policy": "A"}, "label": "regime_A"},
    {"atexog": {"policy": "B"}, "label": "regime_B"},
]

res = m.evaluate(
    scenarios=scenarios,
    compose=lambda p: jnp.sqrt(p[0]) - jnp.sqrt(p[1]),
)
res.summary()
```

## Multi-scenario estimands

`compose` receives the predictions in the order the scenarios were
provided.  You can use any number of scenarios:

```python
scenarios = [
    {"atexog": {"dose": 0}, "label": "placebo"},
    {"atexog": {"dose": 50}, "label": "low"},
    {"atexog": {"dose": 100}, "label": "high"},
]

# Emax-style parameter: (high − placebo) / (low − placebo)
res = m.evaluate(
    scenarios=scenarios,
    compose=lambda p: (p[2] - p[0]) / (p[1] - p[0]),
)
```

## When `evaluate` auto-routes to simulation

If `compose` uses Python control flow (`if`, `for`) on tracer values,
JAX cannot differentiate it and the delta method is impossible.
`pymargins` catches the error and silently reroutes to the session's
simulation or bootstrap method.  The result records the realized
method, so the audit trail is still complete.

To avoid auto-routing, write `compose` with JAX primitives:

| Instead of             | Use                              |
|------------------------|----------------------------------|
| `if x > 0: ...`        | `jnp.where(x > 0, ..., ...)`     |
| `x / y` (unsafe divide)| `jnp.where(y == 0, 0, x / y)`    |
| `max(a, b)`            | `jnp.maximum(a, b)`              |

## See also

- [](scenarios_helpers.md) for `pairwise`, `reference`, `grid`, etc.
- [](contrasts_and_did.md) for linear contrast examples.
- [](math.rst) for the delta-method derivation.
