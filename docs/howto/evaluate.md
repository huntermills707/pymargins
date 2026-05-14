# Nonlinear estimands with `evaluate`

`Margins.evaluate` is the escape hatch for estimands that cannot be written
as a weighted sum of scenario predictions.  Use it for reciprocals,
custom utility functions, ratios of differences, or any other
JAX-differentiable composition that is not linear in the predictions.

For simple differences, risk ratios, odds ratios, and
difference-in-differences, `contrasts` is the better tool — it is
faster, more transparent, and usually more accurate.  See
[](../howto/contrasts.md) for those recipes and
[](../howto/contrasts_vs_evaluate.md) for the full decision guide.

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

## Number needed to treat (NNT)

NNT is the reciprocal of the absolute risk reduction.  Because it is a
reciprocal, it cannot be written as a linear contrast and must go
through `evaluate`:

```python
from pymargins import Margins

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

## Raw ratio on the linear scale

A risk ratio is usually computed with `contrasts` on a `log_scale`
session (`log(p₁) − log(p₀)` back-transformed with `exp`).  That is the
preferred path because the log-ratio is linear and the delta method is
exact.

Use `evaluate` for the raw ratio `p₁ / p₀` only when your field or
journal explicitly requires inference on the ratio scale itself:

```python
m = Margins.linear_scale(fit, at="overall")

scenarios = [
    {"atexog": {"treatment": 1}, "label": "treated"},
    {"atexog": {"treatment": 0}, "label": "control"},
]

res = m.evaluate(
    scenarios=scenarios,
    compose=lambda p: p[0] / p[1],
)
res.summary()
```

Because the ratio is nonlinear on the linear scale, κ is usually larger
than for the log-scale contrast and the CI is wider.

## Ratio of differences (Emax-style)

When the estimand is a ratio in which the numerator and denominator are
themselves differences, `evaluate` is required:

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

## Custom utility / welfare function

Suppose you have a utility function `u(p) = p**0.5` (a concave
transformation of a predicted probability) and you want the
expected utility difference between two policy regimes:

```python
import jax.numpy as jnp

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

- [](../howto/contrasts.md) for risk differences, log-ratios, odds
  ratios, and DiD.
- [](../howto/contrasts_vs_evaluate.md) for the decision flowchart.
- [](../howto/scenarios_helpers.md) for `pairwise`, `reference`, `grid`,
  etc.
- [](../math.rst) for the delta-method derivation.
