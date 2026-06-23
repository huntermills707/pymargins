# Inference scale (`phi` / `phi_inv`)
The estimator-level pair `(phi, phi_inv)` picks the *inference scale* —
the scale on which the delta method is computed, and from whose CI
endpoints the reporting scale is back-transformed.

Mathematically, every estimand is the triple `(h, phi, phi_inv)`:

- `h(β) → scalar or vector`, evaluated on the inference scale;
- `phi`, applied to CI endpoints to lift them to the reporting scale;
- `phi_inv`, used to convert user-supplied null values for tests onto
  the inference scale.

## Why is this an estimator commitment?

Two reasons.

**Comparability.** The curvature of an estimand in `β` depends on the
scale. Fixing `(phi, phi_inv)` at estimator level makes results on the
same scale comparable across calls.

**Inter-call composability requires a shared scale.** Two results
from the same estimator can be subtracted, added, scaled, and combined
through `GraphResult.__sub__` etc., and the combined object
inherits the joint covariance. That arithmetic is only meaningful if
the operands live on the same inference scale.

## Picking a scale

The rule of thumb is that the contrast should be *as linear as
possible* in `β`. That keeps κ small and the symmetric Wald CI honest.

| Quantity                                  | Natural scale     | Constructor keyword             |
|-------------------------------------------|-------------------|---------------------------------|
| additive effect on response               | identity          | `scale="identity"`              |
| rate ratio, risk ratio, hazard ratio       | log               | `scale="log"`                   |
| odds ratio, probability contrast           | logit             | `scale="logit"`                 |
| correlations                               | Fisher z (`tanh`) | `scale=(jnp.tanh, jnp.arctanh)` |

The constructor also accepts an explicit `(phi, phi_inv)` tuple for
custom scales.

## What changes when you change scale?

- The point estimate is computed by `h(β̂)` on the inference scale,
  then back-transformed by `phi`. Different scale, different
  reported estimate.
- The CI is symmetric on the inference scale, asymmetric on the
  reporting scale.
- The hypothesis test in `.test(value=...)` interprets `value` on the
  reporting scale and lifts it via `phi_inv`.