# Inference scale (`phi` / `phi_inv`)

The session-level pair `(phi, phi_inv)` picks the *inference scale* —
the scale on which the delta method and the κ diagnostic are
computed, and from whose CI endpoints the reporting scale is
back-transformed.

Mathematically, every estimand is the triple `(h, phi, phi_inv)`:

- `h(β) → scalar or vector`, evaluated on the inference scale;
- `phi`, applied to CI endpoints to lift them to the reporting scale;
- `phi_inv`, used to convert user-supplied null values for tests onto
  the inference scale.

## Why is this a session commitment?

Two reasons.

**κ has a stable referent only when the scale is fixed.** The
curvature diagnostic measures how nonlinear the estimand is in `β`.
That number depends entirely on the scale. Fixing `(phi, phi_inv)` at
session level is what makes the session-level κ comparable across
calls.

**Inter-call composability requires a shared scale.** Two results
from the same session can be subtracted, added, scaled, and combined
through `MarginsResult.__sub__` etc., and the combined object
inherits the joint covariance. That arithmetic is only meaningful if
the operands live on the same inference scale.

## Picking a scale

The rule of thumb is that the contrast should be *as linear as
possible* in `β`. That keeps κ small and the symmetric Wald CI honest.

| Quantity                                  | Natural scale     | Helper                          |
|-------------------------------------------|-------------------|---------------------------------|
| additive effect on response               | identity          | `Margins.linear_scale`          |
| rate ratio, risk ratio, hazard ratio       | log               | `Margins.log_scale`             |
| odds ratio, probability contrast           | logit             | `Margins.logit_scale`           |
| RR − 1 (lift)                              | `expm1`           | `Margins.lift_scale`            |
| correlations                               | Fisher z (`tanh`) | `Margins.correlation_scale`     |

The constructor also accepts arbitrary `phi=` / `phi_inv=` JAX
callables for custom scales.

## What changes when you change scale?

- The point estimate is computed by `h(β̂)` on the inference scale,
  then back-transformed by `phi`. Different scale, different
  reported estimate.
- The CI is symmetric on the inference scale, asymmetric on the
  reporting scale.
- κ is recomputed per scale, so the fallback decision may change.
- The hypothesis test in `.test(null=...)` interprets `null` on the
  reporting scale and lifts it via `phi_inv`.

See [](kappa_diagnostic.md) for the curvature-fallback link.
