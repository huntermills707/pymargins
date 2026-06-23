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

**Composition operates on a shared scale.** Within an estimator,
`contrasts()` and `evaluate()` combine several scenario estimands — a
linear combination, or a custom `compose` — on the inference scale
before back-transforming to the reporting scale, and `scaled()` rescales
a single result on that scale. Because `(phi, phi_inv)` is fixed at
estimator level, every estimand in such a composition is guaranteed to
share it, so the combination — and the joint covariance it carries — is
well defined. That is only meaningful when all operands live on the same
inference scale. (Composing results from *separate* estimators is
deferred to a future release; see `GComputation.joint`.)

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