# The κ curvature diagnostic

The delta method is a first-order Taylor approximation. The
approximation breaks when the estimand is too curved in `β` for the
local linearization to track the true sampling distribution. The κ
diagnostic measures that curvature and, when it crosses a threshold,
auto-falls-back to simulation.

## Definition

For an estimand `h(β)` with gradient `g = ∂h/∂β |_{β̂}` and Hessian
`H = ∂²h/∂β² |_{β̂}`, define the whitened gradient and Hessian

$$
\tilde g = L^\top g, \qquad \tilde H = L^\top H L,
$$

where `L L^⊤ = V̂` is the Cholesky factor of the parameter
covariance. Skovgaard's relative curvature in this metric is

$$
\kappa = \frac{\lVert \tilde H \rVert}{\lVert \tilde g \rVert^2}.
$$

The whitening transform is critical: without it κ is
*parameterization-dependent* and uninterpretable. With it, κ has the
property that an affine change of variables leaves it invariant.

## Calibration

The default thresholds are taken from the nonlinear-regression
literature:

| κ value      | Interpretation                              |
|--------------|---------------------------------------------|
| κ < 0.1      | delta method is highly reliable             |
| 0.1 ≤ κ < 0.3 | borderline; delta usable but report κ      |
| κ ≥ 0.3      | delta unsafe; auto-fall-back to simulation  |

These are configurable via `kappa_threshold=` on the session.

## What the fallback does

When κ exceeds the threshold and the session was constructed with
`method="delta"`, the call recomputes inference via Krinsky–Robb
simulation. The result records:

- the realized inference method (`"simulation"`, not the requested
  `"delta"`);
- the fallback reason (`"kappa exceeded"`);
- the κ value itself, surfaced on `result.summary()`.

This is meant to be loud. `pymargins`' position is that silent
delta-method use on highly curved estimands is the most common
inference bug in published applied work, and that a tool that just
computes the delta number — as Stata's `margins` and R's
`marginaleffects` both do — gives the analyst no way to know they
should have used something else.

## Disabling

- `kappa_threshold=float("inf")` — keep the requested method, never
  fall back.
- `diagnostics=False` — skip κ entirely (useful in tight loops where
  the second derivative is expensive).

See [](inference_scale.md) for why picking the right scale is the
first move before tightening κ tolerance.
