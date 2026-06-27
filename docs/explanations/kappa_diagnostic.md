# The κ curvature diagnostic

The delta method is a first-order Taylor approximation. The
approximation breaks when the estimand is too curved in `β` for the
local linearization to track the true sampling distribution. The κ
diagnostic measures that curvature; when it is large, switch to
`method="simulation"` or `method="bootstrap"`.

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
| κ ≥ 0.3      | delta unsafe; use simulation or bootstrap   |

Every `GraphResult` carries `result.kappa` — the worst-case curvature of
the estimand on the inference scale. Under `method="auto"`, κ is computed
at compile and used once to choose between delta and simulation; under an
explicit `method=`, κ is still recorded for transparency but does not
change the method.

See [](inference_scale.md) for why picking the right scale is the
first move before judging delta-method reliability.
