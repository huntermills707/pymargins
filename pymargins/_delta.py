"""
pymargins._delta

Apply the delta method to compute variance, confidence intervals, and Wald
tests of estimands given gradients and Σ̂.

The delta method approximates the sampling variance of g(β̂) for any smooth
g via a first-order Taylor expansion:

    Var(g(β̂)) ≈ ∇g(β̂)^T Σ̂ ∇g(β̂)

This module is the numerical kernel for delta-method inference. It is purely
functional: every function takes gradients and a covariance matrix and
returns a numerical result. No knowledge of estimands, sessions, scenarios,
or back-transformations beyond the optional φ argument applied to CI endpoints.

Validity caveats — handled outside this module
----------------------------------------------
The delta method's accuracy depends on:
  1. Smoothness of g near β̂ (curvature; assess via _kappa.kappa)
  2. Approximate normality of β̂ (assess via the model fit)
  3. Quality of Σ̂ (assess via the model framework's vcov machinery)

This module does the math given those inputs. Diagnostics live in _kappa.
"""

from __future__ import annotations
from typing import Callable, Optional, Literal
import jax.numpy as jnp
import numpy as np
from scipy import stats
import warnings


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Alternative = Literal["two-sided", "greater", "less"]


# ---------------------------------------------------------------------------
# Variance and SE
# ---------------------------------------------------------------------------

def delta_variance(
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Var(g(β̂)) ≈ ∇g^T Σ̂ ∇g.

    For scalar g (gradient is a vector), returns a scalar variance.
    For vector g (gradient is a Jacobian), returns the full covariance
    matrix Σ_g of shape (n_outputs, n_outputs); use this for joint inference
    across multiple estimand components.

    Parameters
    ----------
    grad : jax array
        Gradient or Jacobian of the estimand at β̂. Shape (n_params,) for
        scalar estimands or (n_outputs, n_params) for vector estimands.

    cov_params : jax array of shape (n_params, n_params)
        Σ̂, the estimated covariance of β̂. Symmetric positive semi-definite.

    Returns
    -------
    var : jax array
        Variance (scalar) or covariance matrix (n_outputs × n_outputs) of
        the estimand on the inference scale.
    """
    if grad.ndim == 1:
        return grad @ cov_params @ grad
    elif grad.ndim == 2:
        return grad @ cov_params @ grad.T
    else:
        raise ValueError(
            f"gradient must be 1D or 2D; got ndim={grad.ndim}"
        )


def _safe_sqrt_diag(var: jnp.ndarray) -> jnp.ndarray:
    """Return sqrt(diag(var)) with negative entries clipped to 0.

    Numerical noise (e.g., from an ill-conditioned Σ̂) can produce tiny
    negative diagonal entries. Clipping avoids NaN SEs.
    """
    diag = jnp.diag(var)
    return jnp.sqrt(jnp.maximum(diag, 0.0))


def delta_se(
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
) -> jnp.ndarray:
    """Standard error(s) on the inference scale.

    For scalar g, returns the scalar SE = sqrt(Var). For vector g, returns
    the diagonal SEs as an array; the full covariance is available via
    delta_variance.

    SEs are reported on the inference scale, not the reporting scale. A
    log-scale session reports SEs in log units, not in the back-transformed
    units. This is the only statistically meaningful SE for the analysis;
    "SEs on the reporting scale" generally don't have a clean interpretation
    when φ is nonlinear (the reporting-scale CI is asymmetric, so SE in
    those units would mislead).

    Parameters
    ----------
    grad : jax array
        Gradient or Jacobian. See delta_variance.

    cov_params : jax array
        Σ̂.

    Returns
    -------
    se : jax array
        Scalar SE for scalar g, or 1D array of SEs (one per output
        component) for vector g.
    """
    var = delta_variance(grad, cov_params)
    if jnp.ndim(var) == 0:
        return jnp.sqrt(jnp.maximum(var, 0.0))
    else:
        return _safe_sqrt_diag(var)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def delta_confint(
    estimate: jnp.ndarray,
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
    *,
    level: float = 0.95,
    phi: Optional[Callable] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Confidence interval(s) via the delta method.

    Constructs a symmetric CI on the inference scale, then optionally applies
    a back-transform φ to the endpoints. The back-transform produces
    properly asymmetric CIs on the reporting scale by quantile equivariance
    (no Jacobian correction needed; only valid for monotone φ, which all
    standard back-transforms are).

    Parameters
    ----------
    estimate : jax array
        Point estimate g(β̂) on the inference scale.

    grad : jax array
        ∇g(β̂).

    cov_params : jax array
        Σ̂.

    level : float, default 0.95
        Confidence level; constructs a CI containing the true parameter
        with this probability under the model's asymptotic assumptions.

    phi : callable, optional
        Back-transform from inference scale to reporting scale. Applied to
        the symmetric inference-scale CI endpoints. Must be monotone for
        the CI's coverage probability to be preserved. If None, returns
        the CI on the inference scale.

    Returns
    -------
    (lower, upper) : tuple of jax arrays
        CI bounds on the reporting scale (or inference scale if phi is None).
        Shapes match `estimate`.

    Notes
    -----
    Uses ``scipy.stats.norm.ppf``, so this function is not jittable/vmappable.
    If pure-JAX composition is needed, pass precomputed z quantiles.
    """
    if not (0 < level < 1):
        raise ValueError(f"level must be in (0,1), got {level}")
    se = delta_se(grad, cov_params)
    z = stats.norm.ppf(0.5 + level / 2.0)

    lower_inf = estimate - z * se
    upper_inf = estimate + z * se

    if phi is None:
        return lower_inf, upper_inf
    else:
        return phi(lower_inf), phi(upper_inf)


def delta_confint_from_se(
    estimate: jnp.ndarray,
    se: jnp.ndarray,
    *,
    level: float = 0.95,
    phi: Optional[Callable] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a CI directly from a precomputed SE.

    Useful when the caller has already computed delta_se separately (e.g.,
    in a result object) and wants to recompute the CI at a different
    confidence level without redoing the gradient/variance work.

    Parameters
    ----------
    estimate : jax array
        Point estimate on the inference scale.

    se : jax array
        Standard error on the inference scale.

    level : float, default 0.95
        Confidence level.

    phi : callable, optional
        Back-transform.

    Returns
    -------
    (lower, upper) : tuple of jax arrays
    """
    if not (0 < level < 1):
        raise ValueError(f"level must be in (0,1), got {level}")
    z = stats.norm.ppf(0.5 + level / 2.0)
    lower_inf = estimate - z * se
    upper_inf = estimate + z * se
    if phi is None:
        return lower_inf, upper_inf
    return phi(lower_inf), phi(upper_inf)


# ---------------------------------------------------------------------------
# Wald tests
# ---------------------------------------------------------------------------

def delta_wald_test(
    estimate: jnp.ndarray,
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
    null_value: jnp.ndarray = 0.0,
    *,
    alternative: Alternative = "two-sided",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-component Wald test of H₀: g(β) = null_value.

    For scalar g, returns a scalar (z, p) pair. For vector g, returns
    component-wise (z, p) arrays — each component is tested against its
    corresponding null value independently. For a joint hypothesis about
    the entire vector, use joint_wald_test.

    The null value should be specified on the **inference scale**, not the
    reporting scale. For example, with a log-scale session and a
    null hypothesis of "no effect" (RR = 1), pass null_value=0.0 (since
    log(1) = 0), not null_value=1.0. Sessions providing convenience methods
    should accept reporting-scale nulls and convert via phi_inv.

    Parameters
    ----------
    estimate : jax array
        Point estimate(s) on the inference scale.

    grad : jax array
        Gradient or Jacobian.

    cov_params : jax array
        Σ̂.

    null_value : jax array or scalar, default 0.0
        Hypothesized value(s) on the inference scale.

    alternative : str, default "two-sided"
        Direction of the test.

    Returns
    -------
    (z, p) : tuple of jax arrays
        z-statistic(s) and p-value(s).

    Notes
    -----
    Mixes JAX and NumPy/SciPy stats, so not jittable/vmappable.
    This is intentional — p-values are reporting-layer quantities.
    """
    se = delta_se(grad, cov_params)
    # Guard against division by zero: if SE is effectively 0, the estimand is
    # a deterministic function of beta at this point (e.g., a prediction at
    # the exact fitted value). The z-statistic is +inf if estimate > null,
    # -inf if estimate < null, and 0 if they are equal.
    z = jnp.where(
        se < 1e-15,
        jnp.where(estimate == null_value, 0.0, jnp.sign(estimate - null_value) * jnp.inf),
        (estimate - null_value) / se,
    )

    z_np = np.asarray(z)
    if alternative == "two-sided":
        p = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_np)))
    elif alternative == "greater":
        p = 1.0 - stats.norm.cdf(z_np)
    elif alternative == "less":
        p = stats.norm.cdf(z_np)
    else:
        raise ValueError(f"Unknown alternative: {alternative!r}")

    return z, jnp.asarray(p)


def joint_wald_test(
    estimate: jnp.ndarray,
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
    null_value: Optional[jnp.ndarray] = None,
) -> tuple[float, float, int]:
    """Joint Wald test of H₀: g(β) = null_value (full vector equality).

    Tests whether the entire vector of estimands equals a specified null
    vector simultaneously, accounting for their joint covariance. The test
    statistic is

        χ² = (g(β̂) - null)^T Σ_g⁻¹ (g(β̂) - null)

    distributed as χ² with df = length(estimate) under the null.

    Parameters
    ----------
    estimate : jax array of shape (k,)
        Vector of point estimates on the inference scale.

    grad : jax array of shape (k, n_params)
        Jacobian of g.

    cov_params : jax array of shape (n_params, n_params)
        Σ̂.

    null_value : jax array of shape (k,), optional
        Hypothesized vector. Defaults to zero vector (joint H₀ that all
        estimands are zero on the inference scale).

    Returns
    -------
    (chi2, p, df) : tuple
        Test statistic, p-value, degrees of freedom.
    """
    estimate = jnp.atleast_1d(estimate)
    if null_value is None:
        null_value = jnp.zeros_like(estimate)

    diff = estimate - null_value
    Sigma_g = delta_variance(grad, cov_params)

    # Solve rather than invert for numerical stability.
    # If Sigma_g is singular (e.g., perfectly collinear estimands),
    # add a tiny ridge and retry once.
    solved = jnp.linalg.solve(Sigma_g, diff)
    chi2 = float(diff @ solved)
    regularized = False
    if not np.isfinite(chi2):
        ridge = 1e-12 * float(jnp.trace(Sigma_g)) / Sigma_g.shape[0]
        ridge = max(ridge, float(jnp.finfo(Sigma_g.dtype).eps))
        Sigma_g_reg = Sigma_g + ridge * jnp.eye(Sigma_g.shape[0])
        chi2 = float(diff @ jnp.linalg.solve(Sigma_g_reg, diff))
        regularized = True
        warnings.warn(
            f"joint_wald_test: Σ_g was singular; added ridge={ridge:.3e} "
            "to compute the test statistic. Result is regularized.",
            RuntimeWarning,
            stacklevel=2,
        )

    df = int(diff.shape[0])
    p = float(1.0 - stats.chi2.cdf(chi2, df))

    return float(chi2), float(p), int(df)


# ---------------------------------------------------------------------------
# Joint inference for composed results
# ---------------------------------------------------------------------------

def combined_gradient(
    grads: list[jnp.ndarray],
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Combine gradients of multiple estimands via a linear combination.

    Given gradients ∇g₁, ..., ∇gₙ and weights w₁, ..., wₙ, returns the
    gradient of the combined estimand Σᵢ wᵢ gᵢ, which is Σᵢ wᵢ ∇gᵢ. Used
    by `linear_combination()` to compute joint inference across scenarios
    sharing a covariance.

    For scenario predictions stacked into a matrix of gradients (one row
    per scenario), the linear combination has gradient `weights @ grads`.

    Parameters
    ----------
    grads : list of jax arrays
        Gradient vectors, all of shape (n_params,).

    weights : jax array of shape (len(grads),)
        Linear combination weights.

    Returns
    -------
    grad_combined : jax array of shape (n_params,)
        ∇(Σᵢ wᵢ gᵢ).
    """
    if not grads:
        raise ValueError("grads must be a non-empty list")
    expected_shape = grads[0].shape
    for i, g in enumerate(grads):
        if g.shape != expected_shape:
            raise ValueError(
                f"All gradients must have the same shape; got {g.shape} at index {i}, "
                f"expected {expected_shape}"
            )
    grad_stack = jnp.stack(grads, axis=0)
    return weights @ grad_stack


def joint_covariance_of_results(
    grads: list[jnp.ndarray],
    cov_params: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the joint covariance of multiple estimands from a shared β̂.

    For scalar estimands g₁, ..., gₙ all derived from the same fitted model,
    their joint covariance under the delta method is

        Cov(gᵢ, gⱼ) = ∇gᵢ^T Σ̂ ∇gⱼ

    This function returns the full n × n joint covariance matrix, used by
    inter-call composability operators on MarginsResult to produce correct
    joint inference (rather than treating the estimands as independent).

    Parameters
    ----------
    grads : list of jax arrays
        Per-estimand gradients, all of shape (n_params,).

    cov_params : jax array of shape (n_params, n_params)
        Σ̂.

    Returns
    -------
    cov_joint : jax array of shape (n, n)
        Joint covariance of the estimands. Diagonal entries are individual
        delta variances; off-diagonal entries are cross-covariances.
    """
    if not grads:
        raise ValueError("grads must be a non-empty list")
    expected_shape = grads[0].shape
    for i, g in enumerate(grads):
        if g.shape != expected_shape:
            raise ValueError(
                f"All gradients must have the same shape; got {g.shape} at index {i}, "
                f"expected {expected_shape}"
            )
    G = jnp.stack(grads, axis=0)  # shape (n, n_params)
    return G @ cov_params @ G.T


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: SE and CI for a scalar estimand
------------------------------------------

    import jax.numpy as jnp
    from pymargins._gradients import gradient
    from pymargins._delta import delta_se, delta_confint

    def h(beta):
        return jax.scipy.special.expit(x_target @ beta)

    beta_hat = ...        # from fitted model
    Sigma_hat = ...       # from results.cov_params()

    grad = gradient(h, beta_hat)
    estimate = h(beta_hat)
    se = delta_se(grad, Sigma_hat)
    lower, upper = delta_confint(estimate, grad, Sigma_hat, level=0.95)


Example 2: CI on the reporting scale via phi
--------------------------------------------

    # For a log-scale session computing a relative risk
    def h(beta):
        # h = log RR = log(p1) - log(p0) on the inference scale
        p1 = jax.scipy.special.expit(x1 @ beta)
        p0 = jax.scipy.special.expit(x0 @ beta)
        return jnp.log(p1) - jnp.log(p0)

    grad = gradient(h, beta_hat)
    estimate = h(beta_hat)

    # CI on log-RR scale, then exponentiated for reporting as RR
    lower_rr, upper_rr = delta_confint(
        estimate, grad, Sigma_hat,
        level=0.95,
        phi=jnp.exp,
    )
    rr = jnp.exp(estimate)
    # Asymmetric CI on RR with proper coverage


Example 3: Wald test on the inference scale
-------------------------------------------

    from pymargins._delta import delta_wald_test

    # Test H0: log RR = 0 (i.e., RR = 1, no effect)
    z, p = delta_wald_test(estimate, grad, Sigma_hat, null_value=0.0)


Example 4: Joint test of multiple contrasts
-------------------------------------------

    from pymargins._delta import joint_wald_test

    # Vector estimand: 3 contrasts simultaneously
    def h_vec(beta):
        return jnp.array([h_did(beta), h_treatment(beta), h_time(beta)])

    grad = gradient(h_vec, beta_hat)  # shape (3, n_params)
    est = h_vec(beta_hat)
    chi2, p, df = joint_wald_test(est, grad, Sigma_hat)


Example 5: Inter-call composition with shared covariance
--------------------------------------------------------

    from pymargins._delta import combined_gradient, delta_variance

    # Two AMEs from the same session
    grad_a = gradient(h_a, beta_hat)
    grad_b = gradient(h_b, beta_hat)

    # Their difference: gradient is linear in the estimands' gradients
    grad_diff = combined_gradient([grad_a, grad_b], jnp.array([1.0, -1.0]))
    var_diff = delta_variance(grad_diff, Sigma_hat)

    # The joint covariance (for testing both at once)
    cov_joint = joint_covariance_of_results([grad_a, grad_b], Sigma_hat)
"""
