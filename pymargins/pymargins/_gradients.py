"""
pymargins._gradients

Compute gradients (and Hessians) of estimand functions h(β) with respect to
model parameters. This is the foundational numerical layer of the library —
every delta-method computation and every κ diagnostic depends on it.

Backends
--------
- "autodiff" : jax.grad through the entire estimand. Exact to machine
               precision. Requires h to be implementable in jax.numpy.

- "fd"       : Central-difference finite differences on the entire estimand.
               Used only when no JAX path is available. About 10 correct
               digits at fd_step=1e-6 for float64.

- "wrapped_fd": Indistinguishable from "autodiff" at this layer. The FD lives
               inside a custom_jvp on the model's predict function, hidden
               from this module. Use this naming when constructing adapters
               so downstream code knows what's actually happening, even
               though the gradient computation is identical to autodiff.

Design
------
This module is purely functional. It has no knowledge of estimands, sessions,
adapters, or inference. It accepts a callable h: β → scalar/vector and a
parameter vector β, and returns derivatives. All higher-level concerns —
which estimand, which scenario, which session scale — are resolved upstream.
"""

from __future__ import annotations
from typing import Callable, Literal
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GradientBackend = Literal["autodiff", "fd", "wrapped_fd"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gradient(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
) -> jnp.ndarray:
    """Compute ∇_β h(β).

    For scalar-valued h, returns the gradient vector ∂h/∂β_j of shape
    (n_params,). For vector-valued h with k outputs, returns the Jacobian
    matrix of shape (k, n_params).

    The autodiff and wrapped_fd backends produce mathematically identical
    results to within IEEE-754 roundoff (they execute the same JAX ops; the
    "wrapped" naming refers only to FD living inside a custom_jvp upstream).
    The fd backend is a fully numerical fallback for h functions that can't
    be expressed in JAX.

    Parameters
    ----------
    h : callable
        Function to differentiate. Must accept a JAX array of shape
        (n_params,) and return either a scalar JAX array or a 1D JAX array.
        For autodiff/wrapped_fd, h must be composed entirely of JAX-traceable
        operations (no Python conditionals on tracer values, no NumPy ops).
        For fd, h may use any operations on NumPy/JAX arrays.

    beta : jax array of shape (n_params,)
        Parameter vector at which to evaluate the gradient. Typically β̂
        from the fitted model.

    backend : str, default "autodiff"
        Which gradient method to use. See module docstring.

    fd_step : float, default 1e-6
        Perturbation size for the fd backend. The default is calibrated for
        float64 precision; smaller values increase cancellation error,
        larger values increase truncation error.

    Returns
    -------
    grad : jax array
        ∇h(β). Shape (n_params,) for scalar h or (n_outputs, n_params) for
        vector h.

    Raises
    ------
    ValueError
        If backend is not one of the recognized values.

    TypeError
        If h is not differentiable by JAX and backend is "autodiff" or
        "wrapped_fd". Implementations should catch this and route to "fd"
        or to simulation-based inference.
    """
    if backend in ("autodiff", "wrapped_fd"):
        return _gradient_autodiff(h, beta)
    elif backend == "fd":
        return _gradient_fd(h, beta, fd_step)
    else:
        raise ValueError(f"Unknown gradient backend: {backend!r}")


def hessian(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
) -> jnp.ndarray:
    """Compute the Hessian H_h(β) = ∂²h/∂β∂β^T.

    Used by the κ diagnostic to assess delta-method validity. For scalar h,
    returns a square matrix of shape (n_params, n_params). For vector h,
    returns a 3D tensor of shape (n_outputs, n_params, n_params); use
    hessian_vector_product for the more common case where you need only
    H @ v for some direction v.

    Hessian computations through wrapped_fd are FD-quality on the wrapped
    primitive (still ~10 digits) but exact on the surrounding structure.
    Full FD Hessians (backend="fd") compound numerical error and should be
    avoided when any JAX path exists.

    Parameters
    ----------
    h : callable
        Function whose Hessian is requested. Same constraints as gradient().

    beta : jax array of shape (n_params,)
        Evaluation point.

    backend : str, default "autodiff"
        See gradient().

    fd_step : float, default 1e-6
        FD step. For Hessians, FD error scales worse than for gradients —
        prefer autodiff or wrapped_fd whenever possible.

    Returns
    -------
    H : jax array
        Hessian of shape (n_params, n_params) for scalar h, or
        (n_outputs, n_params, n_params) for vector h.
    """
    if backend in ("autodiff", "wrapped_fd"):
        return jax.hessian(h)(beta)
    elif backend == "fd":
        return _hessian_fd(h, beta, fd_step)
    else:
        raise ValueError(f"Unknown gradient backend: {backend!r}")


def directional_derivative(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    direction: jnp.ndarray,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
) -> jnp.ndarray:
    """Compute the directional derivative (∇h)·v for direction v.

    Cheaper than the full gradient when only one direction is needed, e.g.,
    inside iterative methods. For autodiff/wrapped_fd, uses jax.jvp which
    is one forward pass instead of n_params evaluations.

    Parameters
    ----------
    h : callable
        Function to differentiate.

    beta : jax array of shape (n_params,)
        Evaluation point.

    direction : jax array of shape (n_params,)
        Direction vector v.

    backend : str, default "autodiff"
        See gradient().

    fd_step : float, default 1e-6
        FD step.

    Returns
    -------
    deriv : jax array
        (∇h(β))·v. Scalar for scalar h, vector for vector h.
    """
    if backend in ("autodiff", "wrapped_fd"):
        _, deriv = jax.jvp(h, (beta,), (direction,))
        return deriv
    elif backend == "fd":
        plus = h(beta + fd_step * direction)
        minus = h(beta - fd_step * direction)
        return (plus - minus) / (2 * fd_step)
    else:
        raise ValueError(f"Unknown gradient backend: {backend!r}")


def hessian_vector_product(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    direction: jnp.ndarray,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
) -> jnp.ndarray:
    """Compute H_h(β) @ v without forming the full Hessian.

    For large n_params, the full Hessian is n_params² entries while one HVP
    is n_params entries. For κ diagnostics that need only ||Σ^{1/2} H Σ^{1/2}||
    along the worst direction, iterative methods using HVPs are much cheaper
    than materializing H.

    Parameters
    ----------
    Same as directional_derivative, but computes the second-order quantity.

    Returns
    -------
    hvp : jax array of shape (n_params,)
        H_h(β) @ v.
    """
    if backend in ("autodiff", "wrapped_fd"):
        return jax.grad(lambda b: jnp.dot(jax.grad(h)(b), direction))(beta)
    elif backend == "fd":
        # Central-difference HVP: (∇h(β + ε·v) - ∇h(β - ε·v)) / (2ε)
        grad_plus = _gradient_fd(h, beta + fd_step * direction, fd_step)
        grad_minus = _gradient_fd(h, beta - fd_step * direction, fd_step)
        return (grad_plus - grad_minus) / (2 * fd_step)
    else:
        raise ValueError(f"Unknown gradient backend: {backend!r}")


def make_predict_with_fd_jvp(
    predict_native: Callable,
    fd_step: float = 1e-6,
) -> Callable:
    """Wrap a non-JAX predict function as a JAX primitive with FD-based JVP.

    Used by ModelAdapter implementations for black-box models whose predict
    function cannot be reimplemented in JAX. The returned function accepts
    JAX arrays and is fully compatible with jax.grad, jax.hessian, and
    jax.jvp. Internally, it uses central-difference FD to compute directional
    derivatives at the model boundary; downstream autodiff over the estimand
    structure remains exact.

    This is the cleanest way to integrate a non-differentiable model into
    the JAX-based inference pipeline. The custom JVP isolates the FD to the
    one operation that needs it; everything composed with this primitive
    benefits from exact autodiff.

    Parameters
    ----------
    predict_native : callable (beta_np, X) -> array_np
        Native prediction function. Receives NumPy beta of shape (n_params,)
        and an arbitrary X (typically a NumPy array or pandas DataFrame),
        returns a NumPy array of predictions.

    fd_step : float, default 1e-6
        FD step for directional derivatives. The default is appropriate
        for float64; if the model has internal numerical solvers with
        looser tolerances, this may need to be increased.

    Returns
    -------
    predict_wrapped : callable (beta_jax, X) -> array_jax
        JAX-compatible wrapper. jax.grad, jax.hessian, and jax.jvp all work
        through this function. The X argument is passed through unchanged
        (it is not differentiated against; only β is).
    """
    @jax.custom_jvp
    def predict_wrapped(beta, X):
        return jnp.asarray(predict_native(np.asarray(beta), X))

    @predict_wrapped.defjvp
    def predict_wrapped_jvp(primals, tangents):
        beta, X = primals
        beta_dot, _ = tangents
        beta_np = np.asarray(beta)
        beta_dot_np = np.asarray(beta_dot)

        plus = predict_native(beta_np + fd_step * beta_dot_np, X)
        minus = predict_native(beta_np - fd_step * beta_dot_np, X)
        deriv = (plus - minus) / (2 * fd_step)

        return predict_wrapped(beta, X), jnp.asarray(deriv)

    return predict_wrapped


def make_glm_jvp_wrapper(
    family,
) -> Callable:
    """Wrap a GLM prediction with a custom JVP using the link's analytical
    derivative.

    For any GLM with mean function μ = g⁻¹(η) where η = Xβ, the gradient
    w.r.t. β is (dg⁻¹/dη at η) · X. statsmodels' Family objects expose
    this via `family.link.inverse_deriv(eta)`, so one wrapper handles all
    standard GLM families and links uniformly.

    Compared to a JAX reimplementation of the prediction (Path A), this
    wrapper has the advantage of using statsmodels' canonical link
    implementations — useful when statsmodels' predict has nontrivial
    edge-case logic (offsets, exposure, weights handled in nonstandard
    ways) that you don't want to re-implement.

    Parameters
    ----------
    family : statsmodels.genmod.families.Family
        The fitted model's family object. Must expose .link.inverse and
        .link.inverse_deriv.

    Returns
    -------
    predict_wrapped : callable (beta, X, offset=None) -> array
        JAX-compatible prediction. Supports an optional offset added to
        the linear predictor before applying the link inverse.
    """
    link = family.link

    @jax.custom_jvp
    def predict_wrapped(beta, X, offset=None):
        beta_np = np.asarray(beta)
        X_np = np.asarray(X)
        eta = X_np @ beta_np
        if offset is not None:
            eta = eta + np.asarray(offset)
        return jnp.asarray(link.inverse(eta))

    @predict_wrapped.defjvp
    def predict_wrapped_jvp(primals, tangents):
        beta, X, offset = primals
        beta_dot, X_dot, _ = tangents
        beta_np = np.asarray(beta)
        beta_dot_np = np.asarray(beta_dot)
        X_np = np.asarray(X)

        eta = X_np @ beta_np
        if offset is not None:
            eta = eta + np.asarray(offset)

        # Forward
        mu = link.inverse(eta)

        # Tangent: dμ/dt = (dg⁻¹/dη) · (dη/dt)
        # dη/dt = X · β̇ + Ẋ · β  (handles both differentiating directions)
        eta_dot = X_np @ beta_dot_np
        if X_dot is not None and not isinstance(X_dot, type(None)):
            X_dot_np = np.asarray(X_dot)
            eta_dot = eta_dot + X_dot_np @ beta_np

        mu_dot = link.inverse_deriv(eta) * eta_dot

        return jnp.asarray(mu), jnp.asarray(mu_dot)

    return predict_wrapped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gradient_autodiff(h, beta):
    """Use JAX autodiff to compute the gradient. Dispatches to grad or
    jacobian depending on output rank."""
    out = h(beta)
    if jnp.ndim(out) == 0:
        return jax.grad(h)(beta)
    else:
        return jax.jacobian(h)(beta)


def _gradient_fd(h, beta, eps):
    """Central-difference FD gradient. Loops over parameters; for n_params
    parameters this requires 2 * n_params evaluations of h."""
    beta = jnp.asarray(beta)
    n = beta.shape[0]
    f0 = h(beta)
    is_scalar = jnp.ndim(f0) == 0

    if is_scalar:
        grad = np.zeros(n)
        for j in range(n):
            e_j = jnp.zeros(n).at[j].set(eps)
            plus = h(beta + e_j)
            minus = h(beta - e_j)
            grad[j] = float((plus - minus) / (2 * eps))
        return jnp.asarray(grad)
    else:
        k = f0.shape[0]
        grad = np.zeros((k, n))
        for j in range(n):
            e_j = jnp.zeros(n).at[j].set(eps)
            plus = h(beta + e_j)
            minus = h(beta - e_j)
            grad[:, j] = np.asarray((plus - minus) / (2 * eps))
        return jnp.asarray(grad)


def _hessian_fd(h, beta, eps):
    """Central-difference FD Hessian. Loops over parameter pairs; for
    n_params parameters this requires O(n_params²) evaluations.

    Numerical quality is poor compared to autodiff Hessians — FD-of-FD
    compounds the cancellation problem. Avoid when any JAX path exists.
    """
    beta = jnp.asarray(beta)
    n = beta.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = jnp.zeros(n).at[i].set(eps)
            e_j = jnp.zeros(n).at[j].set(eps)
            f_pp = float(h(beta + e_i + e_j))
            f_pm = float(h(beta + e_i - e_j))
            f_mp = float(h(beta - e_i + e_j))
            f_mm = float(h(beta - e_i - e_j))
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return jnp.asarray(H)


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Gradient of a simple estimand via autodiff
-----------------------------------------------------

    import jax.numpy as jnp
    from pymargins._gradients import gradient

    # An estimand: prediction at a fixed x for a logit model
    x = jnp.array([1.0, 0.5, -0.3])
    def h(beta):
        return jax.scipy.special.expit(x @ beta)

    beta_hat = jnp.array([0.2, 1.5, -0.8])
    grad = gradient(h, beta_hat, backend="autodiff")
    # grad has shape (3,) — one entry per parameter


Example 2: Wrapping a non-JAX prediction function
-------------------------------------------------

    from pymargins._gradients import make_predict_with_fd_jvp, gradient

    # Suppose statsmodels_model.predict accepts (beta, X) and returns ndarray
    def native_predict(beta_np, X):
        return statsmodels_model.predict(beta_np, exog=X)

    predict_jax = make_predict_with_fd_jvp(native_predict, fd_step=1e-6)

    # Now build an estimand using predict_jax — it composes with autodiff:
    def h(beta):
        return predict_jax(beta, X_target).mean()

    grad = gradient(h, beta_hat, backend="wrapped_fd")
    # FD is hidden inside predict_jax's JVP; the gradient call is autodiff.


Example 3: Wrapping a GLM with the analytical link derivative
-------------------------------------------------------------

    from pymargins._gradients import make_glm_jvp_wrapper, gradient

    # For a fitted statsmodels GLM result
    family = statsmodels_glm_result.family
    predict_jax = make_glm_jvp_wrapper(family)

    def h(beta):
        return predict_jax(beta, X_target, offset=None).mean()

    grad = gradient(h, beta_hat)
    # Gradient uses family.link.inverse_deriv internally, equivalent to
    # autodiff through a JAX reimplementation but reuses statsmodels' code.


Example 4: Hessian-vector product for κ diagnostic
--------------------------------------------------

    from pymargins._gradients import hessian_vector_product

    # Worst-case curvature direction (e.g., from power iteration)
    v = jnp.ones(n_params) / jnp.sqrt(n_params)
    hv = hessian_vector_product(h, beta_hat, v)
    # hv has shape (n_params,) — H @ v without materializing H
"""
