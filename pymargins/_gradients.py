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
import warnings

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

    Note: ``wrapped_fd`` works for gradients but may not support Hessians
    through ``jax.hessian`` because custom JVPs do not automatically expose
    second-order derivatives. For Hessians of wrapped functions, use
    ``backend="fd"`` (which applies FD to the entire estimand, including the
    wrapped primitive).

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

    Notes
    -----
    ``wrapped_fd`` works for gradients but may not support HVPs through nested
    ``jax.grad`` because custom JVPs do not automatically expose second-order
    derivatives. For HVPs of wrapped functions, use ``backend="fd"``.
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


def _concrete_primal(x):
    """Extract the concrete primal value from a JAX array or JVPTracer.

    Uses a try/except around np.asarray as a fallback because the
    ``.primal`` attribute name is not part of JAX's public API and
    may change across versions.
    """
    if x is None:
        return None
    if hasattr(x, "primal"):
        return x.primal
    try:
        return np.asarray(x)
    except Exception:
        return x


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
        through this function. Both β and X can be differentiated against
        (the latter is needed for dydx slopes).
    """
    @jax.custom_jvp
    def predict_wrapped(beta, X):
        return jnp.asarray(predict_native(np.asarray(beta), X))

    @predict_wrapped.defjvp
    def predict_wrapped_jvp(primals, tangents):
        beta, X = primals
        beta_dot, X_dot = tangents
        # During forward-mode (jax.jvp, jax.hessian) primals may be JVPTracers.
        # Extract the concrete underlying arrays for FD evaluations.
        beta_np = np.asarray(_concrete_primal(beta))
        X_np = np.asarray(_concrete_primal(X))

        # Evaluate at primal to get output shape
        f0 = predict_native(beta_np, X_np)
        is_scalar = np.ndim(f0) == 0
        f0 = np.atleast_1d(f0)
        n_out = f0.shape[0]

        # ------------------------------------------------------------------
        # Beta directional derivative
        # ------------------------------------------------------------------
        # beta_dot may be a JAX tracer (during reverse-mode autodiff),
        # so we cannot convert it to NumPy. Instead, we compute the full
        # Jacobian w.r.t. beta via FD using basis vectors, then compute
        # the matrix-vector product J @ beta_dot in JAX space.
        deriv_beta = jnp.array(0.0)
        if beta_dot is not None:
            n_params = beta_np.shape[0]
            J_beta = np.zeros((n_out, n_params))
            for j in range(n_params):
                e_j = np.zeros(n_params)
                e_j[j] = fd_step
                f_plus = np.atleast_1d(predict_native(beta_np + e_j, X_np))
                f_minus = np.atleast_1d(predict_native(beta_np - e_j, X_np))
                J_beta[:, j] = (f_plus - f_minus) / (2 * fd_step)
            J_beta_jax = jnp.asarray(J_beta)
            if n_out == 1:
                deriv_beta = jnp.dot(J_beta_jax.ravel(), beta_dot)
            else:
                deriv_beta = J_beta_jax @ beta_dot

        # ------------------------------------------------------------------
        # X directional derivative (same tracer-safe pattern)
        # ------------------------------------------------------------------
        deriv_X = jnp.array(0.0)
        if X_dot is not None:
            # WARNING: this path is O(n_obs · n_features) predict_native
            # calls. For a 1000×10 design that is 20_000 calls per HVP.
            # Fine as an adapter fallback, but avoid in large-batch loops.
            X_np = np.asarray(X_np)
            if X_np.ndim != 2:
                raise ValueError(
                    f"make_predict_with_fd_jvp expects a 2D array for X; got {X_np.ndim}D"
                )
            n_obs, n_features = X_np.shape
            J_X = np.zeros((n_out, n_obs, n_features))
            for i in range(n_obs):
                for j in range(n_features):
                    e_ij = np.zeros_like(X_np)
                    e_ij[i, j] = fd_step
                    f_plus = np.atleast_1d(predict_native(beta_np, X_np + e_ij))
                    f_minus = np.atleast_1d(predict_native(beta_np, X_np - e_ij))
                    J_X[:, i, j] = (f_plus - f_minus) / (2 * fd_step)
            J_X_jax = jnp.asarray(J_X)
            deriv_X = jnp.einsum('oij,ij->o', J_X_jax, X_dot)

        deriv = deriv_beta + deriv_X
        if is_scalar:
            deriv = deriv.reshape(())
        return predict_wrapped(beta, X), deriv

    return predict_wrapped


def _jax_link_inverse(link):
    """Return a JAX-native link inverse for common statsmodels links."""
    name = type(link).__name__
    if name == "Logit":
        return lambda z: 1.0 / (1.0 + jnp.exp(-z))
    if name == "Probit":
        return jax.scipy.special.ndtr
    if name == "CLogLog":
        return lambda z: 1.0 - jnp.exp(-jnp.exp(z))
    if name == "LogLog":
        return lambda z: jnp.exp(-jnp.exp(-z))
    if name == "LogC":
        return lambda z: 1.0 - jnp.exp(z)
    if name == "Log":
        return jnp.exp
    if name == "Identity":
        return lambda z: z
    if name == "Power":
        p = float(getattr(link, "power", 1.0))
        if p == 0.0:
            # statsmodels treats Power(0) as the log link
            return jnp.exp
        return lambda z: jnp.where(z <= 0, 0.0, jnp.power(z, 1.0 / p))
    if name == "InversePower":
        return lambda z: jnp.where(jnp.abs(z) < 1e-12, jnp.where(z >= 0, 1e12, -1e12), 1.0 / z)
    if name == "InverseSquared":
        return lambda z: jnp.where(z <= 0, 0.0, 1.0 / jnp.sqrt(z))
    if name == "Sqrt":
        return lambda z: z ** 2
    if name == "Cauchy":
        return lambda z: 0.5 + (1.0 / jnp.pi) * jnp.arctan(z)
    if name == "NegativeBinomial":
        alpha = float(getattr(link, "alpha", 1.0))
        def nb_inv(z):
            ez = jnp.exp(z)
            denom = alpha * (ez - 1.0) + 1.0
            denom = jnp.where(
                jnp.abs(denom) < 1e-12,
                jnp.where(denom >= 0, 1e-12, -1e-12),
                denom,
            )
            return ez / denom
        return nb_inv
    raise NotImplementedError(f"No JAX mapping for link {name!r}")


def _jax_link_inverse_deriv(link):
    """Return a JAX-native link inverse derivative for common statsmodels links."""
    name = type(link).__name__
    if name == "Logit":
        def deriv(z):
            t = jnp.exp(-z)
            return t / (1.0 + t) ** 2
        return deriv
    if name == "Probit":
        c = 1.0 / jnp.sqrt(2.0 * jnp.pi)
        return lambda z: c * jnp.exp(-0.5 * z ** 2)
    if name == "CLogLog":
        return lambda z: jnp.exp(z - jnp.exp(z))
    if name == "LogLog":
        return lambda z: jnp.exp(-z - jnp.exp(-z))
    if name == "LogC":
        return lambda z: -jnp.exp(z)
    if name == "Log":
        return jnp.exp
    if name == "Identity":
        return lambda z: jnp.ones_like(z)
    if name == "Power":
        p = float(getattr(link, "power", 1.0))
        if p == 0.0:
            # statsmodels treats Power(0) as log link → derivative is exp
            return jnp.exp
        return lambda z: jnp.where(z <= 0, 0.0, jnp.power(z, 1.0 / p - 1.0) / p)
    if name == "InversePower":
        return lambda z: jnp.where(jnp.abs(z) < 1e-12, -1e24, -1.0 / (z ** 2))
    if name == "InverseSquared":
        return lambda z: jnp.where(z <= 0, 0.0, -0.5 / (z ** 1.5))
    if name == "Sqrt":
        return lambda z: 2.0 * z
    if name == "Cauchy":
        return lambda z: 1.0 / (jnp.pi * (1.0 + z ** 2))
    if name == "NegativeBinomial":
        alpha = float(getattr(link, "alpha", 1.0))
        def deriv(z):
            ez = jnp.exp(z)
            denom = alpha * (ez - 1.0) + 1.0
            denom = jnp.where(
                jnp.abs(denom) < 1e-12,
                jnp.where(denom >= 0, 1e-12, -1e-12),
                denom,
            )
            return ez / (denom ** 2)
        return deriv
    raise NotImplementedError(f"No JAX mapping for link derivative {name!r}")


def make_glm_jvp_wrapper(
    family,
) -> Callable:
    """Wrap a GLM prediction with a custom JVP using the link's analytical
    derivative.

    For any GLM with mean function μ = g⁻¹(η) where η = Xβ, the gradient
    w.r.t. β is (dg⁻¹/dη at η) · X. This wrapper implements both the
    forward evaluation and the tangent using JAX-native operations for
    common links, making it fully compatible with jax.grad, jax.hessian,
    jax.jvp, and jax.vmap.

    Parameters
    ----------
    family : statsmodels.genmod.families.Family
        The fitted model's family object. Must use a link supported by
        _jax_link_inverse and _jax_link_inverse_deriv.

    Returns
    -------
    predict_wrapped : callable (beta, X, offset=None) -> array
        JAX-compatible prediction. Supports an optional offset added to
        the linear predictor before applying the link inverse.
        Note: offset is passed as a keyword-default arg; while this works
        with jax.grad/jvp/hessian today, jax.jit or nondiff_argnums may
        require baking offset usage into the factory call site.
    """
    link = family.link
    link_inv = _jax_link_inverse(link)
    link_inv_deriv = _jax_link_inverse_deriv(link)

    @jax.custom_jvp
    def predict_wrapped(beta, X, offset=None):
        eta = jnp.asarray(X) @ jnp.asarray(beta)
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        return link_inv(eta)

    @predict_wrapped.defjvp
    def predict_wrapped_jvp(primals, tangents):
        beta, X, offset = primals
        beta_dot, X_dot, offset_dot = tangents

        eta = jnp.asarray(X) @ jnp.asarray(beta)
        if offset is not None:
            eta = eta + jnp.asarray(offset)

        mu = link_inv(eta)

        # Tangent: dμ/dt = (dg⁻¹/dη) · (dη/dt)
        # dη/dt = X · β̇ + Ẋ · β + offseṫ
        eta_dot = jnp.asarray(X) @ beta_dot
        if X_dot is not None:
            eta_dot = eta_dot + X_dot @ jnp.asarray(beta)
        if offset_dot is not None:
            eta_dot = eta_dot + offset_dot

        mu_dot = link_inv_deriv(eta) * eta_dot

        return mu, mu_dot

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
    f0 = h(beta)
    if jnp.ndim(f0) != 0:
        raise ValueError(
            "_hessian_fd only supports scalar-valued estimands. "
            "For vector estimands, compute per-component Hessians or use autodiff."
        )
    if n > 50:
        warnings.warn(
            f"_hessian_fd is O(n²) and explicitly slow (n_params={n}). "
            "Prefer autodiff or wrapped_fd backends.",
            RuntimeWarning,
            stacklevel=3,
        )
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
