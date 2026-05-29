"""
pymargins._kappa

Compute the relative-curvature diagnostic κ for delta-method validity.

Theory
------
The delta method approximates Var(g(β̂)) ≈ ∇g^T Σ̂ ∇g using a first-order
Taylor expansion of g around β̂. The error in this approximation is governed
by the quadratic term, which involves the Hessian H_g.

Skovgaard's relative curvature scales the quadratic contribution against the
linear contribution, making it dimensionless and parameterization-invariant:

    κ = ||Σ̂^{1/2} H_g Σ̂^{1/2}|| / ||Σ̂^{1/2} ∇g||

The whitening transform Σ̂^{1/2} (computed via Cholesky factorization)
expresses both quantities in standard-error units. Without it, κ would
depend on the parameterization of β.

Interpretation
--------------
  κ < 0.1   : delta method is excellent; quadratic term is negligible
  κ < 0.3   : delta method is acceptable; flag if reporting precision matters
  κ ≥ 0.3   : delta method is suspect; prefer simulation or bootstrap

These thresholds are calibrated from the nonlinear-regression literature
(Bates & Watts, Skovgaard) and represent rough operational defaults rather
than sharp boundaries. The library exposes the threshold as a configurable
parameter on the session.

Limitations
-----------
κ catches one mode of delta-method failure (h is too curved in β). It does
NOT catch:
  - Non-normality of β̂ itself (small samples, separation, weak instruments)
  - Misspecified Σ̂ (wrong cluster structure, ignored heteroskedasticity)
  - Boundary effects (variance components, predicted probabilities at 0 or 1)

For these, comparing delta CIs to simulation CIs (delta_sim_disagreement)
or running a full bootstrap is more informative.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ._gradients import GradientBackend, gradient, hessian

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KappaNorm = Literal["spectral", "frobenius"]
KappaVerdict = Literal["delta_reliable", "delta_borderline", "delta_unreliable"]


# ---------------------------------------------------------------------------
# Single-estimand κ
# ---------------------------------------------------------------------------


def _kappa_core(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    cov_params: jnp.ndarray,
    L: jnp.ndarray | None,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
    norm: KappaNorm = "spectral",
) -> float:
    """Core κ computation with optional precomputed Cholesky factor."""
    grad = gradient(h, beta, backend=backend, fd_step=fd_step)
    if grad.ndim != 1:
        raise ValueError(
            f"_kappa_core only supports scalar estimands (grad.ndim==1); got grad.ndim={grad.ndim}. "
            "Use kappa_vector for vector estimands."
        )
    H = hessian(h, beta, backend=backend, fd_step=fd_step)

    if L is None:
        L = jnp.linalg.cholesky(cov_params)

    # Whitening: Σ̂ = L L^T; for β = L u the Hessian in u-coordinates is
    # L^T H L and the gradient is L^T ∇g. Both forms L^T H L and L H L^T
    # have the same spectral norm (similar matrices), but L^T H L is the
    # correct transformed Hessian.
    if jnp.isnan(L).any():
        # Ridge-regularize for rank-deficient Σ̂ (common with HC/cluster estimators)
        diag_mean = jnp.mean(jnp.diag(cov_params))
        ridge = 1e-8 * abs(float(diag_mean))
        reg_cov = cov_params + ridge * jnp.eye(cov_params.shape[0])
        L = jnp.linalg.cholesky(reg_cov)
        if jnp.isnan(L).any():
            return float("inf")

    H_white = L.T @ H @ L
    grad_white = L.T @ grad

    if norm == "spectral":
        # Spectral norm = largest singular value
        num = float(jnp.linalg.norm(H_white, ord=2))
    elif norm == "frobenius":
        num = float(jnp.linalg.norm(H_white, ord="fro"))
    else:
        raise ValueError(f"Unknown norm: {norm!r}")

    den = float(jnp.linalg.norm(grad_white))

    if den == 0.0:
        # Gradient is zero (e.g., at a critical point of h). κ is
        # undefined; return inf to signal that delta is not applicable.
        return float("inf")

    return num / den


def kappa(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    cov_params: jnp.ndarray,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
    norm: KappaNorm = "spectral",
) -> float:
    """Relative curvature κ of a scalar estimand h at β̂.

    Quantifies whether delta-method linearization is reliable for the given
    estimand. Computed as the ratio of whitened-Hessian norm to whitened-
    gradient norm, which makes it dimensionless and invariant to the
    parameterization of β.

    Parameters
    ----------
    h : callable
        Scalar-valued estimand h(β) → scalar. For vector estimands, use
        kappa_vector to compute per-component values.

    beta : jax array of shape (n_params,)
        β̂.

    cov_params : jax array of shape (n_params, n_params)
        Σ̂. Must be positive semi-definite (a Cholesky factorization is
        used internally for whitening).

    backend : str, default "autodiff"
        Gradient/Hessian backend. See _gradients.

    fd_step : float, default 1e-6
        FD step for FD-based backends.

    norm : str, default "spectral"
        Which matrix norm to use for the numerator:
          - "spectral": largest singular value (worst-case curvature
                       direction; standard definition in Skovgaard's work)
          - "frobenius": entry-wise norm (faster; slightly less conservative)

    Returns
    -------
    kappa : float
        Dimensionless curvature. Larger values mean delta is less reliable;
        see module docstring for thresholds.

    Notes
    -----
    Cholesky decomposition fails if cov_params is singular. Callers should
    handle this via a small ridge term added to cov_params, or by routing
    to a different inference method.
    """
    return _kappa_core(
        h, beta, cov_params, L=None, backend=backend, fd_step=fd_step, norm=norm
    )


def kappa_vector(
    h: Callable[[jnp.ndarray], jnp.ndarray],
    beta: jnp.ndarray,
    cov_params: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Per-component κ for a vector-valued estimand.

    For h returning a vector of k outputs, computes κ for each output
    component independently. The overall delta-method validity for the
    vector estimand is determined by the worst component: max(κᵢ).

    Implementation note: this loops over components, calling kappa on each.
    For high-dimensional outputs, this can be vectorized via vmap; we keep
    the simple loop until profiling indicates a need.

    Parameters
    ----------
    h : callable
        Vector-valued estimand h(β) → array of shape (k,).

    beta, cov_params, **kwargs : see kappa()

    Returns
    -------
    kappas : jax array of shape (k,)
        Per-component κ values.
    """
    out = h(beta)
    if jnp.ndim(out) == 0:
        return jnp.array([kappa(h, beta, cov_params, **kwargs)])
    elif jnp.ndim(out) == 1:
        n_outputs = int(out.shape[0])
        kappas = []
        for i in range(n_outputs):

            def h_i(b, i=i):
                return h(b)[i]

            kappas.append(kappa(h_i, beta, cov_params, **kwargs))
        return jnp.asarray(kappas)
    else:
        # Flatten multi-dimensional output and compute per-element kappa
        out_flat = jnp.reshape(out, (-1,))
        kappas = []
        for i in range(len(out_flat)):

            def h_i(b, i=i):
                return jnp.reshape(h(b), (-1,))[i]

            kappas.append(kappa(h_i, beta, cov_params, **kwargs))
        return jnp.asarray(kappas).reshape(out.shape)


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def classify_kappa(
    kappa_value: float,
    *,
    reliable_threshold: float = 0.1,
    borderline_threshold: float = 0.3,
) -> KappaVerdict:
    """Classify a κ value into a verdict tier.

    Thresholds are calibrated from nonlinear-regression literature
    convention. Tools that want stricter or looser thresholds can override
    via the keyword arguments.

    Parameters
    ----------
    kappa_value : float
        κ for an estimand.

    reliable_threshold : float, default 0.1
        Below this, classified "delta_reliable".

    borderline_threshold : float, default 0.3
        Below this, classified "delta_borderline". Above, "delta_unreliable".

    Returns
    -------
    verdict : str
    """
    if not np.isfinite(kappa_value):
        return "delta_unreliable"
    if kappa_value < reliable_threshold:
        return "delta_reliable"
    elif kappa_value < borderline_threshold:
        return "delta_borderline"
    else:
        return "delta_unreliable"


# ---------------------------------------------------------------------------
# Session-level diagnostic
# ---------------------------------------------------------------------------


def session_kappa(
    h_factory: Callable,
    beta: jnp.ndarray,
    cov_params: jnp.ndarray,
    representative_design: list,
    *,
    backend: GradientBackend = "autodiff",
    fd_step: float = 1e-6,
    norm: KappaNorm = "spectral",
    reliable_threshold: float = 0.1,
    borderline_threshold: float = 0.3,
) -> dict:
    """Session-level κ summary across the design space.

    Used by Margins.diagnose() to give an upfront sense of whether delta
    is reliable for the configured analytical posture, before computing
    specific estimands. Samples representative covariate vectors from the
    design and computes κ at each, summarizing the resulting distribution.

    The verdict is conservative: it reports the worst κ across samples,
    so a session that's reliable everywhere except in one region of the
    design space will be flagged. Users can inspect the full distribution
    in the result dict to see whether the worst case is something they
    care about.

    Parameters
    ----------
    h_factory : callable (X) -> h(β)
        Factory that constructs an estimand function from a covariate row
        or matrix. Typically builds a prediction-at-X estimand. The factory
        is called with each element of representative_design and the
        resulting h functions are evaluated for κ.

    beta : jax array
        β̂.

    cov_params : jax array
        Σ̂.

    representative_design : list
        Sampled covariate vectors. For typical use, a sample of rows from
        the model's design matrix or a quantile grid across the input space.
        The library's diagnose() method constructs this automatically.

    backend, fd_step, norm : see kappa()

    reliable_threshold, borderline_threshold : see classify_kappa()

    Returns
    -------
    diagnostic : dict with keys:
        'min'             : minimum κ across samples
        'median'          : median κ
        'max'             : maximum κ (drives the verdict)
        'distribution'    : array of all sampled κ values
        'verdict'         : one of "delta_reliable", "delta_borderline",
                            "delta_unreliable"
        'n_samples'       : number of design points sampled
        'recommendation'  : human-readable suggestion based on verdict
    """
    # Pre-compute Cholesky of cov_params once; it is constant across
    # design points and is the most expensive part of kappa().
    L = jnp.linalg.cholesky(cov_params)
    if jnp.isnan(L).any():
        # Ridge-regularize once for rank-deficient Σ̂ (common with HC/cluster estimators)
        diag_mean = jnp.mean(jnp.diag(cov_params))
        ridge = 1e-8 * abs(float(diag_mean))
        reg_cov = cov_params + ridge * jnp.eye(cov_params.shape[0])
        L = jnp.linalg.cholesky(reg_cov)
        if jnp.isnan(L).any():
            L = None

    kappas = []
    for X in representative_design:
        h_X = h_factory(X)
        kappas.append(
            _kappa_core(
                h_X,
                beta,
                cov_params,
                L=L,
                backend=backend,
                fd_step=fd_step,
                norm=norm,
            )
        )

    kappas = jnp.asarray(kappas)
    max_k = float(kappas.max())
    verdict = classify_kappa(
        max_k,
        reliable_threshold=reliable_threshold,
        borderline_threshold=borderline_threshold,
    )

    recommendations = {
        "delta_reliable": (
            "Delta method is reliable across the design. "
            "Spot-check with method='simulation' for estimands at extreme "
            "covariate values."
        ),
        "delta_borderline": (
            "Delta method is borderline. Consider running specific estimands "
            "with method='simulation' or enabling automatic fallback via "
            "the session's kappa_threshold."
        ),
        "delta_unreliable": (
            "Delta method is unreliable for this analytical posture. "
            "Use method='simulation' or method='bootstrap' for inference, "
            "or reconsider the inference scale (phi)."
        ),
    }

    return {
        "min": float(kappas.min()),
        "median": float(jnp.median(kappas)),
        "max": max_k,
        "distribution": kappas,
        "verdict": verdict,
        "n_samples": len(representative_design),
        "recommendation": recommendations[verdict],
    }


# ---------------------------------------------------------------------------
# Cross-validation against simulation
# ---------------------------------------------------------------------------


def delta_simulation_disagreement(
    estimate: jnp.ndarray,
    grad: jnp.ndarray,
    cov_params: jnp.ndarray,
    h: Callable,
    beta: jnp.ndarray,
    *,
    level: float = 0.95,
    n_sim: int = 4000,
    rng_seed: int | None = None,
    phi: Callable | None = None,
) -> float:
    """Compare delta-method CI to a Krinsky–Robb simulation CI.

    Useful as a second diagnostic alongside κ. While κ measures curvature
    of h in β-space, this measures whether the resulting CIs disagree —
    catching not only curvature but also non-normality of the implied
    sampling distribution of g(β̂) under the model's Σ̂.

    Returns the maximum relative disagreement between the two CIs:

        max(|delta_lower - sim_lower|, |delta_upper - sim_upper|) / |estimate|

    Small values (< 0.05) indicate good agreement. Large values flag
    estimands where delta and simulation produce meaningfully different
    inference, even if κ is moderate.

    Parameters
    ----------
    estimate : jax array
        Point estimate on the inference scale.

    grad : jax array
        ∇h at β̂.

    cov_params : jax array
        Σ̂.

    h : callable
        The estimand. Re-evaluated for each simulation draw.

    beta : jax array
        β̂.

    level : float, default 0.95

    n_sim : int, default 4000
        Simulation sample size. Larger gives more stable estimates of the
        sim CI; 4000 is generally sufficient for diagnostic purposes.

    rng_seed : int, optional
        Seed for reproducibility.

    phi : callable, optional
        Back-transform; applied to both delta and simulation CI endpoints
        for the comparison.

    Returns
    -------
    disagreement : float
        Relative max CI disagreement. By convention, returns +inf if
        estimate is exactly zero (avoids divide by zero).
    """
    from ._delta import delta_confint

    # Delta CI (handles both scalar and vector estimands)
    d_lower, d_upper = delta_confint(
        estimate,
        grad,
        cov_params,
        level=level,
        phi=phi,
    )
    d_lower = np.asarray(d_lower)
    d_upper = np.asarray(d_upper)

    # Simulation CI
    rng = np.random.default_rng(rng_seed)
    Sigma_np = np.asarray(cov_params)
    beta_np = np.asarray(beta)
    draws = rng.multivariate_normal(beta_np, Sigma_np, size=n_sim)
    try:
        h_draws = np.asarray(jax.vmap(h)(jnp.asarray(draws)))
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerBoolConversionError,
        jax.errors.TracerIntegerConversionError,
        jax.errors.UnexpectedTracerError,
    ):
        h_draws = np.array([np.asarray(h(jnp.asarray(b))) for b in draws])

    if phi is not None:
        h_draws = np.asarray(phi(jnp.asarray(h_draws)))

    alpha = (1.0 - level) / 2.0
    s_lower = np.quantile(h_draws, alpha, axis=0)
    s_upper = np.quantile(h_draws, 1.0 - alpha, axis=0)

    # Normalize by the absolute estimate (on the reporting scale if phi is
    # provided). Vector estimands return the maximum per-component
    # disagreement so a single scalar still summarizes the diagnostic.
    est_report = np.asarray(phi(estimate)) if phi is not None else np.asarray(estimate)
    ref = np.abs(est_report)

    diff = np.maximum(np.abs(d_lower - s_lower), np.abs(d_upper - s_upper))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ref > 0, diff / np.where(ref > 0, ref, 1.0), np.inf)
    return float(np.max(rel))


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: κ for a single estimand
----------------------------------

    import jax.numpy as jnp
    from pymargins._kappa import kappa, classify_kappa

    def h(beta):
        # log relative risk
        p1 = jax.scipy.special.expit(x1 @ beta)
        p0 = jax.scipy.special.expit(x0 @ beta)
        return jnp.log(p1) - jnp.log(p0)

    k = kappa(h, beta_hat, Sigma_hat)
    verdict = classify_kappa(k)
    if verdict == "delta_unreliable":
        # Auto-route to simulation
        ...


Example 2: Session-level diagnostic across the design
-----------------------------------------------------

    from pymargins._kappa import session_kappa

    # Sample 50 rows from the design
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X.shape[0], size=50, replace=False)
    representative_X = [X[i] for i in sample_idx]

    def h_factory(x):
        return lambda beta: jnp.log(jax.scipy.special.expit(x @ beta))

    diag = session_kappa(h_factory, beta_hat, Sigma_hat, representative_X)
    print(f"κ range: {diag['min']:.3f} to {diag['max']:.3f}")
    print(f"Verdict: {diag['verdict']}")
    print(diag['recommendation'])


Example 3: Cross-validation against simulation
----------------------------------------------

    from pymargins._kappa import delta_simulation_disagreement
    from pymargins._gradients import gradient

    grad = gradient(h, beta_hat)
    estimate = h(beta_hat)
    disagreement = delta_simulation_disagreement(
        estimate, grad, Sigma_hat, h, beta_hat,
        level=0.95,
        n_sim=4000,
        rng_seed=42,
        phi=jnp.exp,
    )
    if disagreement > 0.05:
        # Delta and simulation give meaningfully different CIs
        print(f"Disagreement: {disagreement:.1%}")
"""
