"""
pymargins._inference

Inference engine: takes an estimand function h, an adapter, and a session
configuration; produces a MarginsResult with point estimate, SE, CI, and
diagnostics. Dispatches between delta, simulation, and bootstrap methods.

This is the orchestration layer above the numerical kernels (_gradients,
_delta, _kappa). It owns the policy logic — when to fall back from delta
to simulation, how to validate adapter capabilities against the requested
method, how to handle non-differentiable estimands.
"""

from __future__ import annotations
from typing import Callable, Optional, Literal, Any
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np

from ._gradients import gradient, GradientBackend
from ._delta import (
    delta_se,
    delta_confint,
    delta_variance,
    delta_wald_test,
    joint_wald_test,
)
from ._kappa import kappa, kappa_vector, classify_kappa, delta_simulation_disagreement
from ._estimands import is_jax_differentiable
from ._adapter import ModelAdapter, InferenceMethod


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Bundles inference-related configuration extracted from the session.

    Passed through the engine so individual functions don't each take a long
    parameter list. Constructed by Margins from the session's attributes
    when invoking the engine.

    Attributes
    ----------
    method : str
        "delta", "simulation", or "bootstrap".

    level : float
        Confidence level for CIs.

    phi : callable, optional
        Back-transform from inference scale to reporting scale.

    phi_inv : callable, optional
        Forward transform from reporting scale to inference scale.

    kappa_threshold : float
        Curvature above which delta auto-falls-back to simulation. Set to
        infinity to disable fallback.

    gradient_backend : str
        Which gradient computation method.

    fd_step : float
        Step size for FD-based gradients.

    n_sim : int
        Sample size for simulation-based inference.

    n_boot : int
        Number of bootstrap replicates.

    rng_seed : int, optional
        For reproducibility.

    diagnostics : bool
        Whether to compute κ and other diagnostics.

    cov_params : jax array
        Σ̂. Pre-extracted from the adapter so the engine doesn't need to
        re-request it.
    """
    method: InferenceMethod = "delta"
    level: float = 0.95
    phi: Optional[Callable] = None
    phi_inv: Optional[Callable] = None
    kappa_threshold: float = 0.3
    gradient_backend: GradientBackend = "autodiff"
    fd_step: float = 1e-6
    n_sim: int = 4000
    n_boot: int = 1000
    rng_seed: Optional[int] = None
    diagnostics: bool = True
    cov_params: Optional[jnp.ndarray] = None


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------

def run_inference(
    h: Callable,
    adapter: ModelAdapter,
    config: InferenceConfig,
    *,
    estimand_metadata: Optional[dict] = None,
) -> dict:
    """Compute estimate, SE, CI, and diagnostics for an estimand.

    Top-level dispatch. Validates that the requested method is supported by
    the adapter, runs the appropriate inference path, and returns a dict of
    results (which Margins wraps into a MarginsResult).

    Parameters
    ----------
    h : callable
        Estimand function on the inference scale.

    adapter : ModelAdapter
        Provides β̂ (via coefficients) and predict.

    config : InferenceConfig
        Inference-time settings.

    estimand_metadata : dict, optional
        Bookkeeping passed through to the result (variable names, scenario
        labels, etc.) for output formatting.

    Returns
    -------
    result_data : dict
        Keys for assembling a MarginsResult: estimate, std_error,
        conf_int_lower, conf_int_upper, method, level, kappa, gradient,
        draws, fallback_triggered, etc.
    """
    method = config.method
    supported = adapter.supported_inference_methods

    if method not in supported:
        raise ValueError(
            f"Adapter {type(adapter).__name__} does not support method "
            f"'{method}'. Supported: {sorted(supported)}."
        )

    if method == "delta":
        # Check differentiability before attempting
        beta = adapter.coefficients()
        if not is_jax_differentiable(h, beta):
            if "simulation" in supported:
                # Auto-route to simulation with a warning marker in the result
                return _run_simulation(h, adapter, config, estimand_metadata,
                                       fallback_reason="non_differentiable")
            elif "bootstrap" in supported:
                return _run_bootstrap(h, adapter, config, estimand_metadata,
                                      fallback_reason="non_differentiable")
            else:
                raise ValueError(
                    "Estimand is not JAX-differentiable, and no fallback "
                    "method is available."
                )
        return _run_delta(h, adapter, config, estimand_metadata)

    elif method == "simulation":
        return _run_simulation(h, adapter, config, estimand_metadata)

    elif method == "bootstrap":
        return _run_bootstrap(h, adapter, config, estimand_metadata)

    else:
        raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Delta path
# ---------------------------------------------------------------------------

def _run_delta(h, adapter, config, estimand_metadata):
    """Delta-method inference with optional κ-based fallback to simulation."""
    beta = adapter.coefficients()
    Sigma = config.cov_params if config.cov_params is not None else adapter.covariance()

    estimate = h(beta)
    grad = gradient(h, beta, backend=config.gradient_backend, fd_step=config.fd_step)

    # Curvature diagnostic
    k = None
    fallback_triggered = False
    fallback_reason = None
    if config.diagnostics:
        if jnp.ndim(estimate) == 0:
            k = kappa(h, beta, Sigma,
                      backend=config.gradient_backend, fd_step=config.fd_step)
        else:
            k = kappa_vector(h, beta, Sigma,
                             backend=config.gradient_backend, fd_step=config.fd_step)

        max_k = float(k) if jnp.ndim(k) == 0 else float(jnp.max(jnp.asarray(k)))
        if max_k > config.kappa_threshold:
            # Auto-fallback to simulation
            sim_result = _run_simulation(
                h, adapter, config, estimand_metadata,
                fallback_reason=f"kappa={max_k:.3f}>threshold={config.kappa_threshold}",
            )
            sim_result["kappa"] = k
            sim_result["fallback_triggered"] = True
            return sim_result

    # Construct CI on inference scale, then back-transform via phi
    lower, upper = delta_confint(
        estimate, grad, Sigma,
        level=config.level, phi=config.phi,
    )
    se = delta_se(grad, Sigma)

    # Optional comparison against simulation
    delta_sim_disagreement = None
    if config.diagnostics and jnp.ndim(estimate) == 0:
        try:
            delta_sim_disagreement = delta_simulation_disagreement(
                estimate, grad, Sigma, h, beta,
                level=config.level,
                n_sim=min(config.n_sim, 1000),  # Smaller sample for diagnostic
                rng_seed=config.rng_seed,
                phi=config.phi,
            )
        except Exception:
            pass  # Diagnostic is best-effort

    estimate_report = config.phi(estimate) if config.phi is not None else estimate

    return {
        "estimate": np.asarray(estimate_report),
        "std_error": np.asarray(se),
        "conf_int_lower": np.asarray(lower),
        "conf_int_upper": np.asarray(upper),
        "method": "delta",
        "level": config.level,
        "kappa": np.asarray(k) if k is not None else None,
        "delta_sim_disagreement": delta_sim_disagreement,
        "fallback_triggered": False,
        "fallback_reason": None,
        "gradient": np.asarray(grad),
        "draws": None,
        "estimand_metadata": estimand_metadata or {},
    }


# ---------------------------------------------------------------------------
# Simulation path
# ---------------------------------------------------------------------------

def _run_simulation(h, adapter, config, estimand_metadata, *, fallback_reason=None):
    """Krinsky–Robb simulation: sample β̃ ~ N(β̂, Σ̂), evaluate h, take
    quantiles for CIs."""
    beta = adapter.coefficients()
    Sigma = config.cov_params if config.cov_params is not None else adapter.covariance()

    rng = np.random.default_rng(config.rng_seed)
    Sigma_np = np.asarray(Sigma)
    beta_np = np.asarray(beta)
    draws_beta = rng.multivariate_normal(beta_np, Sigma_np, size=config.n_sim)

    estimate = h(beta)
    h_draws = np.array([np.asarray(h(jnp.asarray(b))) for b in draws_beta])

    # Apply phi to draws and estimate for reporting
    if config.phi is not None:
        h_draws = np.asarray(config.phi(jnp.asarray(h_draws)))
        estimate_report = np.asarray(config.phi(estimate))
    else:
        estimate_report = np.asarray(estimate)

    alpha = (1.0 - config.level) / 2.0
    lower = np.quantile(h_draws, alpha, axis=0)
    upper = np.quantile(h_draws, 1.0 - alpha, axis=0)

    # SE on inference scale (before phi). Standard practice: report the SE
    # of the inference-scale draws, not the report-scale draws.
    if config.phi is not None:
        # Recompute draws on inference scale for SE
        h_draws_inf = np.array([np.asarray(h(jnp.asarray(b))) for b in draws_beta])
        se = np.std(h_draws_inf, axis=0, ddof=1)
    else:
        se = np.std(h_draws, axis=0, ddof=1)

    return {
        "estimate": estimate_report,
        "std_error": se,
        "conf_int_lower": lower,
        "conf_int_upper": upper,
        "method": "simulation",
        "level": config.level,
        "kappa": None,
        "delta_sim_disagreement": None,
        "fallback_triggered": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "gradient": None,
        "draws": h_draws,
        "estimand_metadata": estimand_metadata or {},
    }


# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------

def _run_bootstrap(h, adapter, config, estimand_metadata, *, fallback_reason=None):
    """Nonparametric bootstrap: refit the model on resampled data, recompute
    h, take quantiles.

    This path requires adapter.refit() to be implemented. The estimand
    function h is typically rebuilt on each refit to use the new adapter's
    predict. Bootstrap is the heaviest inference method; expect O(n_boot)
    refits, which can take minutes for nontrivial models.

    Implementation note for implementers: this function is a sketch. A
    complete implementation needs:
      - Data extraction from the adapter (training data and any indices for
        cluster bootstrap)
      - Resampling strategy (i.i.d., block, cluster) — currently i.i.d. only
      - Estimand reconstruction on each refit (since h depends on the
        adapter's predict, which changes between refits)
      - Parallelization across replicates (joblib or similar)

    The skeleton below illustrates the shape but does not implement the
    estimand-rebuilding logic. Concrete implementations of bootstrap belong
    in the engine that has access to the original estimand-construction
    pipeline.
    """
    raise NotImplementedError(
        "Bootstrap path requires session-level orchestration to rebuild the "
        "estimand against each refit. This function is a placeholder; "
        "the actual implementation lives in Margins._run_bootstrap, which "
        "has access to the scenario specification and can recompute h "
        "against each resampled adapter."
    )


# ---------------------------------------------------------------------------
# Hypothesis testing wrappers
# ---------------------------------------------------------------------------

def run_test(
    estimate: np.ndarray,
    grad: Optional[np.ndarray],
    cov_params: Optional[jnp.ndarray],
    draws: Optional[np.ndarray],
    *,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    method: str = "wald",
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis test on a result, dispatching by method.

    For delta-based results (gradient available), uses Wald test on the
    inference scale. For simulation-based results (draws available), uses
    the empirical distribution of draws to compute a tail probability.

    Parameters
    ----------
    estimate : array
        Point estimate(s) on the inference scale.

    grad : array, optional
        ∇h. Required for Wald tests on delta results.

    cov_params : jax array, optional
        Σ̂. Required for Wald tests.

    draws : array, optional
        Estimand draws. Used for simulation/bootstrap-based tests.

    null_value : float, default 0.0
        Hypothesized value on the inference scale.

    alternative : str
        "two-sided", "greater", or "less".

    method : str, default "wald"
        Test type. Currently only "wald" implemented.

    Returns
    -------
    (statistic, p_value) : tuple of arrays
    """
    if grad is not None and cov_params is not None:
        return delta_wald_test(
            jnp.asarray(estimate),
            jnp.asarray(grad),
            cov_params,
            null_value=null_value,
            alternative=alternative,
        )
    elif draws is not None:
        # Empirical p-value from draws
        if alternative == "two-sided":
            two_tail = 2.0 * np.minimum(
                np.mean(draws <= null_value, axis=0),
                np.mean(draws >= null_value, axis=0),
            )
            p = np.minimum(two_tail, 1.0)
        elif alternative == "greater":
            p = np.mean(draws <= null_value, axis=0)
        elif alternative == "less":
            p = np.mean(draws >= null_value, axis=0)
        else:
            raise ValueError(f"Unknown alternative: {alternative!r}")
        # No statistic in the Wald sense; return effect-size-like value
        statistic = np.mean(draws, axis=0) - null_value
        return statistic, np.asarray(p)
    else:
        raise ValueError(
            "Cannot run test: result has neither gradient nor draws."
        )


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Run delta-method inference on a prediction estimand
--------------------------------------------------------------

    from pymargins._inference import InferenceConfig, run_inference
    from pymargins._estimands import make_prediction_estimand

    config = InferenceConfig(
        method="delta",
        level=0.95,
        phi=None,
        phi_inv=None,
        kappa_threshold=0.3,
        gradient_backend="autodiff",
        diagnostics=True,
        cov_params=adapter.covariance(),
    )

    h = make_prediction_estimand(adapter, X, aggregate="overall")
    result = run_inference(h, adapter, config)
    # result is a dict with estimate, std_error, conf_int_lower/upper,
    # kappa, fallback_triggered, etc.


Example 2: Force simulation method
----------------------------------

    config = InferenceConfig(method="simulation", n_sim=4000, rng_seed=42, ...)
    result = run_inference(h, adapter, config)
    # result["draws"] is the array of simulated estimand values


Example 3: Run a test on a result
---------------------------------

    from pymargins._inference import run_test

    z, p = run_test(
        estimate=result["estimate"],
        grad=result["gradient"],
        cov_params=adapter.covariance(),
        draws=None,
        null_value=0.0,
        alternative="two-sided",
    )
"""
