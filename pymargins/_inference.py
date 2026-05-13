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
from functools import partial
import warnings
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
import threadpoolctl

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

    cluster : array-like, optional
        Cluster IDs for cluster bootstrap. When provided, bootstrap resamples
        clusters with replacement instead of individual rows.

    block_size : int, optional
        Block length for block bootstrap. When provided, bootstrap resamples
        contiguous blocks with replacement instead of individual rows.
        Mutually exclusive with ``cluster``.

    n_jobs : int, default 1
        Number of parallel workers for bootstrap refits. ``n_jobs=-1`` uses
        all available cores. Parallelization is thread-based; BLAS threads
        are capped to 1 per worker via ``threadpoolctl`` to prevent
        oversubscription.

    bootstrap_config : dict, optional
        Advanced bootstrap options. Supported keys:
          - "block_type": "moving" (default), "nonoverlapping", or "circular"
          - "ci_method": "percentile" (default), "basic", "bca", "studentized"
          - "acceleration": float or array, BCa acceleration parameter (a)
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
    n_jobs: int = 1
    rng_seed: Optional[int] = None
    diagnostics: bool = True
    cov_params: Optional[jnp.ndarray] = None
    cluster: Optional[Any] = None
    block_size: Optional[int] = None
    matching: Optional[Any] = None
    """MatchingClient for bootstrap rematching. Ignored by delta and K–R."""

    bootstrap_config: Optional[dict] = None


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------

def run_inference(
    h: Callable,
    adapter: ModelAdapter,
    config: InferenceConfig,
    *,
    estimand_metadata: Optional[dict] = None,
    h_factory: Optional[Callable[[ModelAdapter], Callable]] = None,
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
        beta = adapter.coefficients()
        if not is_jax_differentiable(h, beta):
            if "simulation" in supported:
                # Auto-route to simulation with a warning marker in the result
                warnings.warn(
                    "Estimand is not JAX-differentiable; falling back to simulation.",
                    UserWarning, stacklevel=3,
                )
                return _run_simulation(h, adapter, config, estimand_metadata,
                                       fallback_reason="non_differentiable")
            elif "bootstrap" in supported and h_factory is not None:
                warnings.warn(
                    "Estimand is not JAX-differentiable; falling back to bootstrap.",
                    UserWarning, stacklevel=3,
                )
                return _run_bootstrap(h, adapter, config, estimand_metadata,
                                      fallback_reason="non_differentiable",
                                      h_factory=h_factory)
            else:
                raise ValueError(
                    "Estimand is not JAX-differentiable, and no fallback "
                    "method is available."
                )
        return _run_delta(h, adapter, config, estimand_metadata)

    elif method == "simulation":
        return _run_simulation(h, adapter, config, estimand_metadata)

    elif method == "bootstrap":
        return _run_bootstrap(h, adapter, config, estimand_metadata, h_factory=h_factory)

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

        max_k = float(k) if jnp.ndim(k) == 0 else float(jnp.nanmax(jnp.asarray(k)))
        if max_k > config.kappa_threshold:
            # Auto-fallback to simulation
            warnings.warn(
                f"Delta-method curvature κ={max_k:.3f} exceeds threshold "
                f"({config.kappa_threshold}); falling back to simulation.",
                UserWarning, stacklevel=3,
            )
            sim_result = _run_simulation(
                h, adapter, config, estimand_metadata,
                fallback_reason=f"kappa={max_k:.3f}>threshold={config.kappa_threshold}",
                skip_kappa=True,
            )
            sim_result["kappa"] = np.asarray(k) if k is not None else None
            return sim_result

    # Construct CI on inference scale, then back-transform via phi
    lower, upper = delta_confint(
        estimate, grad, Sigma,
        level=config.level, phi=config.phi,
    )
    se = delta_se(grad, Sigma)

    # Optional comparison against simulation. Works for scalar and vector
    # estimands; for vectors it returns the max per-component disagreement.
    delta_sim_disagreement = None
    if config.diagnostics:
        try:
            delta_sim_disagreement = delta_simulation_disagreement(
                estimate, grad, Sigma, h, beta,
                level=config.level,
                n_sim=min(config.n_sim, 1000),  # Smaller sample for diagnostic
                rng_seed=config.rng_seed,
                phi=config.phi,
            )
        except (ValueError, TypeError, jax.errors.JAXTypeError) as exc:
            warnings.warn(f"delta-simulation diagnostic failed: {exc}", RuntimeWarning)

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
        "draws_inf": None,
        "estimand_metadata": estimand_metadata or {},
        "ci_method": None,
        "bootstrap_extras": None,
    }


# ---------------------------------------------------------------------------
# Simulation path
# ---------------------------------------------------------------------------

def _run_simulation(h, adapter, config, estimand_metadata, *, fallback_reason=None, skip_kappa=False):
    """Krinsky–Robb simulation: sample β̃ ~ N(β̂, Σ̂), evaluate h, take
    quantiles for CIs."""
    beta = adapter.coefficients()
    Sigma = config.cov_params if config.cov_params is not None else adapter.covariance()

    rng = np.random.default_rng(config.rng_seed)
    Sigma_np = np.asarray(Sigma)
    beta_np = np.asarray(beta)
    eigvals = np.linalg.eigvalsh(Sigma_np)
    if np.any(eigvals < -1e-8):
        raise ValueError(
            "Covariance matrix (vcov) is not positive semi-definite. "
            "Simulation draws would produce NaN. "
            f"Minimum eigenvalue: {np.min(eigvals):.3e}. "
            "Check your vcov specification or use a different inference method."
        )
    draws_beta = rng.multivariate_normal(beta_np, Sigma_np, size=config.n_sim)
    if not np.all(np.isfinite(draws_beta)):
        raise ValueError(
            "Simulation draws contain NaN or Inf. The covariance matrix may be "
            "singular or numerically unstable."
        )

    estimate = h(beta)
    try:
        h_draws_inf = np.asarray(jax.vmap(h)(jnp.asarray(draws_beta)))
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerBoolConversionError,
        jax.errors.TracerIntegerConversionError,
        jax.errors.UnexpectedTracerError,
    ):
        # vmap couldn't trace h; fall back to a Python loop. Genuine
        # shape/type bugs in h (TypeError/ValueError) are intentionally not
        # caught — they surface immediately with their original traceback.
        h_draws_inf = np.array([np.asarray(h(np.asarray(b))) for b in draws_beta])

    se = np.std(h_draws_inf, axis=0, ddof=1)

    # Curvature diagnostic. Gated on JAX-differentiability because explicit
    # method="simulation" is how non-differentiable estimands reach the engine.
    # skip_kappa=True avoids redundant recomputation when the delta path already
    # computed κ and is falling back to simulation.
    k = None
    if not skip_kappa and config.diagnostics and is_jax_differentiable(h, beta):
        try:
            if jnp.ndim(estimate) == 0:
                k = kappa(h, beta, Sigma,
                          backend=config.gradient_backend, fd_step=config.fd_step)
            else:
                k = kappa_vector(h, beta, Sigma,
                                 backend=config.gradient_backend, fd_step=config.fd_step)
        except (ValueError, TypeError, jax.errors.JAXTypeError) as exc:
            warnings.warn(f"kappa diagnostic failed: {exc}", RuntimeWarning)

    # Apply phi to draws and estimate for reporting
    if config.phi is not None:
        try:
            h_draws = np.asarray(config.phi(jnp.asarray(h_draws_inf)))
        except (TypeError, ValueError):
            h_draws = np.asarray(config.phi(np.asarray(h_draws_inf)))
        try:
            estimate_report = np.asarray(config.phi(estimate))
        except (TypeError, ValueError):
            estimate_report = np.asarray(config.phi(np.asarray(estimate)))
    else:
        h_draws = h_draws_inf
        estimate_report = np.asarray(estimate)

    alpha = (1.0 - config.level) / 2.0
    lower = np.quantile(h_draws, alpha, axis=0)
    upper = np.quantile(h_draws, 1.0 - alpha, axis=0)

    return {
        "estimate": estimate_report,
        "std_error": se,
        "conf_int_lower": lower,
        "conf_int_upper": upper,
        "method": "simulation",
        "level": config.level,
        "kappa": np.asarray(k) if k is not None else None,
        "delta_sim_disagreement": None,
        "fallback_triggered": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "gradient": None,
        "draws": h_draws,
        "draws_inf": h_draws_inf,
        "estimand_metadata": estimand_metadata or {},
        "ci_method": None,
        "bootstrap_extras": None,
    }


# ---------------------------------------------------------------------------
# BCa helpers
# ---------------------------------------------------------------------------

def _compute_acceleration_jackknife(adapter, h_factory, data, cluster_ids, block_size):
    """Compute BCa acceleration via leave-one-out jackknife.

    Returns None when jackknife is deemed too expensive (n_obs or n_clusters
    > 200) or for block bootstrap where leave-one-out is not well-defined.
    """
    n_obs = len(data)
    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        n_units = len(unique_clusters)
        if n_units > 200:
            return None
        theta_minus = []
        for c in unique_clusters:
            mask = cluster_ids != c
            if hasattr(data, "iloc"):
                resampled = data.iloc[mask]
            else:
                resampled = data[mask]
            new_adapter = adapter.refit(resampled, index=np.where(mask)[0])
            h_b = h_factory(new_adapter)
            theta_minus.append(np.asarray(h_b(new_adapter.coefficients())))
    elif block_size is not None:
        return None
    else:
        if n_obs > 200:
            return None
        theta_minus = []
        for i in range(n_obs):
            mask = np.ones(n_obs, dtype=bool)
            mask[i] = False
            if hasattr(data, "iloc"):
                resampled = data.iloc[mask]
            else:
                resampled = data[mask]
            new_adapter = adapter.refit(resampled, index=np.where(mask)[0])
            h_b = h_factory(new_adapter)
            theta_minus.append(np.asarray(h_b(new_adapter.coefficients())))

    theta_minus = np.stack(theta_minus, axis=0)  # shape (n_units, n_components)
    theta_mean = np.mean(theta_minus, axis=0)
    diffs = theta_mean - theta_minus  # shape (n_units, n_components)
    num = np.sum(diffs ** 3, axis=0)
    den = np.sum(diffs ** 2, axis=0)
    a = np.where(den > 0, num / (6.0 * den ** 1.5), 0.0)
    return a


def _compute_bca_params(h_draws_inf, estimate, adapter, h_factory, data,
                        cluster_ids, block_size, bootstrap_config):
    """Compute BCa bias correction (z0) and acceleration (a)."""
    if not np.all(np.isfinite(estimate)):
        raise ValueError(
            "BCa confidence intervals require a finite estimate, "
            f"got non-finite value(s): {estimate}"
        )
    prop = np.mean(h_draws_inf < np.asarray(estimate), axis=0)
    prop = np.clip(prop, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop)

    a = None
    if bootstrap_config is not None and "acceleration" in bootstrap_config:
        a = np.asarray(bootstrap_config["acceleration"])
    else:
        a = _compute_acceleration_jackknife(
            adapter, h_factory, data, cluster_ids, block_size,
        )
        if a is None:
            warnings.warn(
                "BCa acceleration could not be computed automatically "
                "(jackknife is too expensive for this data size). "
                "Using a=0 (BC, bias-corrected only). Pass "
                "bootstrap_config={'acceleration': value} to supply a "
                "custom acceleration.",
                UserWarning, stacklevel=4,
            )
    return z0, a


def _bca_confint(h_draws_inf, estimate, level, z0, a, phi):
    """Compute BCa confidence interval."""
    alpha = (1.0 - level) / 2.0
    z_alpha = stats.norm.ppf(alpha)
    z_1_alpha = stats.norm.ppf(1.0 - alpha)
    estimate_arr = np.asarray(estimate)

    if a is not None:
        a_arr = np.asarray(a)
        a1 = z0 + z_alpha
        a2 = z0 + z_1_alpha
        denom1 = 1 - a_arr * a1
        denom2 = 1 - a_arr * a2
        b1 = np.where(
            denom1 > 0,
            stats.norm.cdf(z0 + a1 / denom1),
            stats.norm.cdf(2 * z0 + z_alpha),
        )
        b2 = np.where(
            denom2 > 0,
            stats.norm.cdf(z0 + a2 / denom2),
            stats.norm.cdf(2 * z0 + z_1_alpha),
        )
    else:
        b1 = stats.norm.cdf(2 * z0 + z_alpha)
        b2 = stats.norm.cdf(2 * z0 + z_1_alpha)

    lower_inf = np.quantile(h_draws_inf, b1, axis=0)
    upper_inf = np.quantile(h_draws_inf, b2, axis=0)

    if phi is not None:
        lower = np.asarray(phi(lower_inf))
        upper = np.asarray(phi(upper_inf))
    else:
        lower, upper = lower_inf, upper_inf
    return lower, upper


# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------

def _refit_replicate_task(args, adapter, data, matching=None):
    """Module-level helper for bootstrap refit parallelism.

    Must be defined at module level so it can be pickled for
    ProcessPoolExecutor.
    """
    b, idx = args
    if hasattr(data, "iloc"):
        resampled = data.iloc[idx]
    else:
        resampled = data[idx]

    # Rematch after resampling when a matching object is provided
    if matching is not None:
        try:
            resampled = matching.rematch(resampled)
        except Exception as exc:
            if isinstance(exc, (AssertionError, MemoryError, RecursionError, KeyboardInterrupt)):
                raise
            return b, None, exc
        index = None   # rematching breaks the original index mapping
    else:
        index = idx

    try:
        new_adapter = adapter.refit(resampled, index=index)
        if len(new_adapter.coefficients()) != len(adapter.coefficients()):
            raise ValueError(
                f"Parameter count mismatch after refit: "
                f"{len(new_adapter.coefficients())} vs {len(adapter.coefficients())}"
            )
        return b, new_adapter, None
    except Exception as exc:
        # Broad catch is intentional: bootstrap resampling can trigger
        # many model-fitting failure modes (perfect separation, missing
        # data, singular Hessians, convergence failures). These are
        # expected and counted against the 10% failure threshold.
        # Serious errors (assertions, memory exhaustion, recursion limits,
        # interrupts) propagate immediately so they are not silently lost.
        if isinstance(exc, (AssertionError, MemoryError, RecursionError, KeyboardInterrupt)):
            raise
        return b, None, exc


def _run_bootstrap(h, adapter, config, estimand_metadata, *, fallback_reason=None, h_factory=None):
    """Nonparametric bootstrap: refit the model on resampled data, recompute
    h, take quantiles.

    This path requires ``adapter.refit()`` and ``adapter.training_data`` to
    be implemented. On each bootstrap replicate the training data is
    resampled with replacement, the model is refit, the estimand is rebuilt
    via ``h_factory(new_adapter)``, and the estimand is evaluated at the new
    coefficients.

    Supports parallel execution via ``config.n_jobs`` (thread-based, with
    BLAS threads limited to 1 per worker to avoid oversubscription).
    Supports alternative CI methods: percentile (default), basic, bca,
    and studentized.

    Failure handling
    ----------------
    Replicates that fail to refit (e.g. perfect separation, convergence
    failure, singular Hessian) are **silently dropped** and counted against
    a 10 % tolerance threshold. If more than 10 % of replicates fail, a
    ``UserWarning`` is raised and the CI is computed from the successful
    replicates only. Serious errors (``AssertionError``, ``MemoryError``,
    ``RecursionError``, ``KeyboardInterrupt``) propagate immediately.
    """
    if h_factory is None:
        raise ValueError(
            "Bootstrap inference requires h_factory. "
            "The estimand must be rebuilt for each resampled model."
        )
    # Extract training data
    try:
        data = adapter.training_data
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Bootstrap inference requires the adapter to expose training_data. "
            f"{type(adapter).__name__} does not implement it."
        ) from exc

    # When matching is active, resample from matched_data and use its cluster_ids
    if config.matching is not None:
        data = config.matching.matched_data
        cluster_ids = config.matching.cluster_ids
    else:
        cluster_ids = config.cluster

    data = np.asarray(data) if not hasattr(data, "iloc") else data
    n_obs = len(data)

    # Prepare resampling strategy
    block_size = config.block_size
    bootstrap_config = config.bootstrap_config or {}
    block_type = bootstrap_config.get("block_type", "moving")
    ci_method = bootstrap_config.get("ci_method", "percentile")
    if ci_method not in ("percentile", "basic", "bca", "studentized"):
        raise ValueError(
            f"Unsupported bootstrap ci_method: {ci_method!r}. "
            f"Supported: 'percentile', 'basic', 'bca', 'studentized'."
        )

    if cluster_ids is not None and block_size is not None:
        raise ValueError(
            "cluster and block_size are mutually exclusive. "
            "Use cluster for cluster bootstrap or block_size for block bootstrap, not both."
        )

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if len(cluster_ids) != n_obs:
            raise ValueError(
                f"cluster IDs length ({len(cluster_ids)}) must match "
                f"training data length ({n_obs})."
            )
        if np.any(pd.isna(cluster_ids)):
            raise ValueError("cluster IDs must not contain NaN values.")
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        if n_clusters == 0:
            raise ValueError("cluster IDs must not be empty.")

    if block_size is not None:
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        if block_size > n_obs:
            raise ValueError(
                f"block_size ({block_size}) cannot exceed training data length ({n_obs})."
            )
        if block_type not in ("moving", "nonoverlapping", "circular"):
            raise ValueError(
                f"Unsupported block_type: {block_type!r}. "
                f"Supported: 'moving', 'nonoverlapping', 'circular'."
            )

    # NOTE: bootstrap and simulation paths intentionally share the same
    # rng_seed source so that reproducibility is straightforward for users.
    # If you need independent streams when comparing methods, supply
    # distinct rng_seed values to each call.
    rng = np.random.default_rng(config.rng_seed)

    # Pre-generate all resampling indices
    all_idx = []
    for _ in range(config.n_boot):
        if cluster_ids is not None:
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            idx = np.concatenate([
                np.where(cluster_ids == c)[0]
                for c in sampled_clusters
            ])
        elif block_size is not None:
            k = int(np.ceil(n_obs / block_size))
            if block_type == "moving":
                start_positions = rng.integers(0, n_obs - block_size + 1, size=k)
                idx = np.concatenate([
                    np.arange(s, s + block_size)
                    for s in start_positions
                ])
            elif block_type == "circular":
                start_positions = rng.integers(0, n_obs, size=k)
                idx = np.concatenate([
                    np.arange(s, s + block_size) % n_obs
                    for s in start_positions
                ])
            else:  # nonoverlapping
                n_blocks = int(np.ceil(n_obs / block_size))
                if n_blocks == 0:
                    raise ValueError(
                        f"block_size ({block_size}) too large for n_obs ({n_obs})."
                    )
                sampled_blocks = rng.integers(0, n_blocks, size=n_blocks)
                idx = np.concatenate([
                    np.arange(bi * block_size, (bi + 1) * block_size) % n_obs
                    for bi in sampled_blocks
                ])
        else:
            idx = rng.integers(0, n_obs, size=n_obs)
        all_idx.append(idx)

    # Pre-compute JAX differentiability once to avoid per-replicate
    # recompilation from is_jax_differentiable probes.
    jax_diffable = None
    if ci_method == "studentized":
        beta_hat = adapter.coefficients()
        jax_diffable = is_jax_differentiable(h, beta_hat)

    # Step 1: Parallel refit (pure statsmodels/numpy, no JAX).
    n_jobs = config.n_jobs
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1:
        warnings.warn(
            f"Parallel bootstrap (n_jobs={n_jobs}) is experimental. "
            "Process-based execution is used when pickling succeeds; "
            "otherwise it falls back to thread-based execution.",
            RuntimeWarning,
            stacklevel=3,
        )

    if n_jobs != 1:
        import pickle
        try:
            pickle.dumps(adapter)
            pickle.dumps(data)
            if config.matching is not None:
                pickle.dumps(config.matching)
            use_processes = True
        except (TypeError, pickle.PicklingError):
            warnings.warn(
                "Adapter, data, or matching object cannot be pickled; falling back "
                "to thread-based parallel bootstrap. Consider setting n_jobs=1 "
                "for stability.",
                RuntimeWarning,
                stacklevel=3,
            )
            use_processes = False

        if use_processes:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            # Use 'spawn' to avoid deadlocks with JAX's background threads.
            _ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_ctx) as executor:
                refitted = list(executor.map(
                    _refit_replicate_task,
                    enumerate(all_idx),
                    [adapter] * len(all_idx),
                    [data] * len(all_idx),
                    [config.matching] * len(all_idx),
                ))
        else:
            with threadpoolctl.threadpool_limits(limits=1):
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    refitted = list(executor.map(
                        _refit_replicate_task,
                        enumerate(all_idx),
                        [adapter] * len(all_idx),
                        [data] * len(all_idx),
                        [config.matching] * len(all_idx),
                    ))
    else:
        refitted = list(map(
            _refit_replicate_task,
            enumerate(all_idx),
            [adapter] * len(all_idx),
            [data] * len(all_idx),
            [config.matching] * len(all_idx),
        ))

    # Step 2: Serial JAX evaluation (thread-safe, compilation cache friendly).
    # For studentized bootstrap, if the estimand is a module-level kernel
    # partial, pre-build a stable gradient function so JAX can cache the
    # compilation across replicates (see CODE_AUDIT §5.1).
    _kernel_grad_fn = None
    if ci_method == "studentized" and jax_diffable:
        h_probe = h_factory(adapter)
        if isinstance(h_probe, partial) and getattr(h_probe.func, "__pymargins_kernel__", False):
            _kernel_grad_fn = jax.grad(h_probe.func, argnums=0)

    raw_results = []
    for b, new_adapter, refit_exc in refitted:
        if refit_exc is not None:
            raw_results.append(refit_exc)
            continue
        try:
            h_b = h_factory(new_adapter)
            beta_b = new_adapter.coefficients()
            theta_b = np.asarray(h_b(beta_b))

            se_b = None
            if ci_method == "studentized" and jax_diffable:
                try:
                    Sigma_b = new_adapter.covariance()
                    if _kernel_grad_fn is not None and isinstance(h_b, partial):
                        grad_b = _kernel_grad_fn(beta_b, *h_b.args, **h_b.keywords)
                    else:
                        grad_b = gradient(
                            h_b, beta_b,
                            backend=config.gradient_backend,
                            fd_step=config.fd_step,
                        )
                    se_b = np.asarray(delta_se(grad_b, Sigma_b))
                except (ValueError, TypeError, jax.errors.JAXTypeError, np.linalg.LinAlgError):
                    se_b = None

            raw_results.append((theta_b, se_b))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raw_results.append(exc)

    # Collect results
    h_draws_inf = []
    se_draws = []
    n_failures = 0
    max_failures = max(1, int(0.1 * config.n_boot))
    failure_exceptions = []
    for result in raw_results:
        if isinstance(result, Exception):
            n_failures += 1
            failure_exceptions.append(result)
        else:
            theta_b, se_b = result
            h_draws_inf.append(theta_b)
            if ci_method == "studentized":
                se_draws.append(se_b)

    if n_failures > max_failures:
        last_exc = failure_exceptions[-1] if failure_exceptions else None
        raise RuntimeError(
            f"Bootstrap failed on {n_failures} replicates (>{max_failures} "
            f"threshold)."
        ) from last_exc

    if n_failures > 0:
        warnings.warn(
            f"Bootstrap: {n_failures} of {config.n_boot} replicates failed "
            f"({n_failures / config.n_boot:.1%}). CI computed from "
            f"{len(h_draws_inf)} successful replicates.",
            UserWarning,
            stacklevel=3,
        )

    if len(h_draws_inf) == 0:
        raise RuntimeError("All bootstrap replicates failed.")

    h_draws_inf = np.stack(h_draws_inf, axis=0)  # shape (n_boot, ...)

    estimate = h(adapter.coefficients())

    # Apply phi to draws and estimate for reporting
    if config.phi is not None:
        try:
            h_draws = np.asarray(config.phi(jnp.asarray(h_draws_inf)))
        except (TypeError, ValueError):
            h_draws = np.asarray(config.phi(np.asarray(h_draws_inf)))
        try:
            estimate_report = np.asarray(config.phi(estimate))
        except (TypeError, ValueError):
            estimate_report = np.asarray(config.phi(np.asarray(estimate)))
    else:
        h_draws = h_draws_inf
        estimate_report = np.asarray(estimate)

    # Compute CIs
    alpha = (1.0 - config.level) / 2.0
    lower = upper = None
    bca_z0 = None
    bca_a = None
    studentized_t = None
    se_hat = None

    if ci_method == "percentile":
        lower = np.quantile(h_draws, alpha, axis=0)
        upper = np.quantile(h_draws, 1.0 - alpha, axis=0)

    elif ci_method == "basic":
        if config.phi is not None:
            raise ValueError(
                "basic bootstrap CI method is not supported with non-identity "
                "phi transforms because the basic bootstrap formula is not "
                "equivariant under non-linear transforms. Use ci_method='percentile' "
                "or 'bca' instead."
            )
        lower = 2 * np.asarray(estimate) - np.quantile(h_draws_inf, 1.0 - alpha, axis=0)
        upper = 2 * np.asarray(estimate) - np.quantile(h_draws_inf, alpha, axis=0)

    elif ci_method == "bca":
        bca_z0, bca_a = _compute_bca_params(
            h_draws_inf, estimate, adapter, h_factory, data,
            cluster_ids, block_size, bootstrap_config,
        )
        lower, upper = _bca_confint(
            h_draws_inf, estimate, config.level, bca_z0, bca_a, config.phi,
        )

    elif ci_method == "studentized":
        if config.phi is not None:
            raise ValueError(
                "studentized bootstrap CI method is not supported with non-identity "
                "phi transforms because the studentized bootstrap formula is not "
                "equivariant under non-linear transforms. Use ci_method='percentile' "
                "or 'bca' instead."
            )
        valid = [i for i, s in enumerate(se_draws) if s is not None]
        if len(valid) == 0:
            raise RuntimeError(
                "Studentized bootstrap failed: no replicates had valid SE estimates. "
                "This typically means the estimand is not JAX-differentiable or the "
                "adapter does not provide a covariance matrix."
            )
        valid_draws = h_draws_inf[valid]
        valid_se = np.stack([se_draws[i] for i in valid], axis=0)
        studentized_t = (valid_draws - np.asarray(estimate)) / valid_se

        beta_hat = adapter.coefficients()
        Sigma_hat = config.cov_params if config.cov_params is not None else adapter.covariance()
        if is_jax_differentiable(h, beta_hat):
            try:
                grad_hat = gradient(
                    h, beta_hat,
                    backend=config.gradient_backend,
                    fd_step=config.fd_step,
                )
                se_hat = np.asarray(delta_se(grad_hat, Sigma_hat))
            except (ValueError, TypeError, jax.errors.JAXTypeError, np.linalg.LinAlgError):
                se_hat = np.std(h_draws_inf, axis=0, ddof=1)
        else:
            se_hat = np.std(h_draws_inf, axis=0, ddof=1)

        t_lower = np.quantile(studentized_t, alpha, axis=0)
        t_upper = np.quantile(studentized_t, 1.0 - alpha, axis=0)
        lower = np.asarray(estimate) - t_upper * se_hat
        upper = np.asarray(estimate) - t_lower * se_hat

    se = np.std(h_draws_inf, axis=0, ddof=1)

    # κ at β̂ when h is JAX-differentiable
    k = None
    beta_hat = adapter.coefficients()
    Sigma_hat = config.cov_params if config.cov_params is not None else adapter.covariance()
    if config.diagnostics and is_jax_differentiable(h, beta_hat):
        try:
            if jnp.ndim(estimate) == 0:
                k = kappa(h, beta_hat, Sigma_hat,
                          backend=config.gradient_backend, fd_step=config.fd_step)
            else:
                k = kappa_vector(h, beta_hat, Sigma_hat,
                                 backend=config.gradient_backend, fd_step=config.fd_step)
        except (ValueError, TypeError, jax.errors.JAXTypeError) as exc:
            warnings.warn(f"Bootstrap kappa diagnostic failed: {exc}", RuntimeWarning)

    bootstrap_extras = {}
    if bca_z0 is not None:
        bootstrap_extras["z0"] = np.asarray(bca_z0)
    if bca_a is not None:
        bootstrap_extras["a"] = np.asarray(bca_a)
    if studentized_t is not None:
        bootstrap_extras["t_star"] = np.asarray(studentized_t)
    if se_hat is not None:
        bootstrap_extras["se_hat"] = np.asarray(se_hat)
    if not bootstrap_extras:
        bootstrap_extras = None

    return {
        "estimate": estimate_report,
        "std_error": se,
        "conf_int_lower": lower,
        "conf_int_upper": upper,
        "method": "bootstrap",
        "level": config.level,
        "kappa": np.asarray(k) if k is not None else None,
        "delta_sim_disagreement": None,
        "fallback_triggered": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "gradient": None,
        "draws": h_draws,
        "draws_inf": h_draws_inf,
        "estimand_metadata": estimand_metadata or {},
        "ci_method": ci_method,
        "bootstrap_extras": bootstrap_extras,
    }


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
    if method != "wald":
        raise NotImplementedError(f"Test method {method!r} is not implemented.")

    if grad is not None and cov_params is not None:
        return delta_wald_test(
            jnp.asarray(estimate),
            jnp.asarray(grad),
            cov_params,
            null_value=null_value,
            alternative=alternative,
        )
    elif draws is not None:
        # Empirical p-value from draws: compare observed estimate to the
        # simulated sampling distribution.
        estimate = np.asarray(estimate)
        null_value = np.asarray(null_value)
        draws = np.asarray(draws)
        if alternative == "two-sided":
            # Two-sided p-value via 2*min(tail probabilities).
            # Correct for asymmetric simulation distributions.
            p_left = np.mean(draws <= null_value, axis=0)
            p_right = np.mean(draws >= null_value, axis=0)
            p = np.clip(2.0 * np.minimum(p_left, p_right), min=None, max=1.0)
        elif alternative == "greater":
            # H1: effect > null. Under H0, draws are centered at estimate;
            # the null lies to the left. Small p when null is in the lower
            # tail of the simulated distribution.
            p = np.mean(draws <= null_value, axis=0)
        elif alternative == "less":
            # H1: effect < null. Small p when null is in the upper tail.
            p = np.mean(draws >= null_value, axis=0)
        else:
            raise ValueError(f"Unknown alternative: {alternative!r}")
        # Return the observed estimate minus null as the effect-size statistic
        statistic = estimate - null_value
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
