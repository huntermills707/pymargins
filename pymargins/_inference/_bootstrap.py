"""Bootstrap inference implementation.

Intentional broad exceptions
----------------------------
Two ``except Exception`` clauses are retained by design:

- ``_refit_replicate_task`` (line ~160): ``matching.rematch()`` can raise
  arbitrary exceptions from user-supplied matching objects.
- ``_refit_replicate_task`` (line ~176): bootstrap resampling can trigger
  many model-fitting failure modes (perfect separation, missing data,
  singular Hessians, convergence failures). These are expected and counted
  against the 10 % failure threshold. Serious errors (``AssertionError``,
  ``MemoryError``, ``RecursionError``, ``KeyboardInterrupt``) propagate
  immediately so they are not silently lost.

Both clauses are documented inline; this module-level note exists so that
static-analysis audits do not re-flag them.
"""

from __future__ import annotations
from typing import Optional
from functools import partial
import warnings
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
import threadpoolctl
from .._gradients import gradient
from .._delta import delta_se
from .._kappa import kappa, kappa_vector
from .._estimands import is_jax_differentiable
from ._config import InferenceConfig


# ---------------------------------------------------------------------------
# Resample-index generation (extracted for session-level bank reuse)
# ---------------------------------------------------------------------------

def _generate_resample_indices(
    rng_seed,
    n_boot,
    n_obs,
    cluster_ids=None,
    block_size=None,
    block_type="moving",
):
    """Generate the list of resample index arrays for bootstrap.

    Deterministic given the parameters — same inputs always produce the same
    ``all_idx``.  This determinism is what makes session-level resample banks
    possible.
    """
    rng = np.random.default_rng(
        [rng_seed, 1] if rng_seed is not None else None
    )

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

    all_idx = []
    for _ in range(n_boot):
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

    return all_idx


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

    # Use pre-generated resample bank if provided (session-level composition support)
    if config.all_idx is not None:
        all_idx = config.all_idx
    else:
        all_idx = _generate_resample_indices(
            rng_seed=config.rng_seed,
            n_boot=config.n_boot,
            n_obs=n_obs,
            cluster_ids=cluster_ids,
            block_size=block_size,
            block_type=block_type,
        )

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
