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
from tqdm import tqdm
from .._gradients import gradient
from .._delta import delta_se
from .._kappa import kappa, kappa_vector
from .._estimands import is_jax_differentiable, split_kernel_partial, slope_kernel
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

def _compute_acceleration_jackknife(adapter, h_factory, h, data, cluster_ids, block_size):
    """Compute BCa acceleration via leave-one-out jackknife.

    Returns None when jackknife is deemed too expensive (n_obs or n_clusters
    > 200) or for block bootstrap where leave-one-out is not well-defined.
    """
    kernel_info = split_kernel_partial(h)
    static_kwargs = kernel_info["keywords"] if kernel_info is not None else None
    kernel_fn = kernel_info["kernel_fn"] if kernel_info is not None else None
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
            if kernel_fn is not None:
                h_probe = h_factory(new_adapter)
                probe_info = split_kernel_partial(h_probe)
                if probe_info is not None:
                    probe_kw = probe_info["keywords"]
                    data_independent = True
                    for key, val in static_kwargs.items():
                        if isinstance(val, (np.ndarray, jnp.ndarray)):
                            probe_val = probe_kw.get(key)
                            if not isinstance(probe_val, (np.ndarray, jnp.ndarray)):
                                data_independent = False
                                break
                            if val.shape != probe_val.shape or not np.allclose(val, probe_val):
                                data_independent = False
                                break
                    if data_independent:
                        theta_minus.append(np.asarray(kernel_fn(new_adapter.coefficients(), **static_kwargs)))
                    else:
                        theta_minus.append(np.asarray(h_probe(new_adapter.coefficients())))
                else:
                    theta_minus.append(np.asarray(h_probe(new_adapter.coefficients())))
            else:
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


def _compute_bca_params(h_draws_inf, estimate, adapter, h_factory, h, data,
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
            adapter, h_factory, h, data, cluster_ids, block_size,
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


def _try_fast_path(h, adapter, config):
    """Gate the batched fast path: it is eligible only when the user did not
    force ``engine="loop"``, the adapter's ``predict`` is a pure JAX array
    function (``supports_jax_autodiff``), and the estimand is a single
    module-level kernel partial.  Returns ``kernel_info`` or ``None``.

    Data-independent vs data-coupled classification and the per-replicate
    extraction happen in :func:`_run_fast_path`, after refitting.
    """
    bootstrap_config = config.bootstrap_config or {}
    if bootstrap_config.get("engine", "batched") == "loop":
        return None
    if not getattr(adapter, "supports_jax_autodiff", False):
        return None
    kernel_info = split_kernel_partial(h)
    if kernel_info is None:
        return None
    return kernel_info


# ---------------------------------------------------------------------------
# Array-tree helpers for the batched fast path
#
# Kernel partials bind a mix of arrays (``X``, ``Xp``/``Xm``, ``eps_jax``,
# ``weights``), lists of arrays (``scenarios_X``, ``offsets``, ``sw``),
# callables (``predict_fn``, ``transform``, ``compose``, ``phi_inv``) and
# strings (``aggregate``).  These helpers classify and stack the array-bearing
# leaves so the kernel can be vmapped over replicates.
# ---------------------------------------------------------------------------

def _is_arr(x):
    return isinstance(x, (np.ndarray, jnp.ndarray))


def _has_arr(x):
    """True if *x* is an array or a list/tuple containing one."""
    if _is_arr(x):
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_arr(e) for e in x)
    return False


def _arr_tree_equal(a, b):
    """Exact (bitwise) structural+value equality for an array-tree.

    Used for data-independence detection: a design is data-independent only
    if it is *byte-identical* across resamples, not merely close (a loose
    tolerance is exactly what risked a silent-wrong-results false positive).
    """
    if _is_arr(a) and _is_arr(b):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_arr_tree_equal(x, y) for x, y in zip(a, b))
    if _is_arr(a) or _is_arr(b):
        return False
    return a is None and b is None if (a is None or b is None) else True


def _arr_tree_sig(x):
    """Stackable-shape signature; mismatched signatures => ragged => no batch."""
    if _is_arr(x):
        return ("a", tuple(np.shape(x)))
    if isinstance(x, (list, tuple)):
        return ("l", tuple(_arr_tree_sig(e) for e in x))
    if x is None:
        return ("n",)
    return ("s",)


def _stack_arr_tree(items):
    """Stack a list of matching array-trees along a new leading (replicate) axis."""
    first = items[0]
    if _is_arr(first):
        return jnp.stack([jnp.asarray(t) for t in items])
    if isinstance(first, (list, tuple)):
        return [_stack_arr_tree([t[i] for t in items]) for i in range(len(first))]
    return first  # None / scalar: identical across replicates by construction


def _tree_in_axes(tree):
    """vmap in_axes pytree: map axis 0 of every array leaf, broadcast others."""
    if _is_arr(tree):
        return 0
    if isinstance(tree, list):
        return [_tree_in_axes(e) for e in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_in_axes(e) for e in tree)
    return None


def _tree_take(tree, sl):
    """Slice the leading axis of every array leaf in *tree*."""
    if _is_arr(tree):
        return tree[sl]
    if isinstance(tree, list):
        return [_tree_take(e, sl) for e in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_take(e, sl) for e in tree)
    return tree


def _chunked_batched(fn, betas, dyn_stacked, chunk_size):
    """vmap ``fn(beta, dyn)`` over the replicate axis in memory-bounded chunks.

    ``fn`` is jitted once; ``dyn_stacked`` is a (possibly empty) dict pytree
    whose array leaves have a leading replicate axis.  With no dynamic args
    this degenerates to ``vmap(fn)(betas)``.
    """
    n = betas.shape[0]
    if dyn_stacked:
        jitted = jax.jit(fn)
        in_axes = (0, {k: _tree_in_axes(v) for k, v in dyn_stacked.items()})
        out = []
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            dyn_c = {k: _tree_take(v, slice(s, e)) for k, v in dyn_stacked.items()}
            out.append(np.asarray(
                jax.vmap(jitted, in_axes=in_axes)(betas[s:e], dyn_c)
            ))
        return np.concatenate(out, axis=0)

    jitted = jax.jit(lambda beta: fn(beta, {}))
    out = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        out.append(np.asarray(jax.vmap(jitted, in_axes=(0,))(betas[s:e])))
    return np.concatenate(out, axis=0)


def _probe_indices(n, n_probe):
    """Evenly spaced unique indices into ``range(n)`` (always includes 0)."""
    if n <= n_probe:
        return list(range(n))
    return sorted(set(int(i) for i in np.linspace(0, n - 1, n_probe)))


def _split_static_dynamic(orig_kw, probe_kws):
    """Partition kernel kwargs into static vs replicate-varying.

    Non-array leaves (``predict_fn``, ``transform``, ``compose``, ``phi_inv``,
    ``aggregate``) are *always* static and taken from the original estimand:
    the ``supports_jax_autodiff`` gate guarantees ``predict`` is a pure array
    function (identical math across refits — often the same cached object),
    and the composition/scale callables are fixed by the estimand spec, not
    the data.  Array leaves are dynamic iff they are not byte-identical across
    the original and every probed replicate.
    """
    dynamic_keys = []
    for k, v in orig_kw.items():
        if not _has_arr(v):
            continue  # callable / str / None -> static from original
        if not all(_arr_tree_equal(v, pk.get(k)) for pk in probe_kws):
            dynamic_keys.append(k)
    static_kwargs = {k: v for k, v in orig_kw.items() if k not in dynamic_keys}
    return static_kwargs, dynamic_keys


def _run_fast_path(kernel_info, refitted, h_factory, config,
                   ci_method, jax_diffable, chunk_size):
    """Batched (JIT + chunked vmap) replacement for the per-replicate loop.

    Returns ``raw_results`` (one entry per replicate, in ``refitted`` order:
    an ``(theta_b, se_b)`` tuple for successes, the exception for failures)
    or ``None`` to signal the caller to use the legacy loop.  Any structural
    surprise (probe failure, non-kernel rebuild, ragged shapes, JAX error)
    returns ``None`` so the loop produces the authoritative result — the
    batched engine is never allowed to silently diverge.
    """
    kernel_fn = kernel_info["kernel_fn"]
    if kernel_info["args"]:
        return None  # unexpected positional binding; let the loop handle it

    failed = [(b, exc) for b, _, exc in refitted if exc is not None]
    successful = [(b, na) for b, na, exc in refitted if exc is None]
    if not successful:
        return None  # let the loop run the failure-threshold accounting

    orig_kw = dict(kernel_info["keywords"])

    # ---- robust data-independence probe: exact equality over k replicates --
    probe_cache = {}
    probe_kws = []
    for idx in _probe_indices(len(successful), 8):
        _, adp = successful[idx]
        try:
            info = split_kernel_partial(h_factory(adp))
        except Exception:
            return None
        if (info is None or info["kernel_fn"] is not kernel_fn
                or info["args"]):
            return None
        probe_cache[idx] = info
        probe_kws.append(info["keywords"])
    if any(set(pk) != set(orig_kw) for pk in probe_kws):
        return None

    static_kwargs, dynamic_keys = _split_static_dynamic(orig_kw, probe_kws)

    betas = []
    if not dynamic_keys:
        # Data-independent: design is byte-stable across resamples.  No
        # per-replicate h_factory rebuild (the W1 win); only beta varies.
        for _, adp in successful:
            betas.append(adp.coefficients())
        dyn_stacked = {}
    elif kernel_fn is slope_kernel:
        # Data-coupled slope: slope_kernel evaluates an internal ±eps finite
        # difference with eps≈1e-6.  Under JAX's default float32 the
        # (mu_p - mu_m) subtraction catastrophically cancels, and a batched
        # (XLA-fused) reduction over per-replicate Xp/Xm diverges from the
        # eager per-replicate loop well beyond the FP-identity tolerance
        # (observed ~6%).  The loop stays authoritative for this case.
        # (Data-*independent* slopes — fixed Xp/Xm — are exact and handled
        # by the branch above; only the data-coupled variant is excluded.)
        return None
    else:
        # Data-coupled: the design genuinely varies, so it must be rebuilt
        # per replicate (unavoidable, and exactly what the loop does).  Pull
        # the dynamic args straight from h_factory(new_adapter) so the result
        # is identical-by-construction to h_b(beta_b) — including matching's
        # rematch(), expand_scenario, and grid layout.
        dyn_lists = {k: [] for k in dynamic_keys}
        sig = None
        for i, (_b, adp) in enumerate(successful):
            info = probe_cache.get(i)
            if info is None:
                try:
                    info = split_kernel_partial(h_factory(adp))
                except Exception:
                    return None
                if (info is None or info["kernel_fn"] is not kernel_fn
                        or info["args"] or set(info["keywords"]) != set(orig_kw)):
                    return None
            kwb = info["keywords"]
            # Static array leaves must really be invariant; bail (to the
            # loop) rather than batch a wrong constant if a probe missed it.
            for k, v in static_kwargs.items():
                if _has_arr(v) and not _arr_tree_equal(v, kwb.get(k)):
                    return None
            cur = tuple(_arr_tree_sig(kwb[k]) for k in dynamic_keys)
            if sig is None:
                sig = cur
            elif cur != sig:
                return None  # ragged across replicates -> loop
            betas.append(adp.coefficients())
            for k in dynamic_keys:
                dyn_lists[k].append(kwb[k])
        dyn_stacked = {k: _stack_arr_tree(dyn_lists[k]) for k in dynamic_keys}

    betas = jnp.stack([jnp.asarray(b) for b in betas])

    def _value(beta, dyn):
        return kernel_fn(beta, **static_kwargs, **dyn)

    try:
        thetas = _chunked_batched(_value, betas, dyn_stacked, chunk_size)
    except (ValueError, TypeError, jax.errors.JAXTypeError):
        return None

    se_draws = None
    if ci_method == "studentized" and jax_diffable:
        is_vector = np.ndim(thetas) > 1
        gfn = jax.jacobian(kernel_fn, argnums=0) if is_vector else jax.grad(kernel_fn, argnums=0)

        def _grad(beta, dyn):
            return gfn(beta, **static_kwargs, **dyn)

        try:
            grads = _chunked_batched(_grad, betas, dyn_stacked, chunk_size)
            Sigma_stack = jnp.stack(
                [jnp.asarray(na.covariance()) for _, na in successful]
            )
            se_draws = np.asarray(
                jax.vmap(delta_se, in_axes=(0, 0))(jnp.asarray(grads), Sigma_stack)
            )
        except (ValueError, TypeError, jax.errors.JAXTypeError,
                np.linalg.LinAlgError):
            se_draws = None

    result_map = {}
    for i, (b, _) in enumerate(successful):
        se_b = se_draws[i] if (se_draws is not None) else None
        result_map[b] = (np.asarray(thetas[i]), se_b)
    for b, exc in failed:
        result_map[b] = exc
    return [result_map[b] for b, _, _ in refitted]


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

    _refit_iter = enumerate(all_idx)
    if config.progress_bar:
        _refit_iter = tqdm(_refit_iter, total=config.n_boot, desc="Bootstrap refit")

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
                    _refit_iter,
                    [adapter] * len(all_idx),
                    [data] * len(all_idx),
                    [config.matching] * len(all_idx),
                ))
        else:
            with threadpoolctl.threadpool_limits(limits=1):
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    refitted = list(executor.map(
                        _refit_replicate_task,
                        _refit_iter,
                        [adapter] * len(all_idx),
                        [data] * len(all_idx),
                        [config.matching] * len(all_idx),
                    ))
    else:
        refitted = list(map(
            _refit_replicate_task,
            _refit_iter,
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

    # ------------------------------------------------------------------
    # Fast path: JIT + chunked vmap over stacked replicates.
    #
    # Any eligible single-atom kernel partial is batched, at *any* scenario
    # (fixed atexog, at="typical" with free covariates, AME at="overall",
    # and AME under matching).  The evaluation design is either reused
    # (data-independent, detected by exact byte-equality over several
    # probed replicates) or rebuilt per replicate via the same h_factory
    # the loop uses (data-coupled) — so the batched result is identical-by-
    # construction to h_b(beta_b).  Any structural surprise returns None and
    # the legacy loop below produces the authoritative result.
    # ------------------------------------------------------------------
    raw_results = []
    fast_path = _try_fast_path(h, adapter, config)
    if fast_path is not None:
        bootstrap_config = config.bootstrap_config or {}
        chunk_size = bootstrap_config.get("vmap_chunk_size", 256)
        batched = _run_fast_path(
            fast_path, refitted, h_factory, config,
            ci_method, jax_diffable, chunk_size,
        )
        if batched is None:
            fast_path = None
        else:
            raw_results = batched

    if fast_path is None:
        # Legacy per-replicate loop: non-autodiff adapters, multi-atom
        # (over=/grid) estimands, non-kernel composes, or any structural
        # surprise the batched path declined. Authoritative result.
        _eval_iter = refitted
        if config.progress_bar:
            _eval_iter = tqdm(_eval_iter, total=config.n_boot, desc="Bootstrap evaluate")
        for b, new_adapter, refit_exc in _eval_iter:
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
            h_draws_inf, estimate, adapter, h_factory, h, data,
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
            # Reuse the jitted kernel for κ so the forward evaluation is
            # compiled, avoiding an extra eager trace.
            kernel_info_kappa = split_kernel_partial(h)
            if kernel_info_kappa is not None:
                kernel_fn_kappa = kernel_info_kappa["kernel_fn"]
                static_kwargs_kappa = kernel_info_kappa["keywords"]
                def h_jitted(beta):
                    return kernel_fn_kappa(beta, **static_kwargs_kappa)
                h_for_kappa = jax.jit(h_jitted)
            else:
                h_for_kappa = h

            if jnp.ndim(estimate) == 0:
                k = kappa(h_for_kappa, beta_hat, Sigma_hat,
                          backend=config.gradient_backend, fd_step=config.fd_step)
            else:
                k = kappa_vector(h_for_kappa, beta_hat, Sigma_hat,
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
