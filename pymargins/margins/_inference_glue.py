"""Inference configuration and result wrapping helpers."""

from __future__ import annotations

import hashlib
import weakref

import jax.numpy as jnp
import numpy as np

from .._inference import InferenceConfig
from .._result import MarginsResult


def _check_adapter_drift(session):
    """Snapshot β̂ at first materialization and check drift on subsequent calls."""
    cached = session.__dict__.get("_adapter_beta_snapshot")
    current = np.asarray(session.adapter.coefficients())
    if cached is None:
        session._adapter_beta_snapshot = current
        return
    # NaN-aware equality: NaN == NaN is True here so rank-deficient fits
    # with dropped-term NaNs do not falsely trip drift detection.
    if not np.allclose(cached, current, equal_nan=True):
        raise RuntimeError(
            "adapter.coefficients() has changed since the inference "
            "cache was built. The cache is invalid. Construct a new "
            "Margins session."
        )


def _bootstrap_bank_key(session) -> str:
    """Deterministic key for a session's bootstrap resample bank.

    The key encodes every determinant of the resample index sequence so
    that two calls with the same key are guaranteed to use identical
    ``all_idx`` (and therefore composable).
    """
    parts = [
        str(session.rng_seed),
        str(session.n_boot),
    ]
    try:
        data = session.adapter.training_data
        n_obs = len(data)
    except (NotImplementedError, AttributeError, TypeError):
        n_obs = 0
    parts.append(str(n_obs))

    if session.matching is not None:
        cluster_ids = session.matching.cluster_ids
    else:
        cluster_ids = session.cluster
    if cluster_ids is not None:
        arr = np.asarray(cluster_ids)
        parts.append(hashlib.sha256(arr.tobytes()).hexdigest()[:16])
    else:
        parts.append("none")

    parts.append(str(session.block_size))
    bc = session.bootstrap_config or {}
    parts.append(str(bc.get("block_type", "moving")))
    # Sort keys for determinism
    parts.append(str(sorted(bc.items())))

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _bootstrap_resample_bank(session):
    """Return (all_idx, bank_id) for the session, generating if needed.

    The bank is cached on the session instance so that every bootstrap
    call in the same session reuses the same resample indices.
    """
    if session.method != "bootstrap":
        return None, None

    bank_key = _bootstrap_bank_key(session)
    cache_attr = "_bootstrap_bank_cache"

    if hasattr(session, cache_attr):
        cached_key, cached_idx = getattr(session, cache_attr)
        if cached_key == bank_key:
            return cached_idx, bank_key

    from .._inference._bootstrap import _generate_resample_indices

    try:
        data = session.adapter.training_data
        n_obs = len(data)
    except (NotImplementedError, AttributeError, TypeError):
        n_obs = 0

    if session.matching is not None:
        cluster_ids = session.matching.cluster_ids
    else:
        cluster_ids = session.cluster

    block_size = session.block_size
    bc = session.bootstrap_config or {}
    block_type = bc.get("block_type", "moving")

    all_idx = _generate_resample_indices(
        rng_seed=session.rng_seed,
        n_boot=session.n_boot,
        n_obs=n_obs,
        cluster_ids=cluster_ids,
        block_size=block_size,
        block_type=block_type,
    )
    setattr(session, cache_attr, (bank_key, all_idx))
    return all_idx, bank_key


def _bootstrap_states_bank(session):
    """Return (all_states, failures, bank_id), harvesting if needed.

    The states cache is keyed by the same bootstrap bank key so that
    invalidation rules are identical to the resample-index bank.
    """
    if session.method != "bootstrap":
        return None, None, None

    bank_key = _bootstrap_bank_key(session)
    cache_attr = "_bootstrap_states_cache"

    if hasattr(session, cache_attr):
        cached_key, cached_states, cached_failures = getattr(session, cache_attr)
        if cached_key == bank_key:
            return cached_states, cached_failures, bank_key

    all_idx, _ = _bootstrap_resample_bank(session)

    try:
        data = session.adapter.training_data
    except (NotImplementedError, AttributeError, TypeError):
        data = None

    if session.matching is not None:
        data = session.matching.matched_data

    if data is None:
        raise NotImplementedError(
            "Bootstrap inference requires the adapter to expose training_data. "
            f"{type(session.adapter).__name__} does not implement it."
        )

    from .._inference._bootstrap import _harvest_bootstrap_states

    states, failures = _harvest_bootstrap_states(
        session.adapter,
        data,
        all_idx,
        matching=session.matching,
        n_jobs=session.n_jobs,
        progress=session.progress_bar,
    )
    setattr(session, cache_attr, (bank_key, states, failures))
    return states, failures, bank_key


def _simulation_bank_key(session) -> str:
    """Deterministic key for a session's simulation draws bank."""
    parts = [
        str(session.rng_seed),
        str(session.n_sim),
    ]
    cov = _frozen_cov(session)
    parts.append(hashlib.sha256(np.asarray(cov).tobytes()).hexdigest()[:16])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _simulation_draws_bank(session):
    """Return (draws, bank_id) for the session, generating if needed.

    The bank is cached on the session instance so that every simulation
    call in the same session reuses the same β* draws.
    """
    if session.method != "simulation":
        return None, None

    bank_key = _simulation_bank_key(session)
    cache_attr = "_simulation_draws_cache"

    if hasattr(session, cache_attr):
        cached_key, cached_draws = getattr(session, cache_attr)
        if cached_key == bank_key:
            return cached_draws, bank_key

    beta = session.adapter.coefficients()
    Sigma = _frozen_cov(session)
    rng = np.random.default_rng(
        [session.rng_seed, 0] if session.rng_seed is not None else None
    )
    from .._inference._simulation import _generate_simulation_draws

    draws = _generate_simulation_draws(beta, Sigma, rng, session.n_sim)
    setattr(session, cache_attr, (bank_key, draws))
    return draws, bank_key


def _inference_config(session) -> InferenceConfig:
    """Build the InferenceConfig for a single call.

    All inference-related settings are session-level; per-call overrides
    are not supported by design. Switching method, level, vcov, or scale
    requires constructing a new ``Margins`` instance.
    """
    _check_adapter_drift(session)
    all_idx, _ = _bootstrap_resample_bank(session)
    all_states, all_states_failures, _ = _bootstrap_states_bank(session)
    sim_draws, _ = _simulation_draws_bank(session)
    return InferenceConfig(
        method=session.method,
        level=session.level,
        phi=session.phi,
        phi_inv=session.phi_inv,
        kappa_threshold=session.kappa_threshold,
        gradient_backend=session.gradient_backend,
        fd_step=session.fd_step,
        n_sim=session.n_sim,
        n_boot=session.n_boot,
        n_jobs=session.n_jobs,
        rng_seed=session.rng_seed,
        diagnostics=session.diagnostics,
        cov_params=_frozen_cov(session),
        cluster=session.cluster,
        block_size=session.block_size,
        matching=session.matching,
        bootstrap_config=session.bootstrap_config,
        all_idx=all_idx,
        all_states=all_states,
        all_states_failures=all_states_failures,
        sim_draws=sim_draws,
        progress_bar=session.progress_bar,
    )


def _frozen_cov(session) -> jnp.ndarray:
    """Resolve Σ̂ once per call and cache on the instance.

    Σ̂ is part of the session's analytical posture (vcov_spec is
    session-level). Caching ensures every result from this session
    carries the same Σ̂ reference even if the underlying model object
    is later mutated or re-fit by the user.
    """
    if not hasattr(session, "_cov_cache"):
        session._cov_cache = session.adapter.covariance(session.vcov_spec)
    return session._cov_cache


def _wrap_result(session, result_data: dict) -> MarginsResult:
    """Wrap a raw result dict from the engine in a MarginsResult.

    The session's resolved Σ̂ is frozen onto the result so downstream
    composition and hypothesis tests do not re-fetch it from the
    adapter (which could change if the underlying model is mutated).
    """
    n_obs = 0
    try:
        n_obs = len(session.adapter.training_data)
    except (NotImplementedError, AttributeError, TypeError):
        pass

    # Expand labels with outcome suffixes for multi-outcome models
    meta = dict(result_data.get("estimand_metadata", {}))
    n_outcomes = session.adapter.n_outcomes
    if n_outcomes > 1 and not meta.get("outcome_sliced"):
        old_labels = meta.get("labels")
        outcome_labels = session.adapter.outcome_labels or [
            str(i) for i in range(n_outcomes)
        ]
        if old_labels is None:
            old_labels = [""]
        expanded = []
        for lab in old_labels:
            for k in range(n_outcomes):
                suffix = outcome_labels[k]
                expanded.append(f"{lab} ({suffix})" if lab else suffix)
        meta["labels"] = expanded

        # Record outcome shape for structured downstream readers (G7a)
        est = np.asarray(result_data.get("estimate", []))
        n_atoms = 1
        if est.ndim >= 1:
            n_atoms = est.shape[0] if est.ndim == 2 else est.size // n_outcomes
        meta["_outcome_shape"] = {
            "n_outcomes": n_outcomes,
            "outcome_labels": outcome_labels,
            "n_atoms": n_atoms,
        }
        # Assert the load-bearing invariant (wrinkle 8)
        if est.ndim == 2:
            assert est.shape == (n_atoms, n_outcomes), (
                f"Invariant violation: estimate.shape {est.shape} does not match "
                f"expected (n_atoms={n_atoms}, n_outcomes={n_outcomes}). "
                f"The adapter may be emitting (n_outcomes, n_atoms) instead of "
                f"(n_atoms, n_outcomes)."
            )

    # Attach resample bank ID for bootstrap composition (G1 bootstrap)
    _, bank_id = _bootstrap_resample_bank(session)

    return MarginsResult(
        estimate=np.asarray(result_data["estimate"]),
        std_error=np.asarray(result_data["std_error"]),
        conf_int_lower=np.asarray(result_data["conf_int_lower"]),
        conf_int_upper=np.asarray(result_data["conf_int_upper"]),
        method=result_data["method"],
        level=result_data["level"],
        n_obs=n_obs,
        kappa=result_data.get("kappa"),
        delta_sim_disagreement=result_data.get("delta_sim_disagreement"),
        fallback_triggered=result_data.get("fallback_triggered", False),
        fallback_reason=result_data.get("fallback_reason"),
        estimand_metadata=meta,
        gradient=result_data.get("gradient"),
        draws=result_data.get("draws"),
        draws_inf=result_data.get("draws_inf"),
        cov_params=np.asarray(_frozen_cov(session)),
        phi=session.phi,
        phi_inv=session.phi_inv,
        session=weakref.ref(session),
        ci_method=result_data.get("ci_method"),
        bootstrap_extras=result_data.get("bootstrap_extras"),
        resample_bank_id=bank_id if result_data.get("method") == "bootstrap" else None,
        n_boot_effective=result_data.get("n_boot_effective"),
        n_boot_failed=result_data.get("n_boot_failed"),
    )
