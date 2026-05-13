"""Inference configuration and result wrapping helpers."""

from __future__ import annotations
from typing import Any
import weakref

import jax.numpy as jnp
import numpy as np

from .._inference import InferenceConfig, run_inference
from .._result import MarginsResult


def _inference_config(session) -> InferenceConfig:
    """Build the InferenceConfig for a single call.

    All inference-related settings are session-level; per-call overrides
    are not supported by design. Switching method, level, vcov, or scale
    requires constructing a new ``Margins`` instance.
    """
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
    if session.adapter.n_outcomes > 1 and not meta.get("outcome_sliced"):
        old_labels = meta.get("labels")
        outcome_labels = session.adapter.outcome_labels or [
            str(i) for i in range(session.adapter.n_outcomes)
        ]
        if old_labels is None:
            old_labels = [""]
        expanded = []
        for lab in old_labels:
            for k in range(session.adapter.n_outcomes):
                suffix = outcome_labels[k]
                expanded.append(f"{lab} ({suffix})" if lab else suffix)
        meta["labels"] = expanded

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
    )
