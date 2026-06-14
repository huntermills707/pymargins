"""Doctrine executor. Design §5, req §5. Added in 0.4.0 (R3)."""

from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np

from pymargins._estimands import is_jax_differentiable
from pymargins._inference._bootstrap import _run_bootstrap
from pymargins._inference._delta import _run_delta
from pymargins._inference._simulation import _run_simulation
from pymargins._soundness._constants import (
    REPLICATE_FAILURE_NOTE,
    REPLICATE_FAILURE_WARN,
)
from pymargins._soundness._predicates import CompileError, SoundnessWarning

from ._banks import BankSet
from ._queries import CompiledQuery, WiringFacts, build_inference_config


def execute_query(
    compiled: CompiledQuery,
    *,
    adapter: Any,
    plan: Any,
    wiring_facts: WiringFacts,
    banks: BankSet,
    frozen_cov: Any,
    n_jobs: int = 1,
    progress_bar: bool = False,
) -> dict:
    """Run one compiled query through the resolved method.

    Returns the kernel result dict (see implementation-guide G1.3) — the
    result layer (R4) wraps it.
    """
    method = plan.method_resolved

    config = build_inference_config(
        plan,
        adapter,
        wiring_facts,
        banks,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
        frozen_cov=frozen_cov,
    )

    if method == "delta":
        beta = jnp.asarray(adapter.coefficients())
        if not is_jax_differentiable(compiled.h, beta):
            raise CompileError(
                "Non-differentiable estimand under method='delta'. "
                'Use method="simulation" (§4.8).'
            )
        result = _run_delta(
            compiled.h,
            adapter,
            config,
            compiled.estimand_metadata,
        )
        # Defensive fallback: Σ̂ should always be pre-resolved by R2.2, but if
        # a caller passes None, recompute once rather than storing an object
        # array that silently passes the `cov_params is not None` guard.
        if frozen_cov is not None:
            result["cov_params"] = np.asarray(frozen_cov)
        else:
            result["cov_params"] = np.asarray(adapter.covariance())

    elif method == "simulation":
        beta = np.asarray(adapter.coefficients())
        draws = banks.sim_draws(beta=beta, cov=frozen_cov, n_sim=plan.n_sim)
        config.sim_draws = draws
        result = _run_simulation(
            compiled.h,
            adapter,
            config,
            compiled.estimand_metadata,
        )

    elif method == "bootstrap":
        result = _run_bootstrap_query(
            compiled,
            adapter,
            config,
            plan,
            wiring_facts,
            banks,
            n_jobs,
            progress_bar,
        )
        _record_replicate_failures(result)

    else:
        raise AssertionError(f"Unreachable method: {method!r}")

    return result


def _resolve_resample_source(adapter: Any, wiring_facts: WiringFacts) -> tuple[Any, int]:
    """Return (data, n_obs) for the bootstrap resampler.

    The resample source is normally ``adapter.training_data``, but a
    transform pipeline may declare a ``source_data`` override.

    Mirrors the legacy glue's guard: a missing ``training_data`` implementation
    is tolerated until we actually need the data, at which point the kernel
    raises a bootstrap-specific ``NotImplementedError``.
    """
    try:
        data = adapter.training_data
        n_obs = len(data)
    except (NotImplementedError, AttributeError, TypeError):
        data = None
        n_obs = 0

    if wiring_facts.transforms:
        for stage in wiring_facts.transforms:
            source_data = getattr(stage, "source_data", None)
            if source_data is not None:
                data = source_data
                n_obs = len(data)
                break
    return data, n_obs


def _resolve_bootstrap_clusters(
    wiring_facts: WiringFacts,
) -> tuple[Any | None, Any | None]:
    """Return (cluster_ids, strata) honoring matching and survey design."""
    if wiring_facts.matching is not None:
        cluster_ids = wiring_facts.matching.cluster_ids
    else:
        cluster_ids = wiring_facts.cluster

    strata = None
    if wiring_facts.design is not None:
        cluster_ids = wiring_facts.design.psu
        strata = wiring_facts.design.strata

    return cluster_ids, strata


def _run_bootstrap_query(
    compiled: CompiledQuery,
    adapter: Any,
    config: Any,
    plan: Any,
    wiring_facts: WiringFacts,
    banks: BankSet,
    n_jobs: int,
    progress_bar: bool,
) -> dict:
    """Execute the bootstrap path: indices → states → kernel."""
    data, n_obs = _resolve_resample_source(adapter, wiring_facts)

    if wiring_facts.matching is not None:
        data = wiring_facts.matching.matched_data

    if data is None:
        raise NotImplementedError(
            "Bootstrap inference requires the adapter to expose training_data. "
            f"{type(adapter).__name__} does not implement it."
        )

    cluster_ids, strata = _resolve_bootstrap_clusters(wiring_facts)

    indices = banks.resample_indices(
        n_obs=n_obs,
        B=plan.B,
        cluster=cluster_ids,
        block=wiring_facts.block,
        block_type=wiring_facts.block_type,
        strata=strata,
    )
    states, failures = banks.bootstrap_states(
        adapter=adapter,
        data=data,
        indices=indices,
        matching=wiring_facts.matching,
        transforms=wiring_facts.transforms,
        n_jobs=n_jobs,
        progress=progress_bar,
    )
    config.all_idx = indices
    config.all_states = states
    config.all_states_failures = failures

    return _run_bootstrap(
        compiled.h,
        adapter,
        config,
        compiled.estimand_metadata,
        h_factory=compiled.h_factory,
    )


def _record_replicate_failures(result: dict) -> None:
    """Append diagnostics / warnings for bootstrap replicate failure rates.

    The result dict may carry the same ``estimand_metadata`` object that was
    frozen into the ``CompiledQuery``. Copy before annotating so replaying the
    same query does not accumulate duplicate diagnostics.
    """
    n_eff = result.get("n_boot_effective")
    n_fail = result.get("n_boot_failed")
    if n_eff is None or n_fail is None:
        return
    total = n_eff + n_fail
    if total == 0:
        return
    rate = n_fail / total

    estimand_metadata = dict(result.get("estimand_metadata", {}))
    diagnostics = list(estimand_metadata.get("diagnostics", []))

    if rate > REPLICATE_FAILURE_WARN:
        msg = (
            f"Bootstrap replicate failure rate {rate:.1%} exceeds "
            f"{REPLICATE_FAILURE_WARN:.0%}; SE/CI may be unreliable."
        )
        diagnostics.append(msg)
        warnings.warn(msg, SoundnessWarning, stacklevel=3)
    elif rate > REPLICATE_FAILURE_NOTE:
        msg = (
            f"Bootstrap replicate failure rate {rate:.1%} exceeds "
            f"{REPLICATE_FAILURE_NOTE:.0%}."
        )
        diagnostics.append(msg)

    if diagnostics:
        estimand_metadata["diagnostics"] = diagnostics
        result["estimand_metadata"] = estimand_metadata
