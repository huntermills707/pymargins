from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .._adapter import InferenceMethod
from .._gradients import GradientBackend


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
    phi: Callable | None = None
    phi_inv: Callable | None = None
    kappa_threshold: float = 0.3
    gradient_backend: GradientBackend = "autodiff"
    fd_step: float = 1e-6
    n_sim: int = 4000
    n_boot: int = 1000
    n_jobs: int = 1
    rng_seed: int | None = None
    diagnostics: bool = True
    cov_params: jnp.ndarray | None = None
    cluster: Any | None = None
    block_size: int | None = None
    strata: Any | None = None
    survey_design: Any | None = None
    matching: Any | None = None
    """MatchingClient for bootstrap rematching. Ignored by delta and K–R."""

    bootstrap_config: dict | None = None
    all_idx: list | None = None
    """Pre-generated resample indices for bootstrap. When provided, the
    bootstrap engine reuses these instead of generating fresh indices.
    Used by session-level resample banks to guarantee matched replicates
    across composable bootstrap calls."""

    all_states: list | None = None
    """Pre-harvested bootstrap states (refitted adapters). When provided,
    the bootstrap engine skips refitting and replays h over the cached
    states. Used by session-level bootstrap state banks."""

    all_states_failures: list | None = None
    """Failed replicates corresponding to ``all_states``. Paired with
    ``all_states`` so failure accounting is consistent across calls."""

    sim_draws: Any | None = None
    """Pre-generated simulation β* draws. When provided, the simulation
    engine reuses these instead of drawing fresh values. Used by
    session-level simulation draw banks."""

    progress_bar: bool = False
    """Whether to show a progress bar during bootstrap execution.
    Requires ``tqdm`` to be installed."""
