"""Seed tree and bank derivations.

Implements the seed model from design §9.4 and req. §5.
Added in 0.4.0 (R1); renamed in 0.4.0 (R7).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def resample_indices(
    seed: int | None,
    n: int,
    B: int,
    cluster: Any | None = None,
    block: int | None = None,
    block_type: str = "moving",
    strata: Any | None = None,
) -> list[np.ndarray]:
    """Generate bootstrap resample indices."""
    from pymargins._inference._bootstrap import (
        _generate_resample_indices,
        _validate_resample_options,
    )

    _validate_resample_options(cluster, block, block_type, n)

    return _generate_resample_indices(
        rng_seed=seed,
        n_boot=B,
        n_obs=n,
        cluster_ids=cluster,
        block_size=block,
        block_type=block_type,
        strata=strata,
    )


def sim_draws(
    seed: int | None,
    n_sim: int,
    beta: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Generate simulation draws."""
    rng = np.random.default_rng([seed, 0] if seed is not None else None)
    from pymargins._inference._simulation import _generate_simulation_draws

    return _generate_simulation_draws(beta, cov, rng, n_sim)


def seed_sequence_for_branch(
    master_seed: int | None,
    branch_id: int,
    n_replicates: int,
) -> list[int]:
    """Spawn a seed sequence for one fan branch.

    Uses ``numpy.random.SeedSequence`` for reproducible scheduling.
    """
    if master_seed is None:
        return [None] * n_replicates
    ss = np.random.SeedSequence(master_seed)
    branches = ss.spawn(branch_id + 1)
    branch_ss = branches[branch_id]
    child_seeds = branch_ss.spawn(n_replicates)
    return [int(c.generate_state(1)[0]) for c in child_seeds]
