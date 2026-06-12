"""Per-estimator inference banks.

Implements the bank model from design §9.4 and req. §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BankRetentionError(RuntimeError):
    """Raised when a novel query exceeds the retention budget."""

    def __init__(self) -> None:
        super().__init__(
            "This query needs replicate products that were not retained from the "
            "fan run. Issue queries together so products are captured in one pass; "
            "re-running is deterministic (same seed tree), but costs another fan "
            "execution."
        )


@dataclass
class BankSet:
    """Per-(estimator, branch) inference banks.

    Build once, replay across queries. Added in 0.4.0 (R1).
    """

    plan_hash: str
    branch_id: int
    seed: int | None
    _index_bank: dict = field(default_factory=dict, repr=False)
    _states_bank: dict = field(default_factory=dict, repr=False)
    _states_failures_bank: dict = field(default_factory=dict, repr=False)
    _draws_bank: dict = field(default_factory=dict, repr=False)

    def _index_key(
        self,
        *,
        n_obs: int,
        B: int,
        cluster: Any | None,
        block: int | None,
        block_type: str,
        strata: Any | None,
    ) -> tuple:
        cluster_bytes = (
            np.asarray(cluster).tobytes() if cluster is not None else b""
        )
        strata_bytes = np.asarray(strata).tobytes() if strata is not None else b""
        return (n_obs, B, cluster_bytes, block, block_type, strata_bytes)

    def resample_indices(
        self,
        *,
        n_obs: int,
        B: int,
        cluster: Any | None = None,
        block: int | None = None,
        block_type: str = "moving",
        strata: Any | None = None,
    ) -> list[np.ndarray]:
        """Return bootstrap resample indices, building once per key."""
        from ._seeds import legacy_resample_indices

        key = self._index_key(
            n_obs=n_obs,
            B=B,
            cluster=cluster,
            block=block,
            block_type=block_type,
            strata=strata,
        )
        if key not in self._index_bank:
            self._index_bank[key] = legacy_resample_indices(
                seed=self.seed,
                n=n_obs,
                B=B,
                cluster=cluster,
                block=block,
                block_type=block_type,
                strata=strata,
            )
        return self._index_bank[key]

    def bootstrap_states(
        self,
        *,
        adapter: Any,
        data: Any,
        indices: list[np.ndarray],
        matching: Any | None = None,
        transforms: list | None = None,
        n_jobs: int = 1,
        progress: bool = False,
    ) -> tuple[list, list]:
        """Harvest and cache bootstrap refitted states.

        Returns (states, failures) as produced by ``_harvest_bootstrap_states``.
        """
        from pymargins._inference._bootstrap import _harvest_bootstrap_states

        # Key on the index array identity and pipeline identity.
        idx_key = tuple(id(arr) for arr in indices)
        pipe_key = tuple(type(s).__name__ for s in (transforms or []))
        match_key = type(matching).__name__ if matching is not None else ""
        key = (idx_key, match_key, pipe_key, n_jobs)
        if key not in self._states_bank:
            states, failures = _harvest_bootstrap_states(
                adapter=adapter,
                data=data,
                all_idx=indices,
                matching=matching,
                transforms=transforms,
                rng_seed=self.seed,
                n_jobs=n_jobs,
                progress=progress,
            )
            self._states_bank[key] = states
            self._states_failures_bank[key] = failures
        return self._states_bank[key], self._states_failures_bank[key]

    def _draws_key(self, beta: np.ndarray, cov: np.ndarray, n_sim: int) -> tuple:
        return (np.asarray(beta).tobytes(), np.asarray(cov).tobytes(), n_sim)

    def sim_draws(
        self,
        *,
        beta: np.ndarray,
        cov: np.ndarray,
        n_sim: int,
    ) -> np.ndarray:
        """Return simulation beta draws, building once per key."""
        from ._seeds import legacy_sim_draws

        key = self._draws_key(beta, cov, n_sim)
        if key not in self._draws_bank:
            self._draws_bank[key] = legacy_sim_draws(
                seed=self.seed,
                n_sim=n_sim,
                beta=beta,
                cov=cov,
            )
        return self._draws_bank[key]
