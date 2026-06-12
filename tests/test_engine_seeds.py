"""Determinism tests for the seed/bank derivations.

Design §9.4, req §5. Added in 0.4.0 (R1).
"""

from __future__ import annotations

import numpy as np
import pytest

from pymargins._engine._seeds import (
    legacy_resample_indices,
    legacy_sim_draws,
    seed_sequence_for_branch,
)
from pymargins._inference._bootstrap import _generate_resample_indices
from pymargins._inference._simulation import _generate_simulation_draws


def _make_kwargs(kind: str, n: int = 12):
    if kind == "iid":
        return {}
    if kind == "cluster":
        return {"cluster": np.repeat(np.arange(4), n // 4)}
    if kind == "block-moving":
        return {"block": 3, "block_type": "moving"}
    if kind == "block-circular":
        return {"block": 3, "block_type": "circular"}
    if kind == "stratified":
        return {
            "cluster": np.repeat(np.arange(4), n // 4),
            "strata": np.repeat(np.arange(2), n // 2),
        }
    raise ValueError(kind)


@pytest.mark.parametrize("seed", [0, 7, 20260611])
@pytest.mark.parametrize(
    "kind",
    [
        "iid",
        "cluster",
        "block-moving",
        "block-circular",
        "stratified",
    ],
)
def test_resample_indices_deterministic(seed, kind):
    n = 12
    B = 4
    kwargs = _make_kwargs(kind, n)

    a = legacy_resample_indices(seed=seed, n=n, B=B, **kwargs)
    b = legacy_resample_indices(seed=seed, n=n, B=B, **kwargs)
    assert len(a) == len(b) == B
    for aa, bb in zip(a, b, strict=True):
        assert np.array_equal(aa, bb)


def test_resample_indices_regression_golden():
    """Layer-4 regression golden: pasted literal arrays from a verified run.

    Generated 2026-06-12 with legacy_resample_indices(n=12, B=4) across the
    full {iid, cluster, block-moving, block-circular, stratified} matrix and
    seeds {0, 7, 20260611}. Regeneration requires a ledger entry.
    """
    expected = {
        (0, "iid"): [
            np.array([6, 10, 11, 6, 3, 9, 2, 11, 10, 0, 0, 2]),
            np.array([3, 9, 3, 0, 0, 8, 11, 7, 5, 0, 11, 1]),
            np.array([0, 1, 2, 9, 0, 11, 0, 1, 5, 6, 8, 3]),
            np.array([4, 0, 7, 4, 8, 3, 2, 8, 5, 5, 10, 1]),
        ],
        (0, "cluster"): [
            np.array([6, 7, 8, 9, 10, 11, 9, 10, 11, 6, 7, 8]),
            np.array([3, 4, 5, 9, 10, 11, 0, 1, 2, 9, 10, 11]),
            np.array([9, 10, 11, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            np.array([3, 4, 5, 9, 10, 11, 3, 4, 5, 0, 1, 2]),
        ],
        (0, "block-moving"): [
            np.array([5, 6, 7, 8, 9, 10, 9, 10, 11, 5, 6, 7]),
            np.array([3, 4, 5, 8, 9, 10, 2, 3, 4, 9, 10, 11]),
            np.array([8, 9, 10, 0, 1, 2, 0, 1, 2, 2, 3, 4]),
            np.array([2, 3, 4, 7, 8, 9, 2, 3, 4, 0, 1, 2]),
        ],
        (0, "block-circular"): [
            np.array([6, 7, 8, 10, 11, 0, 11, 0, 1, 6, 7, 8]),
            np.array([3, 4, 5, 9, 10, 11, 2, 3, 4, 11, 0, 1]),
            np.array([10, 11, 0, 0, 1, 2, 0, 1, 2, 2, 3, 4]),
            np.array([3, 4, 5, 9, 10, 11, 3, 4, 5, 0, 1, 2]),
        ],
        (0, "stratified"): [
            np.array([3, 4, 5, 3, 4, 5, 9, 10, 11, 9, 10, 11]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 6, 7, 8]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8]),
        ],
        (7, "iid"): [
            np.array([10, 9, 10, 1, 3, 2, 2, 1, 6, 5, 6, 6]),
            np.array([10, 11, 4, 3, 2, 3, 4, 6, 7, 2, 7, 0]),
            np.array([0, 9, 7, 10, 1, 10, 11, 4, 5, 4, 6, 6]),
            np.array([9, 2, 5, 8, 5, 5, 8, 6, 6, 0, 1, 4]),
        ],
        (7, "cluster"): [
            np.array([9, 10, 11, 9, 10, 11, 9, 10, 11, 0, 1, 2]),
            np.array([3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            np.array([6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]),
            np.array([9, 10, 11, 9, 10, 11, 3, 4, 5, 3, 4, 5]),
        ],
        (7, "block-moving"): [
            np.array([8, 9, 10, 7, 8, 9, 8, 9, 10, 1, 2, 3]),
            np.array([2, 3, 4, 1, 2, 3, 2, 3, 4, 1, 2, 3]),
            np.array([5, 6, 7, 4, 5, 6, 5, 6, 7, 5, 6, 7]),
            np.array([8, 9, 10, 9, 10, 11, 3, 4, 5, 3, 4, 5]),
        ],
        (7, "block-circular"): [
            np.array([10, 11, 0, 9, 10, 11, 10, 11, 0, 1, 2, 3]),
            np.array([3, 4, 5, 2, 3, 4, 2, 3, 4, 1, 2, 3]),
            np.array([6, 7, 8, 5, 6, 7, 6, 7, 8, 6, 7, 8]),
            np.array([10, 11, 0, 11, 0, 1, 4, 5, 6, 3, 4, 5]),
        ],
        (7, "stratified"): [
            np.array([3, 4, 5, 3, 4, 5, 9, 10, 11, 6, 7, 8]),
            np.array([0, 1, 2, 0, 1, 2, 6, 7, 8, 6, 7, 8]),
            np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 9, 10, 11]),
            np.array([3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8]),
        ],
        (20260611, "iid"): [
            np.array([8, 2, 6, 6, 10, 7, 0, 6, 6, 9, 1, 2]),
            np.array([10, 1, 3, 8, 10, 1, 4, 1, 5, 2, 0, 8]),
            np.array([3, 1, 10, 8, 8, 3, 3, 10, 1, 8, 11, 8]),
            np.array([3, 9, 11, 8, 2, 3, 10, 7, 3, 3, 8, 2]),
        ],
        (20260611, "cluster"): [
            np.array([6, 7, 8, 0, 1, 2, 6, 7, 8, 6, 7, 8]),
            np.array([9, 10, 11, 6, 7, 8, 0, 1, 2, 6, 7, 8]),
            np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 0, 1, 2]),
            np.array([9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ],
        (20260611, "block-moving"): [
            np.array([7, 8, 9, 1, 2, 3, 5, 6, 7, 5, 6, 7]),
            np.array([8, 9, 10, 5, 6, 7, 0, 1, 2, 5, 6, 7]),
            np.array([5, 6, 7, 7, 8, 9, 0, 1, 2, 2, 3, 4]),
            np.array([8, 9, 10, 1, 2, 3, 2, 3, 4, 7, 8, 9]),
        ],
        (20260611, "block-circular"): [
            np.array([8, 9, 10, 2, 3, 4, 6, 7, 8, 6, 7, 8]),
            np.array([10, 11, 0, 7, 8, 9, 0, 1, 2, 6, 7, 8]),
            np.array([6, 7, 8, 9, 10, 11, 1, 2, 3, 2, 3, 4]),
            np.array([10, 11, 0, 1, 2, 3, 3, 4, 5, 8, 9, 10]),
        ],
        (20260611, "stratified"): [
            np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 9, 10, 11]),
            np.array([3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            np.array([3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8]),
            np.array([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]),
        ],
    }

    for (seed, kind), exp in expected.items():
        got = legacy_resample_indices(seed=seed, n=12, B=4, **_make_kwargs(kind, 12))
        assert len(got) == len(exp), (seed, kind)
        for g, e in zip(got, exp, strict=True):
            np.testing.assert_array_equal(g, e)


def test_sim_draws_regression_golden():
    """Layer-4 regression golden for simulation draws."""
    beta = np.array([0.5, -0.2])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    got = legacy_sim_draws(seed=0, n_sim=5, beta=beta, cov=cov)
    expected = np.array(
        [
            [4.81896882557240780e-01, -1.57692700358071147e-01],
            [5.56818703515076341e-01, -1.32028650826709060e-02],
            [5.38570367374557324e-01, -3.72823357399233835e-01],
            [7.56231882913327769e-01, 1.53282432872774266e-01],
            [2.17189194404668884e-01, -3.62823675683039526e-01],
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)

    # Wrapper-vs-derivation identity.
    rng = np.random.default_rng([0, 0])
    direct = _generate_simulation_draws(beta, cov, rng, 5)
    np.testing.assert_array_equal(got, direct)


def test_resample_indices_wrapper_matches_direct():
    """The indices wrapper uses [seed, 1] internally."""
    got = legacy_resample_indices(seed=7, n=8, B=2)
    direct = _generate_resample_indices(
        rng_seed=7,
        n_boot=2,
        n_obs=8,
        cluster_ids=None,
        block_size=None,
        block_type="moving",
        strata=None,
    )
    assert len(got) == len(direct)
    for g, d in zip(got, direct, strict=True):
        np.testing.assert_array_equal(g, d)


def test_spawn_tree_order_invariant():
    """Branch seeds are distinct but deterministic regardless of order."""
    s = 123
    n = 4
    b0 = seed_sequence_for_branch(s, 0, n)
    b1 = seed_sequence_for_branch(s, 1, n)
    b0_again = seed_sequence_for_branch(s, 0, n)
    assert b0 == b0_again
    assert b0 != b1
    assert len(b0) == len(b1) == n
