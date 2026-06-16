"""Tests for block bootstrap resampling."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation, steps
from pymargins._adapter import auto_detect_adapter
from pymargins._inference import InferenceConfig, _run_bootstrap

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_time_series():
    """Synthetic time series data with AR(1) structure."""
    rng = np.random.default_rng(88)
    n = 200
    x = rng.standard_normal(n)
    # AR(1) error: y_t = 0.5 + 0.8*x_t + e_t, e_t = 0.7*e_{t-1} + u_t
    u = rng.standard_normal(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + u[t]
    y = 0.5 + 0.8 * x + e
    df = pd.DataFrame({"x": x, "y": y})
    return df


@pytest.fixture
def ols_fit_formula(df_time_series):
    return smf.ols("y ~ x", data=df_time_series).fit()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_block_size_too_large_raises(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=len(df_time_series) + 1)
    est = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
    )
    with pytest.raises(ValueError, match="block_size"):
        est.dydx("x")


def test_block_size_zero_raises(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=0)
    est = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
    )
    with pytest.raises(ValueError, match="block_size"):
        est.dydx("x")


def test_invalid_block_type_raises(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=5, block_type="invalid")
    est = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
    )
    with pytest.raises(ValueError, match="block_type"):
        est.dydx("x")


def test_cluster_and_block_size_mutually_exclusive(ols_fit_formula, df_time_series):
    node = steps.input(
        df_time_series,
        cluster=df_time_series.index,
        block=5,
    )
    est = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        est.dydx("x")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_block_bootstrap_reproducible(ols_fit_formula, df_time_series):
    node1 = steps.input(df_time_series, block=5)
    node2 = steps.input(df_time_series, block=5)
    m1 = GComputation(
        node1,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    m2 = GComputation(
        node2,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    res1 = m1.dydx("x")
    res2 = m2.dydx("x")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Block bootstrap differs from i.i.d. on dependent data
# ---------------------------------------------------------------------------


def test_block_bootstrap_differs_from_iid(ols_fit_formula, df_time_series):
    node_block = steps.input(df_time_series, block=10)
    m_block = GComputation(
        node_block,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=200,
        seed=42,
    )
    m_iid = GComputation(
        ols_fit_formula,
        method="bootstrap",
        B=200,
        seed=42,
    )
    res_block = m_block.dydx("x")
    res_iid = m_iid.dydx("x")
    # SEs should differ; for AR data, block bootstrap SE should be larger
    assert res_block.std_error != res_iid.std_error


# ---------------------------------------------------------------------------
# Block types produce different resampling
# ---------------------------------------------------------------------------


def test_moving_vs_nonoverlapping_different(ols_fit_formula, df_time_series):
    node_moving = steps.input(df_time_series, block=10, block_type="moving")
    node_nonover = steps.input(df_time_series, block=10, block_type="nonoverlapping")
    m_moving = GComputation(
        node_moving,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    m_nonover = GComputation(
        node_nonover,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    res_moving = m_moving.dydx("x")
    res_nonover = m_nonover.dydx("x")
    # They should produce different results
    assert res_moving.std_error != res_nonover.std_error


# ---------------------------------------------------------------------------
# Circular block bootstrap works
# ---------------------------------------------------------------------------


def test_circular_block_bootstrap(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=10, block_type="circular")
    m = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    res = m.dydx("x")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Block bootstrap SE roughly matches analytical HAC SE
# ---------------------------------------------------------------------------


def test_block_bootstrap_se_larger_than_iid(ols_fit_formula, df_time_series):
    """For AR data, block-bootstrap SE should be larger than i.i.d. bootstrap SE."""
    node_block = steps.input(df_time_series, block=10)
    m_block = GComputation(
        node_block,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=400,
        seed=42,
    )
    res_block = m_block.dydx("x")

    m_iid = GComputation(
        ols_fit_formula,
        method="bootstrap",
        B=400,
        seed=42,
    )
    res_iid = m_iid.dydx("x")

    # Block bootstrap should be more conservative for dependent data
    assert res_block.std_error > res_iid.std_error


# ---------------------------------------------------------------------------
# Multiplicity: block bootstrap preserves block structure
# ---------------------------------------------------------------------------


def test_block_bootstrap_multiplicity(ols_fit_formula):
    """Verify that resampled data preserves contiguous blocks."""
    tiny_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    fit = smf.ols("y ~ x", data=tiny_df).fit()
    adapter = auto_detect_adapter(fit)

    config = InferenceConfig(
        method="bootstrap",
        n_boot=1,
        rng_seed=1,
        block_size=2,
        bootstrap_config={"block_type": "moving"},
    )

    def h_factory(a):
        def h(beta):
            # Return the sum of y values as a scalar
            return float(a.training_data["y"].sum())

        return h

    result = _run_bootstrap(
        lambda b: float(adapter.training_data["y"].sum()),
        adapter,
        config,
        {},
        h_factory=h_factory,
    )
    # With block_size=2 and n=6, k=ceil(6/2)=3 blocks
    # Resampled length should be 3*2=6
    resampled_sum = result["draws"][0]
    # The sum should be a combination of 3 block sums from {3, 5, 7, 9, 11}
    # Minimum: 3+3+3=9, Maximum: 11+11+11=33
    assert 9.0 <= resampled_sum <= 33.0


# ---------------------------------------------------------------------------
# Delta and simulation paths ignore block_size
# ---------------------------------------------------------------------------


def test_delta_ignores_block_size(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=5)
    m = GComputation(node, outcome=ols_fit_formula, method="delta")
    res = m.dydx("x")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


def test_simulation_ignores_block_size(ols_fit_formula, df_time_series):
    node = steps.input(df_time_series, block=5)
    m = GComputation(
        node,
        outcome=ols_fit_formula,
        method="simulation",
        n_sim=500,
    )
    res = m.dydx("x")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Non-overlapping block bootstrap with edge case
# ---------------------------------------------------------------------------


def test_nonoverlapping_block_size_does_not_divide_n(ols_fit_formula, df_time_series):
    """When n is not divisible by block_size, NBB should still work."""
    # n=200, block_size=7 → n_blocks=28, remainder=4 ignored
    node = steps.input(df_time_series, block=7, block_type="nonoverlapping")
    m = GComputation(
        node,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
        seed=42,
    )
    res = m.dydx("x")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
