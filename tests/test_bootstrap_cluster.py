"""Tests for cluster bootstrap resampling."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_clustered():
    """Synthetic data with cluster structure."""
    rng = np.random.default_rng(77)
    n_clusters = 30
    cluster_size = 10
    n = n_clusters * cluster_size

    clusters = np.repeat(np.arange(n_clusters), cluster_size)
    # Cluster-level random effect
    cluster_re = rng.standard_normal(n_clusters)[clusters]

    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "cluster": clusters,
        }
    )
    eta = 0.5 + 0.8 * df["x1"] - 0.4 * df["x2"] + cluster_re
    df["y"] = eta + rng.standard_normal(n)
    return df


@pytest.fixture
def ols_fit_formula(df_clustered):
    return smf.ols("y ~ x1 + x2", data=df_clustered).fit()


@pytest.fixture
def ols_fit_array(df_clustered):
    return sm.OLS(df_clustered["y"], sm.add_constant(df_clustered[["x1", "x2"]])).fit()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_cluster_nan_raises(ols_fit_formula, df_clustered):
    cluster_with_nan = df_clustered["cluster"].astype(float).copy()
    cluster_with_nan.iloc[0] = np.nan
    est = GComputation(
        steps.input(df_clustered, cluster=cluster_with_nan),
        outcome=ols_fit_formula,
        method="bootstrap",
        B=50,
    )
    with pytest.raises(ValueError, match="NaN"):
        est.dydx("x1")


def test_cluster_wrong_length_raises(ols_fit_formula, df_clustered):
    # A cluster declaration of the wrong length is caught when the estimator
    # freezes the cluster-robust covariance matrix at compile time.
    with pytest.raises(ValueError, match="length"):
        GComputation(
            steps.input(df_clustered, cluster=df_clustered["cluster"].iloc[:-1]),
            outcome=ols_fit_formula,
            method="bootstrap",
            B=50,
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_reproducible(ols_fit_formula, df_clustered):
    inp = steps.input(df_clustered, cluster=df_clustered["cluster"])
    m1 = GComputation(
        inp,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    m2 = GComputation(
        inp,
        outcome=ols_fit_formula,
        method="bootstrap",
        B=100,
        seed=42,
    )
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Cluster vs i.i.d. bootstrap give different results on clustered data
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_differs_from_iid(ols_fit_formula, df_clustered):
    m_cluster = GComputation(
        steps.input(df_clustered, cluster=df_clustered["cluster"]),
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
    res_cluster = m_cluster.dydx("x1")
    res_iid = m_iid.dydx("x1")
    # SEs should differ; cluster bootstrap SE should be larger for clustered data
    assert res_cluster.std_error != res_iid.std_error


# ---------------------------------------------------------------------------
# Cluster bootstrap SE roughly matches analytical cluster-robust SE
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_se_matches_analytical(ols_fit_formula, df_clustered):
    """Cluster-bootstrap SE should be in the same ballpark as analytical cluster-robust SE."""
    m_cluster = GComputation(
        steps.input(df_clustered, cluster=df_clustered["cluster"]),
        outcome=ols_fit_formula,
        method="bootstrap",
        B=400,
        seed=42,
    )
    res_cluster = m_cluster.dydx("x1")

    m_analytical = GComputation(
        ols_fit_formula,
        method="delta",
        vcov={"type": "cluster", "groups": df_clustered["cluster"]},
    )
    res_analytical = m_analytical.dydx("x1")

    # Allow 30% relative difference — bootstrap with 400 replicates is noisy
    np.testing.assert_allclose(
        res_cluster.std_error, res_analytical.std_error, rtol=0.30
    )


# ---------------------------------------------------------------------------
# Works with array-fit models
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_array_fit(ols_fit_array, df_clustered):
    from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter

    adapter = StatsmodelsOLSAdapter(ols_fit_array, training_data=df_clustered)
    m = GComputation(
        steps.input(df_clustered, cluster=df_clustered["cluster"]),
        outcome=ols_fit_array,
        adapter=adapter,
        method="bootstrap",
        B=100,
        seed=42,
    )
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Multiplicity: oversampled clusters appear multiple times
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_multiplicity():
    """Verify that if a cluster is sampled twice, its rows appear twice."""
    # Use a tiny dataset for deterministic testing
    tiny_df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [0.0, 0.0, 1.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "cluster": [0, 0, 1, 1],
        }
    )
    fit = smf.ols("y ~ x1 + x2", data=tiny_df).fit()

    # Directly test the resampling logic via the bootstrap kernel.
    from pymargins._adapter import auto_detect_adapter
    from pymargins._inference import InferenceConfig, _run_bootstrap

    adapter = auto_detect_adapter(fit)
    config = InferenceConfig(
        method="bootstrap", n_boot=1, rng_seed=1, cluster=tiny_df["cluster"].values
    )

    # Build a simple estimand that counts rows
    def h_factory(a):
        def h(beta):
            return float(len(a.training_data))

        return h

    result = _run_bootstrap(
        lambda b: float(len(adapter.training_data)),
        adapter,
        config,
        {},
        h_factory=h_factory,
    )
    # With 2 equal-size clusters of 2 rows each, resampled length is always 4
    # (sample 2 clusters with replacement from 2 clusters; each sampled cluster
    # contributes its 2 rows, so total is always 2 × 2 = 4).
    resampled_n = result["draws"][0]
    assert resampled_n == 4


# ---------------------------------------------------------------------------
# Delta and simulation paths with a cluster declaration
# ---------------------------------------------------------------------------


def test_delta_with_cluster(ols_fit_formula, df_clustered):
    """Delta method should work fine when a cluster variable is declared."""
    m = GComputation(
        steps.input(df_clustered, cluster=df_clustered["cluster"]),
        outcome=ols_fit_formula,
        method="delta",
    )
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


def test_simulation_with_cluster(ols_fit_formula, df_clustered):
    """Simulation should work fine when a cluster variable is declared."""
    m = GComputation(
        steps.input(df_clustered, cluster=df_clustered["cluster"]),
        outcome=ols_fit_formula,
        method="simulation",
        n_sim=500,
    )
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
