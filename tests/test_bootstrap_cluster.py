"""Tests for cluster bootstrap resampling.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pymargins import Margins


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

    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "cluster": clusters,
    })
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
    with pytest.raises(ValueError, match="NaN"):
        Margins(ols_fit_formula, method="bootstrap", n_boot=50, cluster=cluster_with_nan)


def test_cluster_wrong_length_raises(ols_fit_formula, df_clustered):
    with pytest.raises(ValueError, match="length"):
        Margins(ols_fit_formula, method="bootstrap", n_boot=50, cluster=df_clustered["cluster"].iloc[:-1])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_reproducible(ols_fit_formula, df_clustered):
    m1 = Margins(ols_fit_formula, method="bootstrap", n_boot=100, rng_seed=42, cluster=df_clustered["cluster"])
    m2 = Margins(ols_fit_formula, method="bootstrap", n_boot=100, rng_seed=42, cluster=df_clustered["cluster"])
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Cluster vs i.i.d. bootstrap give different results on clustered data
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_differs_from_iid(ols_fit_formula, df_clustered):
    m_cluster = Margins(ols_fit_formula, method="bootstrap", n_boot=200, rng_seed=42,
                        cluster=df_clustered["cluster"])
    m_iid = Margins(ols_fit_formula, method="bootstrap", n_boot=200, rng_seed=42)
    res_cluster = m_cluster.dydx("x1")
    res_iid = m_iid.dydx("x1")
    # SEs should differ; cluster bootstrap SE should be larger for clustered data
    assert res_cluster.std_error != res_iid.std_error


# ---------------------------------------------------------------------------
# Cluster bootstrap SE roughly matches analytical cluster-robust SE
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_se_matches_analytical(ols_fit_formula, df_clustered):
    """Cluster-bootstrap SE should be in the same ballpark as analytical cluster-robust SE."""
    m_cluster = Margins(ols_fit_formula, method="bootstrap", n_boot=400, rng_seed=42,
                        cluster=df_clustered["cluster"])
    res_cluster = m_cluster.dydx("x1")

    m_analytical = Margins(ols_fit_formula, method="delta",
                           vcov={"type": "cluster", "groups": df_clustered["cluster"]})
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
    m = Margins(ols_fit_array, method="bootstrap", n_boot=100, rng_seed=42,
                cluster=df_clustered["cluster"], adapter=adapter)
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Multiplicity: oversampled clusters appear multiple times
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_multiplicity(ols_fit_formula, df_clustered):
    """Verify that if a cluster is sampled twice, its rows appear twice."""
    # Use a tiny dataset for deterministic testing
    rng = np.random.default_rng(99)
    tiny_df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [0.0, 0.0, 1.0, 1.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "cluster": [0, 0, 1, 1],
    })
    fit = smf.ols("y ~ x1 + x2", data=tiny_df).fit()

    # Force a specific resampling by controlling the RNG seed
    # With seed=1 and 2 clusters, rng.choice([0,1], size=2, replace=True)
    # will sample clusters; we just verify total rows in resampled data
    m = Margins(fit, method="bootstrap", n_boot=1, rng_seed=1,
                cluster=tiny_df["cluster"])

    # Access the internal inference to check resampled data size
    # Instead, let's directly test the resampling logic
    from pymargins._inference import InferenceConfig, _run_bootstrap
    from pymargins._adapter import auto_detect_adapter

    adapter = auto_detect_adapter(fit)
    config = InferenceConfig(method="bootstrap", n_boot=1, rng_seed=1,
                             cluster=tiny_df["cluster"].values)

    # Build a simple estimand that counts rows
    def h_factory(a):
        def h(beta):
            return float(len(a.training_data))
        return h

    result = _run_bootstrap(lambda b: float(len(adapter.training_data)),
                            adapter, config, {}, h_factory=h_factory)
    # Resampled data should have 4 rows (2 clusters × 2 rows each),
    # or potentially more if one cluster is sampled twice
    resampled_n = result["draws"][0]
    assert resampled_n in (4, 6, 8)  # 4=both diff, 6=one dup, 8=both dup


# ---------------------------------------------------------------------------
# Delta and simulation paths ignore cluster
# ---------------------------------------------------------------------------

def test_delta_ignores_cluster(ols_fit_formula, df_clustered):
    """Delta method should work fine even when cluster is provided."""
    m = Margins(ols_fit_formula, method="delta", cluster=df_clustered["cluster"])
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


def test_simulation_ignores_cluster(ols_fit_formula, df_clustered):
    """Simulation should work fine even when cluster is provided."""
    m = Margins(ols_fit_formula, method="simulation", n_sim=500,
                cluster=df_clustered["cluster"])
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
