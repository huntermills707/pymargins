"""Tests for StatsmodelsGEEAdapter.

See IMPLEMENTATION_GUIDE.md §0.3.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.statsmodels_gee import StatsmodelsGEEAdapter
from pymargins._adapter import auto_detect_adapter
from pymargins import Margins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_binary():
    """Synthetic data with a binary outcome and clusters."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "treatment": rng.binomial(1, 0.5, n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "group": np.repeat(np.arange(20), 10),
    })
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.8 * df["treatment"]
    df["y"] = (rng.uniform(size=n) < (1 / (1 + np.exp(-eta)))).astype(float)
    return df


@pytest.fixture
def df_count():
    """Synthetic data with a count outcome and clusters."""
    rng = np.random.default_rng(43)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "group": np.repeat(np.arange(20), 10),
    })
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    df["y"] = rng.poisson(np.exp(eta))
    return df


@pytest.fixture
def fit_gee_logit_formula(df_binary):
    fit = smf.gee(
        "y ~ x1 + x2 + treatment + C(region)",
        groups="group",
        data=df_binary,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    return fit


@pytest.fixture
def fit_gee_logit_array(df_binary):
    X = df_binary[["x1", "x2", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_binary["y"].values
    groups = df_binary["group"].values
    fit = sm.GEE(
        y, X, groups=groups,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    return fit


@pytest.fixture
def fit_gee_poisson_formula(df_count):
    fit = smf.gee(
        "y ~ x1 + x2",
        groups="group",
        data=df_count,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    return fit


# ---------------------------------------------------------------------------
# 1. Construction and auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_gee_logit(fit_gee_logit_formula):
    adapter = auto_detect_adapter(fit_gee_logit_formula)
    assert isinstance(adapter, StatsmodelsGEEAdapter)


def test_adapter_coefficients(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        fit_gee_logit_formula.params.values,
        rtol=1e-10,
    )


def test_adapter_training_data_formula(fit_gee_logit_formula, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    pd.testing.assert_frame_equal(adapter.training_data, df_binary)


def test_adapter_training_data_array_requires_explicit(fit_gee_logit_array, df_binary):
    with pytest.raises(ValueError, match="training_data must be provided"):
        StatsmodelsGEEAdapter(fit_gee_logit_array)
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_array, training_data=df_binary)
    assert adapter.training_data is df_binary


# ---------------------------------------------------------------------------
# 2. Covariance / vcov flavors
# ---------------------------------------------------------------------------

def test_covariance_default(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    Sigma = adapter.covariance()
    assert Sigma.shape == (len(fit_gee_logit_formula.params),) * 2
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_gee_logit_formula.cov_params().values,
        rtol=1e-10,
    )


def test_covariance_default_is_robust(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    Sigma = adapter.covariance()
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_gee_logit_formula.cov_robust,
        rtol=1e-10,
    )


def test_covariance_naive(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    Sigma = adapter.covariance("naive")
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_gee_logit_formula.cov_naive,
        rtol=1e-10,
    )


def test_covariance_robust(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    Sigma = adapter.covariance("robust")
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_gee_logit_formula.cov_robust,
        rtol=1e-10,
    )


def test_covariance_robust_bc_unavailable(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    if fit_gee_logit_formula.cov_robust_bc is None:
        with pytest.raises(ValueError, match="cov_robust_bc is not available"):
            adapter.covariance("robust_bc")
    else:
        Sigma = adapter.covariance("robust_bc")
        np.testing.assert_allclose(
            np.asarray(Sigma),
            fit_gee_logit_formula.cov_robust_bc,
            rtol=1e-10,
        )


def test_covariance_ndarray_override(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    custom = np.eye(len(fit_gee_logit_formula.params))
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


def test_covariance_cluster_returns_robust(fit_gee_logit_formula, df_binary):
    """GEE's robust covariance is inherently cluster-robust."""
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    groups = np.arange(len(df_binary)) % 10
    Sigma = adapter.covariance({"type": "cluster", "groups": groups})
    assert Sigma.ndim == 2
    assert Sigma.shape[0] == len(fit_gee_logit_formula.params)
    np.testing.assert_allclose(np.asarray(Sigma), fit_gee_logit_formula.cov_robust, rtol=1e-10)


def test_covariance_cluster_groups_missing(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires 'groups'"):
        adapter.covariance({"type": "cluster"})


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_statsmodels(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    mu_sm = fit_gee_logit_formula.predict(adapter.training_data.iloc[:5])
    np.testing.assert_allclose(np.asarray(mu), mu_sm.values, rtol=1e-10)


def test_predict_poisson_matches_statsmodels(fit_gee_poisson_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_poisson_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    mu_sm = fit_gee_poisson_formula.predict(adapter.training_data.iloc[:5])
    np.testing.assert_allclose(np.asarray(mu), mu_sm.values, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Design matrix construction
# ---------------------------------------------------------------------------

def test_design_matrix_from_df_formula(fit_gee_logit_formula, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    X = adapter.design_matrix_from_df(df_binary.iloc[:5])
    assert X.ndim == 2
    assert X.shape[0] == 5
    assert X.shape[1] == len(fit_gee_logit_formula.model.exog_names)


def test_design_matrix_from_df_array(fit_gee_logit_array, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_array, training_data=df_binary)
    subset = df_binary[["x1", "x2", "treatment"]].iloc[:5].copy()
    subset.insert(0, "const", 1.0)
    X = adapter.design_matrix_from_df(subset)
    assert X.ndim == 2
    assert X.shape == (5, 4)


# ---------------------------------------------------------------------------
# 5. Variable metadata
# ---------------------------------------------------------------------------

def test_variable_metadata(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert "treatment" in meta
    assert "region" in meta
    assert meta["x1"].var_type == "continuous"
    assert meta["treatment"].var_type == "binary"
    assert meta["region"].var_type == "categorical"


# ---------------------------------------------------------------------------
# 6. Column index lookup
# ---------------------------------------------------------------------------

def test_column_index_continuous(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)
    assert adapter._exog_names[idx] == "x1"


def test_column_index_categorical_raises(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    with pytest.raises(ValueError, match="categorical"):
        adapter.column_index_of_variable("region")


# ---------------------------------------------------------------------------
# 7. Bootstrap / refit
# ---------------------------------------------------------------------------

def test_refit_formula(fit_gee_logit_formula, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    resampled = df_binary.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsGEEAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_array(fit_gee_logit_array, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_array, training_data=df_binary)
    resampled = df_binary.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsGEEAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_preserves_cov_struct(fit_gee_logit_formula, df_binary):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    resampled = df_binary.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert type(new_adapter._cov_struct) == type(adapter._cov_struct)


# ---------------------------------------------------------------------------
# Attach-time validation
# ---------------------------------------------------------------------------

def test_attach_rejects_unsupported_vcov_string(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    with pytest.raises(ValueError, match="StatsmodelsGEEAdapter does not support vcov='HAC'"):
        Margins(fit_gee_logit_formula, adapter=adapter, vcov="HAC")


def test_attach_rejects_unsupported_vcov_dict(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    with pytest.raises(ValueError, match="StatsmodelsGEEAdapter does not support vcov dict with type='hac'"):
        Margins(fit_gee_logit_formula, adapter=adapter, vcov={"type": "hac"})


def test_attach_rejects_cluster_without_groups(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires 'groups'"):
        Margins(fit_gee_logit_formula, adapter=adapter, vcov={"type": "cluster"})


def test_attach_accepts_supported_vcov(fit_gee_logit_formula):
    adapter = StatsmodelsGEEAdapter(fit_gee_logit_formula)
    # naive string
    m = Margins(fit_gee_logit_formula, adapter=adapter, vcov="naive")
    assert m.vcov_spec == "naive"
    # robust string
    m2 = Margins(fit_gee_logit_formula, adapter=adapter, vcov="robust")
    assert m2.vcov_spec == "robust"
    # ndarray
    cov = np.eye(len(fit_gee_logit_formula.params))
    m3 = Margins(fit_gee_logit_formula, adapter=adapter, vcov=cov)
    assert m3.vcov_spec is cov
