"""Tests for StatsmodelsMixedLMAdapter.

See IMPLEMENTATION_GUIDE.md §0.3.
"""

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_mixedlm import StatsmodelsMixedLMAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_mixed():
    """Synthetic data with clusters for mixed models."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
            "group": np.repeat(np.arange(20), 10),
        }
    )
    # Random intercept per group
    group_effect = rng.standard_normal(20)[df["group"].values]
    df["y"] = (
        1.0
        + 0.5 * df["x1"]
        - 0.3 * df["x2"]
        + 0.8 * df["treatment"]
        + group_effect
        + rng.standard_normal(n) * 0.5
    )
    return df


@pytest.fixture
def fit_mixed_formula(df_mixed):
    return smf.mixedlm(
        "y ~ x1 + x2 + treatment + C(region)", groups="group", data=df_mixed
    ).fit()


@pytest.fixture
def fit_mixed_array(df_mixed):
    X = df_mixed[["x1", "x2", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_mixed["y"].values
    groups = df_mixed["group"].values
    return sm.MixedLM(y, X, groups=groups).fit()


# ---------------------------------------------------------------------------
# 1. Construction and auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_mixedlm(fit_mixed_formula):
    adapter = auto_detect_adapter(fit_mixed_formula)
    assert isinstance(adapter, StatsmodelsMixedLMAdapter)


def test_adapter_coefficients(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        fit_mixed_formula.fe_params.values,
        rtol=1e-10,
    )


def test_adapter_training_data_formula(fit_mixed_formula, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    pd.testing.assert_frame_equal(adapter.training_data, df_mixed)


def test_adapter_training_data_array_requires_explicit(fit_mixed_array, df_mixed):
    with pytest.raises(ValueError, match="training_data must be provided"):
        StatsmodelsMixedLMAdapter(fit_mixed_array)
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_array, training_data=df_mixed)
    assert adapter.training_data is df_mixed


# ---------------------------------------------------------------------------
# 2. Covariance / vcov flavors
# ---------------------------------------------------------------------------


def test_covariance_default(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    Sigma = adapter.covariance()
    n_fe = len(fit_mixed_formula.fe_params)
    assert Sigma.shape == (n_fe, n_fe)
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_mixed_formula.cov_params().values[:n_fe, :n_fe],
        rtol=1e-10,
    )


def test_covariance_matches_bse_fe(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    Sigma = adapter.covariance()
    np.testing.assert_allclose(
        np.diag(np.asarray(Sigma)),
        fit_mixed_formula.bse_fe.values**2,
        rtol=1e-10,
    )


def test_covariance_ndarray_override(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    n_fe = len(fit_mixed_formula.fe_params)
    custom = np.eye(n_fe)
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


def test_covariance_rejects_hc(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    with pytest.raises(ValueError, match="does not support vcov='HC0'"):
        adapter.covariance("HC0")


def test_covariance_rejects_cluster_dict(fit_mixed_formula, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    groups = np.arange(len(df_mixed)) % 10
    with pytest.raises(ValueError, match="does not support vcov dict"):
        adapter.covariance({"type": "cluster", "groups": groups})


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels_pa(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    mu_sm = fit_mixed_formula.predict(adapter.training_data.iloc[:5])
    np.testing.assert_allclose(np.asarray(mu), mu_sm.values, rtol=1e-10)


def test_predict_is_linear(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    expected = np.asarray(X) @ np.asarray(beta)
    np.testing.assert_allclose(np.asarray(mu), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Design matrix construction
# ---------------------------------------------------------------------------


def test_design_matrix_from_df_formula(fit_mixed_formula, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    X = adapter.design_matrix_from_df(df_mixed.iloc[:5])
    assert X.ndim == 2
    assert X.shape[0] == 5
    assert X.shape[1] == len(fit_mixed_formula.model.exog_names)


def test_design_matrix_from_df_array(fit_mixed_array, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_array, training_data=df_mixed)
    subset = df_mixed[["x1", "x2", "treatment"]].iloc[:5].copy()
    subset.insert(0, "const", 1.0)
    X = adapter.design_matrix_from_df(subset)
    assert X.ndim == 2
    assert X.shape == (5, 4)


# ---------------------------------------------------------------------------
# 5. Variable metadata
# ---------------------------------------------------------------------------


def test_variable_metadata(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
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


def test_column_index_continuous(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)
    assert adapter._exog_names[idx] == "x1"


def test_column_index_categorical_raises(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    with pytest.raises(ValueError, match="categorical"):
        adapter.column_index_of_variable("region")


# ---------------------------------------------------------------------------
# 7. Bootstrap / refit
# ---------------------------------------------------------------------------


def test_refit_formula(fit_mixed_formula, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    resampled = df_mixed.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsMixedLMAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_array(fit_mixed_array, df_mixed):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_array, training_data=df_mixed)
    resampled = df_mixed.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsMixedLMAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Attach-time validation
# ---------------------------------------------------------------------------


def test_attach_rejects_unsupported_vcov_string(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    with pytest.raises(ValueError, match="Unsupported vcov string: 'HAC'"):
        GComputation(fit_mixed_formula, adapter=adapter, vcov="HAC")


def test_attach_rejects_unsupported_vcov_dict(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    with pytest.raises(
        ValueError,
        match="StatsmodelsMixedLMAdapter does not support vcov dict with type='hac'",
    ):
        GComputation(fit_mixed_formula, adapter=adapter, vcov={"type": "hac"})


def test_attach_rejects_cluster_without_groups(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    with pytest.raises(
        ValueError, match="does not support vcov dict with type='cluster'"
    ):
        GComputation(fit_mixed_formula, adapter=adapter, vcov={"type": "cluster"})


def test_attach_accepts_supported_vcov(fit_mixed_formula):
    adapter = StatsmodelsMixedLMAdapter(fit_mixed_formula)
    # ndarray only
    cov = np.eye(len(fit_mixed_formula.fe_params))
    est = GComputation(fit_mixed_formula, adapter=adapter, vcov=cov)
    np.testing.assert_allclose(np.asarray(est._compiled.frozen_cov), cov, rtol=1e-10)
