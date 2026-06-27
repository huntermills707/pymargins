"""Tests for StatsmodelsRLMAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_rlm import StatsmodelsRLMAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_rlm():
    """Synthetic data for RLM."""
    rng = np.random.default_rng(45)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    df["y"] = 1.0 + 2.0 * df["x1"] - 1.5 * df["x2"] + rng.standard_normal(n)
    # Add some outliers
    df.loc[rng.choice(n, size=10, replace=False), "y"] += 20.0
    return df


@pytest.fixture
def rlm_fit_array(df_rlm):
    return sm.RLM(df_rlm["y"], sm.add_constant(df_rlm[["x1", "x2"]])).fit()


@pytest.fixture
def rlm_fit_formula(df_rlm):
    return smf.rlm("y ~ x1 + x2", data=df_rlm).fit()


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_formula(rlm_fit_formula):
    adapter = auto_detect_adapter(rlm_fit_formula)
    assert isinstance(adapter, StatsmodelsRLMAdapter)


def test_auto_detect_array_requires_training_data(rlm_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(rlm_fit_array)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = rlm_fit_formula.predict(df[:10])
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_jax_differentiable(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    beta = adapter.coefficients()

    def mean_pred(b):
        return jnp.mean(adapter.predict(b, X))

    grad = jax.grad(mean_pred)(beta)
    assert grad.shape == beta.shape


# ---------------------------------------------------------------------------
# Coefficients and covariance
# ---------------------------------------------------------------------------


def test_coefficients_shape(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    p = len(rlm_fit_formula.model.exog_names)
    assert adapter.coefficients().shape == (p,)


def test_covariance_default(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


def test_covariance_user_supplied(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    V = np.eye(adapter.coefficients().shape[0])
    cov = adapter.covariance(V)
    np.testing.assert_array_equal(cov, V)


def test_covariance_rejects_hc(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    with pytest.raises(ValueError, match="vcov=None or a user-supplied ndarray"):
        adapter.covariance("HC0")


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    p = len(rlm_fit_formula.model.exog_names)
    assert X.shape[1] == p


def test_variable_metadata(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


# ---------------------------------------------------------------------------
# End-to-end via GComputation
# ---------------------------------------------------------------------------


def test_margins_predict_aap(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    est = GComputation(rlm_fit_formula, adapter=adapter)
    res = est.predict()
    assert res.estimate.size == 1


def test_margins_dydx(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    est = GComputation(rlm_fit_formula, adapter=adapter)
    res = est.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula(rlm_fit_formula):
    adapter = StatsmodelsRLMAdapter(rlm_fit_formula)
    new_adapter = adapter.refit(adapter.training_data)
    assert isinstance(new_adapter, StatsmodelsRLMAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array(rlm_fit_array, df_rlm):
    adapter = StatsmodelsRLMAdapter(rlm_fit_array, training_data=df_rlm)
    new_adapter = adapter.refit(df_rlm)
    assert isinstance(new_adapter, StatsmodelsRLMAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )
