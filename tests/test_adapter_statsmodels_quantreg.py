"""Tests for StatsmodelsQuantRegAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_quantreg import StatsmodelsQuantRegAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_quantreg():
    """Synthetic data for quantile regression."""
    rng = np.random.default_rng(46)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    df["y"] = 1.0 + 2.0 * df["x1"] - 1.5 * df["x2"] + rng.standard_normal(n)
    return df


@pytest.fixture
def quantreg_fit_array(df_quantreg):
    return sm.QuantReg(
        df_quantreg["y"], sm.add_constant(df_quantreg[["x1", "x2"]])
    ).fit(q=0.5)


@pytest.fixture
def quantreg_fit_formula(df_quantreg):
    return smf.quantreg("y ~ x1 + x2", data=df_quantreg).fit(q=0.5)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_formula(quantreg_fit_formula):
    adapter = auto_detect_adapter(quantreg_fit_formula)
    assert isinstance(adapter, StatsmodelsQuantRegAdapter)


def test_auto_detect_array_requires_training_data(quantreg_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(quantreg_fit_array)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = quantreg_fit_formula.predict(df[:10])
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_jax_differentiable(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
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


def test_coefficients_shape(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    p = len(quantreg_fit_formula.model.exog_names)
    assert adapter.coefficients().shape == (p,)


def test_covariance_default(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


def test_covariance_rejects_hc(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    with pytest.raises(ValueError, match="vcov=None or a user-supplied ndarray"):
        adapter.covariance("HC0")


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    p = len(quantreg_fit_formula.model.exog_names)
    assert X.shape[1] == p


def test_variable_metadata(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------


def test_margins_predict_aap(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    m = Margins.linear_scale(quantreg_fit_formula, adapter=adapter)
    res = m.predict()
    assert res.estimate.size == 1


def test_margins_dydx(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    m = Margins.linear_scale(quantreg_fit_formula, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    new_adapter = adapter.refit(adapter.training_data)
    assert isinstance(new_adapter, StatsmodelsQuantRegAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array(quantreg_fit_array, df_quantreg):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_array, training_data=df_quantreg)
    new_adapter = adapter.refit(df_quantreg)
    assert isinstance(new_adapter, StatsmodelsQuantRegAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Quantile preservation
# ---------------------------------------------------------------------------


def test_quantile_preserved(quantreg_fit_formula):
    adapter = StatsmodelsQuantRegAdapter(quantreg_fit_formula)
    assert adapter._quantile == 0.5
