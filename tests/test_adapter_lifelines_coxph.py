"""Tests for LifelinesCoxPHAdapter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.lifelines_coxph import LifelinesCoxPHAdapter
from pymargins._adapter import auto_detect_adapter
from pymargins import Margins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_survival():
    """Synthetic survival data."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    hazard = np.exp(0.5 + 0.3 * df["x1"] - 0.2 * df["x2"])
    df["T"] = rng.exponential(1.0 / hazard)
    df["E"] = (rng.random(n) < 0.8).astype(int)
    return df


@pytest.fixture
def coxph_fit_formula(df_survival):
    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col="T", event_col="E", formula="x1 + x2")
    return cph


@pytest.fixture
def coxph_fit_array(df_survival):
    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col="T", event_col="E")
    return cph


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_formula_requires_training_data(coxph_fit_formula):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(coxph_fit_formula)


def test_auto_detect_array_requires_training_data(coxph_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(coxph_fit_array)


# ---------------------------------------------------------------------------
# Construction and core data access
# ---------------------------------------------------------------------------

def test_adapter_coefficients(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        coxph_fit_formula.params_.values,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_lifelines_formula(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    # lifelines predict_partial_hazard returns exp((X - X_mean) @ beta)
    sm_pred = coxph_fit_formula.predict_partial_hazard(df[["x1", "x2"]].iloc[:10])
    np.testing.assert_allclose(our_pred, sm_pred.values, atol=1e-6)


def test_predict_matches_lifelines_array(coxph_fit_array, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_array, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = coxph_fit_array.predict_partial_hazard(df[["x1", "x2"]].iloc[:10])
    np.testing.assert_allclose(our_pred, sm_pred.values, atol=1e-6)


def test_predict_jax_differentiable(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
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

def test_coefficients_shape(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    assert adapter.coefficients().shape == (2,)


def test_covariance_default(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_covariance_rejects_hc(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    with pytest.raises(ValueError, match="does not support"):
        adapter.covariance("HC0")


def test_covariance_ndarray_override(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    custom = np.eye(len(coxph_fit_formula.params_))
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------

def test_design_matrix_formula(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    assert X.shape == (5, 2)


def test_design_matrix_array(coxph_fit_array, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_array, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    assert X.shape == (5, 2)


def test_design_matrix_missing_columns(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    bad_df = df_survival[["T", "E"]].copy()
    with pytest.raises(Exception):  # patsy raises PatsyError for formula fit
        adapter.design_matrix_from_df(bad_df)


def test_variable_metadata(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert meta["x1"].var_type == "continuous"


def test_variable_metadata_is_cached(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    meta1 = adapter.variable_metadata()
    meta2 = adapter.variable_metadata()
    assert meta1 is meta2


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------

def test_margins_predict(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    m = Margins.log_scale(coxph_fit_formula, adapter=adapter)
    res = m.predict()
    assert res.estimate.size == 1


def test_margins_dydx(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    m = Margins.log_scale(coxph_fit_formula, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Bootstrap end-to-end
# ---------------------------------------------------------------------------

def test_bootstrap_end_to_end(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    m = Margins.log_scale(coxph_fit_formula, adapter=adapter, at="typical",
                          method="bootstrap", n_boot=50, rng_seed=42)
    rd = m.dydx("x1")
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


# ---------------------------------------------------------------------------
# Attach-time validation
# ---------------------------------------------------------------------------

def test_attach_rejects_unsupported_vcov_string(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    with pytest.raises(ValueError, match="LifelinesCoxPHAdapter does not support vcov='HC0'"):
        Margins(coxph_fit_formula, adapter=adapter, vcov="HC0")


def test_attach_rejects_unsupported_vcov_dict(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    with pytest.raises(ValueError, match="LifelinesCoxPHAdapter does not support vcov dict"):
        Margins(coxph_fit_formula, adapter=adapter, vcov={"type": "cluster", "groups": df_survival["E"]})


def test_attach_accepts_supported_vcov(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    # default (None)
    m1 = Margins.log_scale(coxph_fit_formula, adapter=adapter)
    assert m1.vcov_spec is None
    # ndarray
    cov = np.eye(len(coxph_fit_formula.params_))
    m2 = Margins(coxph_fit_formula, adapter=adapter, vcov=cov)
    assert m2.vcov_spec is cov


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def test_refit_formula(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesCoxPHAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array(coxph_fit_array, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_array, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesCoxPHAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_changes_coefficients(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHAdapter(coxph_fit_formula, training_data=df_survival)
    resampled = df_survival.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LifelinesCoxPHAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )
