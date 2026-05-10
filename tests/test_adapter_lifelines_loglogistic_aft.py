"""Tests for LifelinesLogLogisticAFTAdapter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lifelines import LogLogisticAFTFitter

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.lifelines_loglogistic_aft import LifelinesLogLogisticAFTAdapter
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
def loglogistic_fit(df_survival):
    ll = LogLogisticAFTFitter()
    ll.fit(df_survival, duration_col="T", event_col="E")
    return ll


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_array_requires_training_data(loglogistic_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(loglogistic_fit)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_lifelines(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    # lifelines predict_survival_function at prediction_time
    t = adapter._prediction_time
    sm_pred = loglogistic_fit.predict_survival_function(
        df[["x1", "x2"]].iloc[:10], times=[t]
    ).values.flatten()
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_jax_differentiable(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
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

def test_coefficients_shape(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    # 3 alpha params (x1, x2, Intercept) + 1 beta param (Intercept)
    assert adapter.coefficients().shape == (4,)


def test_covariance_default(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (4, 4)


def test_covariance_rejects_hc(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    with pytest.raises(ValueError, match="only supports"):
        adapter.covariance("HC0")


def test_covariance_ndarray_override(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    custom = np.eye(4)
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------

def test_design_matrix(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    # 3 columns: x1, x2, Intercept
    assert X.shape == (5, 3)


def test_design_matrix_missing_columns(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    bad_df = df_survival[["T", "E"]].copy()
    with pytest.raises(ValueError, match="Missing columns"):
        adapter.design_matrix_from_df(bad_df)


def test_variable_metadata(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert meta["x1"].var_type == "continuous"


def test_variable_metadata_is_cached(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    meta1 = adapter.variable_metadata()
    meta2 = adapter.variable_metadata()
    assert meta1 is meta2


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------

def test_margins_predict(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    m = Margins(loglogistic_fit, adapter=adapter)
    res = m.predict()
    assert res.estimate.size == 1


def test_margins_dydx(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    m = Margins(loglogistic_fit, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Bootstrap end-to-end
# ---------------------------------------------------------------------------

def test_bootstrap_end_to_end(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    m = Margins(loglogistic_fit, adapter=adapter, at="typical",
                method="bootstrap", n_boot=50, rng_seed=42)
    rd = m.dydx("x1")
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


# ---------------------------------------------------------------------------
# Attach-time validation
# ---------------------------------------------------------------------------

def test_attach_rejects_unsupported_vcov_string(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    with pytest.raises(ValueError, match="LifelinesLogLogisticAFTAdapter only supports vcov=None"):
        Margins(loglogistic_fit, adapter=adapter, vcov="HC0")


def test_attach_rejects_unsupported_vcov_dict(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    with pytest.raises(ValueError, match="LifelinesLogLogisticAFTAdapter only supports vcov=None"):
        Margins(loglogistic_fit, adapter=adapter, vcov={"type": "cluster", "groups": df_survival["E"]})


def test_attach_accepts_supported_vcov(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    # default (None)
    m1 = Margins(loglogistic_fit, adapter=adapter)
    assert m1.vcov_spec is None
    # ndarray
    cov = np.eye(4)
    m2 = Margins(loglogistic_fit, adapter=adapter, vcov=cov)
    assert m2.vcov_spec is cov


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def test_refit_array(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesLogLogisticAFTAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_changes_coefficients(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    resampled = df_survival.sample(frac=1.0, replace=True, random_state=42).reset_index(drop=True)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LifelinesLogLogisticAFTAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Prediction time
# ---------------------------------------------------------------------------

def test_default_prediction_time(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(loglogistic_fit, training_data=df_survival)
    assert adapter._prediction_time > 0
    observed = df_survival.loc[df_survival["E"] == 1, "T"]
    np.testing.assert_allclose(adapter._prediction_time, np.median(observed), rtol=0.1)


def test_custom_prediction_time(loglogistic_fit, df_survival):
    adapter = LifelinesLogLogisticAFTAdapter(
        loglogistic_fit, training_data=df_survival, prediction_time=1.0
    )
    assert adapter._prediction_time == 1.0
