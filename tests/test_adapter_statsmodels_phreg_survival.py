"""Tests for StatsmodelsPHRegSurvivalAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapters.statsmodels_phreg_survival import (
    StatsmodelsPHRegSurvivalAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_survival():
    """Synthetic survival data."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    hazard = np.exp(0.5 + 0.3 * df["x1"] - 0.2 * df["x2"])
    df["T"] = rng.exponential(1.0 / hazard)
    df["E"] = (rng.random(n) < 0.8).astype(int)
    return df


@pytest.fixture
def phreg_fit(df_survival):
    return PHReg(
        df_survival["T"].values,
        df_survival[["x1", "x2"]].values,
        status=df_survival["E"].values,
    ).fit()


# ---------------------------------------------------------------------------
# Construction and prediction
# ---------------------------------------------------------------------------


def test_adapter_coefficients(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        phreg_fit.params,
        rtol=1e-10,
    )


def test_predict_shape(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    assert pred.shape == (5,)
    assert np.all((pred >= 0) & (pred <= 1))


def test_predict_matches_native(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    t = adapter._prediction_time
    native_pred = phreg_fit.model.predict(
        phreg_fit.params,
        endog=np.full(5, t),
        exog=df_survival[["x1", "x2"]].iloc[:5].values,
        pred_type="surv",
    )
    np.testing.assert_allclose(our_pred, native_pred.predicted_values, rtol=1e-5)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_covariance_default(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_covariance_rejects_hc(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    with pytest.raises(ValueError, match="only supports"):
        adapter.covariance("HC0")


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 2)


def test_variable_metadata(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


def test_supported_inference_methods(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    est = GComputation(
        adapter=adapter,
        at="typical",
        method="bootstrap",
        B=50,
        seed=42,
    )
    rd = est.predict()
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert 0 <= float(rd.estimate) <= 1
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


def test_bootstrap_dydx(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    est = GComputation(
        adapter=adapter,
        at="typical",
        method="bootstrap",
        B=50,
        seed=42,
    )
    rd = est.dydx("x1")
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, StatsmodelsPHRegSurvivalAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_changes_coefficients(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    resampled = df_survival.sample(frac=1.0, replace=True, random_state=42).reset_index(
        drop=True
    )
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsPHRegSurvivalAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Prediction time
# ---------------------------------------------------------------------------


def test_default_prediction_time(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(phreg_fit, training_data=df_survival)
    assert adapter._prediction_time > 0
    observed = df_survival.loc[df_survival["E"] == 1, "T"]
    np.testing.assert_allclose(adapter._prediction_time, np.median(observed), rtol=0.1)


def test_custom_prediction_time(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegSurvivalAdapter(
        phreg_fit, training_data=df_survival, prediction_time=1.0
    )
    assert adapter._prediction_time == 1.0
