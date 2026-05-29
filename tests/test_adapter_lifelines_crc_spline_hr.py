"""Tests for LifelinesCRCSplineHRAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lifelines")
from lifelines import CRCSplineFitter

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapters.lifelines_crc_spline_hr import LifelinesCRCSplineHRAdapter


@pytest.fixture
def df_survival():
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
def crc_fit(df_survival):
    crc = CRCSplineFitter(n_baseline_knots=3)
    crc.fit(
        df_survival,
        duration_col="T",
        event_col="E",
        regressors={"beta_": "x1 + x2", "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"},
    )
    return crc


def test_predict_matches_manual(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    # Manual relative risk = exp(X @ beta_beta)
    beta_beta = crc_fit.params_.loc["beta_"].values
    X_np = np.asarray(X)
    manual_rr = np.exp(X_np @ beta_beta)
    np.testing.assert_allclose(our_pred, manual_rr, rtol=1e-4)


def test_coefficients_shape(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    # beta_ group: Intercept, x1, x2 = 3
    assert adapter.coefficients().shape == (3,)


def test_covariance_default(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (3, 3)


def test_design_matrix(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 3)  # Intercept, x1, x2


def test_supported_inference_methods(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    assert adapter.supported_inference_methods == {"delta", "simulation", "bootstrap"}


def test_delta_end_to_end(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    m = Margins(crc_fit, adapter=adapter, at="typical", method="delta")
    rd = m.predict()
    # High curvature may trigger fallback to simulation; both are valid
    assert rd.method in ("delta", "simulation")
    assert np.isfinite(float(rd.estimate))
    assert float(rd.estimate) > 0
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_refit(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesCRCSplineHRAdapter)


def test_gradient_backend_recommendation(crc_fit, df_survival):
    adapter = LifelinesCRCSplineHRAdapter(crc_fit, training_data=df_survival)
    assert adapter.gradient_backend_recommendation == "autodiff"
