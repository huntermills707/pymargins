"""Tests for LifelinesCRCSplineAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lifelines")
from lifelines import CRCSplineFitter

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.lifelines_crc_spline import LifelinesCRCSplineAdapter


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


def test_auto_detect_requires_training_data(crc_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(crc_fit)


def test_predict_matches_lifelines(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    t = adapter._prediction_time
    sm_pred = crc_fit.predict_survival_function(df[["x1", "x2"]].iloc[:10], times=[t])
    np.testing.assert_allclose(our_pred, sm_pred.values.flatten(), rtol=1e-4)


def test_coefficients_shape(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    # beta_ (3) + gamma0_ (1) + gamma1_ (1) + gamma2_ (1) = 6
    assert adapter.coefficients().shape == (6,)


def test_covariance_default(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (6, 6)


def test_design_matrix(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 3)  # x1, x2, Intercept


def test_supported_inference_methods(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    est = GComputation(
        crc_fit,
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


def test_refit(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesCRCSplineAdapter)


def test_default_prediction_time(crc_fit, df_survival):
    adapter = LifelinesCRCSplineAdapter(crc_fit, training_data=df_survival)
    assert adapter._prediction_time > 0
