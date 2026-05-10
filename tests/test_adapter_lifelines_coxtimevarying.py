"""Tests for LifelinesCoxTimeVaryingAdapter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("lifelines")
from lifelines import CoxTimeVaryingFitter

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.lifelines_coxtimevarying import LifelinesCoxTimeVaryingAdapter
from pymargins import Margins


@pytest.fixture
def df_survival_tv():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    hazard = np.exp(0.5 + 0.3 * df["x1"] - 0.2 * df["x2"])
    T = rng.exponential(1.0 / hazard)
    E = (rng.random(n) < 0.8).astype(int)
    df["id"] = np.arange(n)
    df["start"] = 0.0
    df["stop"] = T
    df["E"] = E
    return df


@pytest.fixture
def ctv_fit(df_survival_tv):
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(df_survival_tv, id_col="id", start_col="start", stop_col="stop", event_col="E")
    return ctv


def test_predict_matches_lifelines(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    sm_pred = ctv_fit.predict_partial_hazard(df[["x1", "x2"]].iloc[:10])
    np.testing.assert_allclose(our_pred, sm_pred.values.flatten(), rtol=1e-4)


def test_coefficients_shape(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter.coefficients().shape == (2,)


def test_covariance_default(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_design_matrix(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    X = adapter.design_matrix_from_df(df_survival_tv[:5])
    assert X.shape == (5, 2)


def test_supported_inference_methods(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter.supported_inference_methods == {"delta", "simulation", "bootstrap"}


def test_delta_end_to_end(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    m = Margins(ctv_fit, adapter=adapter, at="typical", method="delta")
    rd = m.predict()
    assert rd.method == "delta"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.estimate) > 0
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_refit(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    new_adapter = adapter.refit(df_survival_tv)
    assert isinstance(new_adapter, LifelinesCoxTimeVaryingAdapter)


def test_gradient_backend_recommendation(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter.gradient_backend_recommendation == "autodiff"
