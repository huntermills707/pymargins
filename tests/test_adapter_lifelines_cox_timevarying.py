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

from pymargins._adapters.lifelines_cox_timevarying import LifelinesCoxTimeVaryingSurvivalAdapter
from pymargins._adapter import auto_detect_adapter
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
    # Format as long-form time-varying (one row per subject for simplicity)
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


def test_auto_detect_requires_training_data(ctv_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(ctv_fit)


def test_predict_matches_lifelines(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    # CoxTimeVaryingFitter has no predict_survival_function;
    # compute manually from baseline survival and partial hazard
    t = adapter._prediction_time
    ph = ctv_fit.predict_partial_hazard(df[["x1", "x2"]].iloc[:10])
    S0 = ctv_fit.baseline_survival_
    col = S0.columns[0]
    times = S0.index.values
    surv = S0[col].values
    S0_t = float(np.interp(t, times, surv))
    sm_pred = S0_t ** ph.values.flatten()
    np.testing.assert_allclose(our_pred, sm_pred, rtol=1e-4)


def test_coefficients_shape(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter.coefficients().shape == (2,)


def test_covariance_default(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_design_matrix(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    X = adapter.design_matrix_from_df(df_survival_tv[:5])
    assert X.shape == (5, 2)


def test_supported_inference_methods(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    m = Margins(ctv_fit, adapter=adapter, at="typical",
                method="bootstrap", n_boot=50, rng_seed=42)
    rd = m.predict()
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert 0 <= float(rd.estimate) <= 1
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_refit(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    new_adapter = adapter.refit(df_survival_tv)
    assert isinstance(new_adapter, LifelinesCoxTimeVaryingSurvivalAdapter)


def test_default_prediction_time(ctv_fit, df_survival_tv):
    adapter = LifelinesCoxTimeVaryingSurvivalAdapter(ctv_fit, training_data=df_survival_tv)
    assert adapter._prediction_time > 0
