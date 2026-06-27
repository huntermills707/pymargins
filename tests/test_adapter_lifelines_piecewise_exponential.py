"""Tests for LifelinesPiecewiseExponentialAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lifelines")
from lifelines import PiecewiseExponentialRegressionFitter

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.lifelines_piecewise_exponential import (
    LifelinesPiecewiseExponentialAdapter,
)


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
def pe_fit(df_survival):
    pe = PiecewiseExponentialRegressionFitter(breakpoints=[1.0, 2.0])
    pe.fit(df_survival, duration_col="T", event_col="E")
    return pe


def test_auto_detect_requires_training_data(pe_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(pe_fit)


def test_predict_matches_lifelines(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    t = adapter._prediction_time
    sm_pred = pe_fit.predict_survival_function(df[["x1", "x2"]].iloc[:10], times=[t])
    np.testing.assert_allclose(our_pred, sm_pred.values.flatten(), rtol=1e-4)


def test_coefficients_shape(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    # 3 intervals * 3 params (x1, x2, Intercept) = 9
    assert adapter.coefficients().shape == (9,)


def test_covariance_default(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (9, 9)


def test_design_matrix(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 3)  # x1, x2, Intercept


def test_supported_inference_methods(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    m = GComputation(
        pe_fit,
        adapter=adapter,
        at="typical",
        method="bootstrap",
        B=50,
        seed=42,
    )
    rd = m.predict()
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert 0 <= float(rd.estimate) <= 1
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_refit(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesPiecewiseExponentialAdapter)


def test_default_prediction_time(pe_fit, df_survival):
    adapter = LifelinesPiecewiseExponentialAdapter(pe_fit, training_data=df_survival)
    assert adapter._prediction_time > 0
