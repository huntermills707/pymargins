"""Tests for LifelinesGeneralizedGammaAdapter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lifelines import GeneralizedGammaRegressionFitter

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.lifelines_generalized_gamma import LifelinesGeneralizedGammaAdapter
from pymargins._adapter import auto_detect_adapter
from pymargins import Margins


@pytest.fixture
def df_survival():
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
def gg_fit(df_survival):
    gg = GeneralizedGammaRegressionFitter()
    gg.fit(df_survival, duration_col="T", event_col="E")
    return gg


def test_auto_detect_requires_training_data(gg_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(gg_fit)


def test_predict_matches_lifelines(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    t = adapter._prediction_time
    sm_pred = gg_fit.predict_survival_function(df[["x1", "x2"]].iloc[:10], times=[t])
    np.testing.assert_allclose(our_pred, sm_pred.values.flatten(), rtol=1e-4)


def test_coefficients_shape(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    assert adapter.coefficients().shape == (6,)  # sigma_x1, sigma_x2, mu_x1, mu_x2, lambda_x1, lambda_x2


def test_covariance_default(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (6, 6)


def test_design_matrix(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 2)


def test_supported_inference_methods(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    m = Margins(gg_fit, adapter=adapter, at="typical",
                method="bootstrap", n_boot=50, rng_seed=42)
    rd = m.predict()
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert 0 <= float(rd.estimate) <= 1
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_refit(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesGeneralizedGammaAdapter)


def test_default_prediction_time(gg_fit, df_survival):
    adapter = LifelinesGeneralizedGammaAdapter(gg_fit, training_data=df_survival)
    assert adapter._prediction_time > 0
