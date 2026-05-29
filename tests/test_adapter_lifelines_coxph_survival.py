"""Tests for LifelinesCoxPHSurvivalAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lifelines")
from lifelines import CoxPHFitter

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapters.lifelines_coxph_survival import LifelinesCoxPHSurvivalAdapter

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
def coxph_fit_formula(df_survival):
    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col="T", event_col="E", formula="x1 + x2")
    return cph


# ---------------------------------------------------------------------------
# Construction and prediction
# ---------------------------------------------------------------------------


def test_adapter_coefficients(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        coxph_fit_formula.params_.values,
        rtol=1e-10,
    )


def test_predict_matches_lifelines(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))

    t = adapter._prediction_time
    sm_pred = coxph_fit_formula.predict_survival_function(
        df[["x1", "x2"]].iloc[:10], times=[t]
    ).values.flatten()
    np.testing.assert_allclose(our_pred, sm_pred, rtol=1e-5)


def test_predict_shape(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    X = adapter.design_matrix_from_df(df_survival[:5])
    pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    assert pred.shape == (5,)
    assert np.all((pred >= 0) & (pred <= 1))


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_covariance_default(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_covariance_rejects_hc(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    with pytest.raises(ValueError, match="only supports"):
        adapter.covariance("HC0")


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    assert X.shape == (5, 2)


def test_variable_metadata(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


def test_supported_inference_methods(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    assert adapter.supported_inference_methods == {"bootstrap"}


def test_bootstrap_end_to_end(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    m = Margins(
        coxph_fit_formula,
        adapter=adapter,
        at="typical",
        method="bootstrap",
        n_boot=50,
        rng_seed=42,
    )
    rd = m.predict()
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert 0 <= float(rd.estimate) <= 1
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


def test_bootstrap_dydx(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    m = Margins(
        coxph_fit_formula,
        adapter=adapter,
        at="typical",
        method="bootstrap",
        n_boot=50,
        rng_seed=42,
    )
    rd = m.dydx("x1")
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, LifelinesCoxPHSurvivalAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_changes_coefficients(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    resampled = df_survival.sample(frac=1.0, replace=True, random_state=42).reset_index(
        drop=True
    )
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LifelinesCoxPHSurvivalAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Prediction time
# ---------------------------------------------------------------------------


def test_default_prediction_time(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival
    )
    assert adapter._prediction_time > 0
    observed = df_survival.loc[df_survival["E"] == 1, "T"]
    np.testing.assert_allclose(adapter._prediction_time, np.median(observed), rtol=0.1)


def test_custom_prediction_time(coxph_fit_formula, df_survival):
    adapter = LifelinesCoxPHSurvivalAdapter(
        coxph_fit_formula, training_data=df_survival, prediction_time=1.0
    )
    assert adapter._prediction_time == 1.0
