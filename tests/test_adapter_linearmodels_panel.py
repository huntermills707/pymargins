"""Tests for linearmodels panel adapters."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from linearmodels.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)

from pymargins import GComputation
from pymargins._adapters.linearmodels_panel import LinearmodelsPanelAdapter


@pytest.fixture
def panel_data():
    np.random.seed(42)
    n = 100
    t = 5
    entities = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    df = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": np.random.randn(n * t) + 0.5 * entities + 0.3 * times,
            "x1": np.random.randn(n * t),
            "x2": np.random.randn(n * t),
        }
    ).set_index(["entity", "time"])
    return df


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def test_adapter_from_panelols(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_from_pooledols(panel_data):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_from_random_effects(panel_data):
    mod = RandomEffects.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_from_first_difference(panel_data):
    mod = FirstDifferenceOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_from_between(panel_data):
    mod = BetweenOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def test_design_matrix_from_df(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    X = adapter.design_matrix_from_df(panel_data)
    assert X.shape == (len(panel_data), 2)
    assert jnp.allclose(X[:, 0], jnp.asarray(panel_data["x1"].values))


def test_design_matrix_injects_intercept(panel_data):
    mod = PooledOLS.from_formula("y ~ 1 + x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    assert "Intercept" in adapter._exog_names
    df = panel_data.reset_index(drop=True)[["x1", "x2"]]
    X = adapter.design_matrix_from_df(df)
    assert X.shape == (len(df), 3)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_native(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    X = adapter.design_matrix_from_df(panel_data)
    beta = adapter.coefficients()
    preds = adapter.predict(beta, X)
    native_preds = res.predict(exog=panel_data[["x1", "x2"]])
    assert jnp.allclose(preds, jnp.asarray(native_preds.values).ravel(), atol=1e-4)


def test_predict_matches_native_pooled(panel_data):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    X = adapter.design_matrix_from_df(panel_data)
    beta = adapter.coefficients()
    preds = adapter.predict(beta, X)
    native_preds = res.predict(exog=panel_data[["x1", "x2"]])
    assert jnp.allclose(preds, jnp.asarray(native_preds.values).ravel(), atol=1e-4)


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------


def test_variable_metadata(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta


# ---------------------------------------------------------------------------
# Bootstrap refit
# ---------------------------------------------------------------------------


def test_refit(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    resampled = panel_data.sample(frac=1.0, replace=True)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LinearmodelsPanelAdapter)
    assert new_adapter.coefficients().shape == adapter.coefficients().shape


def test_refit_pooled(panel_data):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res)
    resampled = panel_data.sample(frac=1.0, replace=True)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LinearmodelsPanelAdapter)


# ---------------------------------------------------------------------------
# End-to-end via GComputation
# ---------------------------------------------------------------------------


def test_margins_predict_panelols(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    m = GComputation(res)
    pred = m.predict()
    assert pred.estimate is not None
    assert pred.std_error is not None


def test_margins_dydx_panelols(panel_data):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    m = GComputation(res)
    slope = m.dydx("x1")
    assert slope.estimate is not None
    assert slope.std_error is not None


def test_margins_predict_pooledols(panel_data):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    m = GComputation(res)
    pred = m.predict()
    assert pred.estimate is not None
    assert pred.std_error is not None


def test_margins_contrasts_pooledols(panel_data):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    m = GComputation(res)
    c = m.contrasts(
        scenarios=[
            {"atexog": {"x1": panel_data["x1"].quantile(0.25)}},
            {"atexog": {"x1": panel_data["x1"].quantile(0.75)}},
        ],
        contrasts=[+1, -1],
    )
    assert c.estimate is not None


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_panelols(panel_data):
    from pymargins._adapters import _detect_adapter_class

    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_data)
    res = mod.fit()
    cls = _detect_adapter_class(res)
    assert cls is LinearmodelsPanelAdapter


def test_auto_detect_pooledols(panel_data):
    from pymargins._adapters import _detect_adapter_class

    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_data)
    res = mod.fit()
    cls = _detect_adapter_class(res)
    assert cls is LinearmodelsPanelAdapter


# ---------------------------------------------------------------------------
# Formula interface (Track B)
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_poly_data():
    rng = np.random.default_rng(42)
    n = 100
    t = 5
    entities = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    age = rng.normal(50, 10, size=n * t)
    df = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "age": age,
            "y": 0.2 * age + 0.01 * (age**2) + 0.5 * entities + rng.normal(n * t),
        }
    ).set_index(["entity", "time"])
    return df


def test_formula_verification_passes_panelols(panel_poly_data):
    """B.4 verification succeeds for a correctly specified formula (no intercept)."""
    from pymargins._formula import FormulaSpec

    mod = PanelOLS.from_formula(
        "y ~ age + I(age**2) + EntityEffects", data=panel_poly_data
    )
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res, training_data=panel_poly_data)
    # EntityEffects absorbs the intercept, so formula must suppress it
    spec = FormulaSpec("y ~ 0 + age + I(age**2)", panel_poly_data)
    spec.verify_against(adapter)


def test_formula_verification_fails_on_wrong_formula_panelols(panel_poly_data):
    """B.4 verification fails when the formula includes an absorbed intercept."""
    from pymargins._formula import FormulaSpec

    mod = PanelOLS.from_formula(
        "y ~ age + I(age**2) + EntityEffects", data=panel_poly_data
    )
    res = mod.fit()
    adapter = LinearmodelsPanelAdapter(res, training_data=panel_poly_data)
    # Wrong: includes Intercept which the model absorbed
    spec = FormulaSpec("y ~ age + I(age**2)", panel_poly_data)
    with pytest.raises(ValueError, match="Formula verification failed"):
        spec.verify_against(adapter)


def test_dydx_with_formula_panelols(panel_poly_data):
    """B.6 Acceptance #2: PanelOLS with I(age**2) yields correct dydx via formula=."""
    mod = PanelOLS.from_formula(
        "y ~ age + I(age**2) + EntityEffects", data=panel_poly_data
    )
    res = mod.fit()
    # Use a larger fd_step to avoid float32 precision loss in the quadratic
    # finite-difference at age ~ 50 (default 1e-6 is too small for float32).
    adapter = LinearmodelsPanelAdapter(
        res,
        training_data=panel_poly_data,
        formula="y ~ 0 + age + I(age**2)",
    )
    m = GComputation(adapter, at="mean", fd_step=1e-4)
    slope = m.dydx("age")
    expected = (
        res.params["age"]
        + 2 * res.params["I(age ** 2)"] * panel_poly_data["age"].mean()
    )
    assert np.isclose(float(slope.estimate), expected, rtol=1e-3)
    assert float(slope.std_error) > 0
