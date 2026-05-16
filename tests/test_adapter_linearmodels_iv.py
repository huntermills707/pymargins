"""Tests for linearmodels IV adapters."""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from linearmodels.iv import IV2SLS, IVGMM, IVLIML

from pymargins import Margins
from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter


@pytest.fixture
def iv_data():
    np.random.seed(42)
    n = 200
    z = np.random.randn(n)
    w = np.random.randn(n)
    x = 0.5 * z + 0.3 * w + np.random.randn(n)
    y = 1.0 + 2.0 * x + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w})
    return df


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

def test_adapter_from_iv2sls(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_reconstructs_training_data(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res)
    assert adapter.training_data is not None
    assert set(adapter.training_data.columns) >= {"y", "x", "z", "w"}


def test_adapter_from_ivgmm(iv_data):
    mod = IVGMM.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    assert adapter.coefficients().shape == (2,)


def test_adapter_from_ivliml(iv_data):
    mod = IVLIML.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    assert adapter.coefficients().shape == (2,)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def test_design_matrix_from_df(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    X = adapter.design_matrix_from_df(iv_data)
    assert X.shape == (len(iv_data), 2)
    assert jnp.allclose(X[:, 0], 1.0)  # Intercept
    assert jnp.allclose(X[:, 1], jnp.asarray(iv_data["x"].values))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_native(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    X = adapter.design_matrix_from_df(iv_data)
    beta = adapter.coefficients()
    preds = adapter.predict(beta, X)
    # linearmodels predict() returns fitted values
    native_preds = res.predict(data=iv_data)
    assert jnp.allclose(preds, jnp.asarray(native_preds.values).ravel(), atol=1e-4)


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------

def test_variable_metadata(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    meta = adapter.variable_metadata()
    assert "x" in meta
    assert "z" in meta
    assert "w" in meta


# ---------------------------------------------------------------------------
# Bootstrap refit
# ---------------------------------------------------------------------------

def test_refit(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    resampled = iv_data.sample(frac=1.0, replace=True)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LinearmodelsIVAdapter)
    assert new_adapter.coefficients().shape == adapter.coefficients().shape


# ---------------------------------------------------------------------------
# End-to-end via Margins
# ---------------------------------------------------------------------------

def test_margins_predict_iv2sls(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    m = Margins(model=None, adapter=adapter)
    pred = m.predict()
    assert pred.estimate is not None
    assert pred.std_error is not None


def test_margins_dydx_iv2sls(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_data)
    m = Margins(model=None, adapter=adapter)
    slope = m.dydx("x")
    assert slope.estimate is not None
    assert slope.std_error is not None


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_iv2sls(iv_data):
    from pymargins._adapters import _detect_adapter_class
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    cls = _detect_adapter_class(res)
    assert cls is LinearmodelsIVAdapter


def test_auto_detect_ivgmm(iv_data):
    from pymargins._adapters import _detect_adapter_class
    mod = IVGMM.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    cls = _detect_adapter_class(res)
    assert cls is LinearmodelsIVAdapter


# ---------------------------------------------------------------------------
# Formula interface (Track B)
# ---------------------------------------------------------------------------

@pytest.fixture
def iv_poly_data():
    rng = np.random.default_rng(42)
    n = 300
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    age = rng.normal(50, 10, size=n)
    x = 0.5 * z + 0.3 * w + rng.standard_normal(n)
    y = (
        1.0
        + 0.2 * age
        + 0.01 * (age ** 2)
        + 2.0 * x
        + rng.standard_normal(n)
    )
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w, "age": age})
    return df


def test_formula_verification_passes_iv2sls(iv_poly_data):
    """B.4 verification succeeds for a correctly specified formula."""
    from pymargins._formula import FormulaSpec
    mod = IV2SLS.from_formula("y ~ 1 + age + I(age**2) + [x ~ z + w]", data=iv_poly_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_poly_data)
    spec = FormulaSpec("y ~ age + I(age**2) + x", iv_poly_data)
    # Should not raise
    spec.verify_against(adapter)


def test_formula_verification_fails_on_wrong_formula_iv2sls(iv_poly_data):
    """B.4 verification fails when the formula omits a term."""
    from pymargins._formula import FormulaSpec
    mod = IV2SLS.from_formula("y ~ 1 + age + I(age**2) + [x ~ z + w]", data=iv_poly_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res, training_data=iv_poly_data)
    spec = FormulaSpec("y ~ age + x", iv_poly_data)  # missing I(age**2)
    with pytest.raises(ValueError, match="Formula verification failed"):
        spec.verify_against(adapter)


def test_dydx_with_formula_iv2sls(iv_poly_data):
    """B.6 Acceptance #2: IV with I(age**2) yields correct dydx via formula=."""
    mod = IV2SLS.from_formula("y ~ 1 + age + I(age**2) + [x ~ z + w]", data=iv_poly_data)
    res = mod.fit()
    # Use a larger fd_step to avoid float32 precision loss in the quadratic
    # finite-difference at age ~ 50 (default 1e-6 is too small for float32).
    m = Margins.linear_scale(
        res,
        formula="y ~ age + I(age**2) + x",
        data=iv_poly_data,
        at="mean",
        fd_step=1e-4,
    )
    slope = m.dydx("age")
    # Expected slope: beta_age + 2 * beta_age_sq * mean(age)
    expected = res.params["age"] + 2 * res.params["I(age ** 2)"] * iv_poly_data["age"].mean()
    assert np.isclose(float(slope.estimate), expected, rtol=1e-3)
    assert float(slope.std_error) > 0
