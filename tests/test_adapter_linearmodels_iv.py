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


def test_adapter_requires_training_data(iv_data):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_data)
    res = mod.fit()
    with pytest.raises(ValueError, match="training_data"):
        LinearmodelsIVAdapter(res)


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
