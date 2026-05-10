"""Tests for linearmodels AbsorbingLS adapter."""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from linearmodels.iv import AbsorbingLS

from pymargins import Margins
from pymargins._adapters.linearmodels_absorbing import LinearmodelsAbsorbingAdapter


@pytest.fixture
def absorb_data():
    np.random.seed(42)
    n = 200
    entities = np.random.randint(0, 20, n)
    times = np.random.randint(0, 5, n)
    df = pd.DataFrame({
        "y": np.random.randn(n) + 0.5 * entities + 0.3 * times,
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "entity": entities,
        "time": times,
    })
    return df


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

def test_adapter_from_absorbingls(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    assert adapter.coefficients().shape == (2,)
    assert adapter.covariance().shape == (2, 2)


def test_adapter_training_data_explicit(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res, training_data=absorb_data)
    assert adapter.coefficients().shape == (2,)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def test_design_matrix_from_df(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    X = adapter.design_matrix_from_df(absorb_data)
    assert X.shape == (len(absorb_data), 2)
    assert jnp.allclose(X[:, 0], jnp.asarray(absorb_data["x1"].values))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_manual(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    X = adapter.design_matrix_from_df(absorb_data)
    beta = adapter.coefficients()
    preds = adapter.predict(beta, X)
    # Manual X @ beta
    manual = jnp.asarray(X @ beta)
    assert jnp.allclose(preds, manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------

def test_variable_metadata(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta


# ---------------------------------------------------------------------------
# Bootstrap refit
# ---------------------------------------------------------------------------

def test_refit(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    resampled = absorb_data.sample(frac=1.0, replace=True).reset_index(drop=True)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LinearmodelsAbsorbingAdapter)
    assert new_adapter.coefficients().shape == adapter.coefficients().shape


# ---------------------------------------------------------------------------
# End-to-end via Margins
# ---------------------------------------------------------------------------

def test_margins_predict(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    m = Margins(model=None, adapter=adapter)
    pred = m.predict()
    assert pred.estimate is not None
    assert pred.std_error is not None


def test_margins_dydx(absorb_data):
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    adapter = LinearmodelsAbsorbingAdapter(res)
    m = Margins(model=None, adapter=adapter)
    slope = m.dydx("x1")
    assert slope.estimate is not None
    assert slope.std_error is not None


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect(absorb_data):
    from pymargins._adapters import _detect_adapter_class
    mod = AbsorbingLS(
        absorb_data[["y"]],
        absorb_data[["x1", "x2"]],
        absorb=absorb_data[["entity", "time"]],
    )
    res = mod.fit()
    cls = _detect_adapter_class(res)
    assert cls is LinearmodelsAbsorbingAdapter
