"""Tests for statsmodels zero-inflated count model adapters."""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP,
)

from pymargins import Margins
from pymargins._adapters.statsmodels_zi import StatsmodelsZIAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zi_data():
    """Generate zero-inflated count data."""
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    z = np.random.randn(n)
    lam = np.exp(0.5 + 0.3 * x)
    pi = 1 / (1 + np.exp(-(-0.5 + 0.2 * z)))
    y = np.random.poisson(lam)
    y[np.random.rand(n) < pi] = 0
    return pd.DataFrame({"y": y, "x": x, "z": z})


@pytest.fixture
def zinb_data():
    """Generate zero-inflated negative binomial data."""
    np.random.seed(42)
    n = 500
    x = np.random.randn(n)
    z = np.random.randn(n)
    mu = np.exp(0.5 + 0.3 * x)
    pi = 1 / (1 + np.exp(-(-0.5 + 0.2 * z)))
    alpha = 1.0
    # NB2 parameterization: variance = mu + alpha * mu**2
    size = 1.0 / alpha
    prob = size / (size + mu)
    y = np.random.negative_binomial(size, prob, size=n)
    y[np.random.rand(n) < pi] = 0
    return pd.DataFrame({"y": y, "x": x, "z": z})


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_zip(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    m = Margins(res)
    assert isinstance(m.adapter, StatsmodelsZIAdapter)


def test_auto_detect_zinb(zinb_data):
    mod = ZeroInflatedNegativeBinomialP.from_formula(
        "y ~ x", exog_infl=zinb_data[["z"]], data=zinb_data,
    )
    res = mod.fit(disp=False)
    m = Margins(res)
    assert isinstance(m.adapter, StatsmodelsZIAdapter)


# ---------------------------------------------------------------------------
# Coefficients & covariance
# ---------------------------------------------------------------------------

def test_coefficients_shape(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)
    coef = adapter.coefficients()
    assert coef.shape == (len(res.params),)
    assert coef.dtype == jnp.float32 or coef.dtype == jnp.float64


def test_covariance_default(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)
    cov = adapter.covariance()
    assert cov.shape == (len(res.params), len(res.params))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_predict_matches_native_zip(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    X = adapter.design_matrix_from_df(zi_data)
    beta = adapter.coefficients()
    pred = adapter.predict(beta, X)
    native_pred = res.predict()

    assert np.allclose(np.asarray(pred), native_pred, rtol=1e-4)


def test_predict_matches_native_zinb(zinb_data):
    mod = ZeroInflatedNegativeBinomialP.from_formula(
        "y ~ x", exog_infl=zinb_data[["z"]], data=zinb_data,
    )
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    X = adapter.design_matrix_from_df(zinb_data)
    beta = adapter.coefficients()
    pred = adapter.predict(beta, X)
    native_pred = res.predict()

    assert np.allclose(np.asarray(pred), native_pred, rtol=1e-4)


# ---------------------------------------------------------------------------
# JAX differentiability
# ---------------------------------------------------------------------------

def test_jax_differentiability(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    X = adapter.design_matrix_from_df(zi_data)
    beta = adapter.coefficients()

    import jax
    grad = jax.grad(lambda b: jnp.mean(adapter.predict(b, X)))(beta)
    assert grad.shape == beta.shape
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------

def test_margins_predict_zip(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    m = Margins(res)

    pred = m.predict()
    native_pred = res.predict()
    assert np.isclose(float(pred.estimate), float(native_pred.mean()), rtol=1e-3)


def test_margins_dydx_zip(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    m = Margins(res)

    # dydx on count variable x
    slope = m.dydx("x")
    assert np.isfinite(float(slope.estimate))
    assert np.isfinite(float(slope.std_error))

    # dydx on inflation variable z
    slope_z = m.dydx("z")
    assert np.isfinite(float(slope_z.estimate))
    assert np.isfinite(float(slope_z.std_error))


# ---------------------------------------------------------------------------
# Bootstrap refit
# ---------------------------------------------------------------------------

def test_bootstrap_refit_zip(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    resampled = zi_data.sample(n=len(zi_data), replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsZIAdapter)
    assert len(new_adapter.coefficients()) == len(adapter.coefficients())


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------

def test_variable_metadata(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    meta = adapter.variable_metadata()
    assert "x" in meta
    assert "z" in meta
    assert meta["x"].var_type == "continuous"
    assert meta["z"].var_type == "continuous"


# ---------------------------------------------------------------------------
# Column index lookup
# ---------------------------------------------------------------------------

def test_column_index_of_variable(zi_data):
    mod = ZeroInflatedPoisson.from_formula("y ~ x", exog_infl=zi_data[["z"]], data=zi_data)
    res = mod.fit(disp=False)
    adapter = StatsmodelsZIAdapter(res)

    # In the concatenated design matrix [infl | count], z is at index 0, x is at index 2
    assert adapter.column_index_of_variable("z") == 0
    assert adapter.column_index_of_variable("x") == 2
