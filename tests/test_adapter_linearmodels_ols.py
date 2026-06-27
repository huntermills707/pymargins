"""Tests for linearmodels OLSResults auto-detection (IV2SLS without endogenous vars)."""

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS

from pymargins import GComputation
from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter


@pytest.fixture
def ols_data():
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 1.0 + 2.0 * x + np.random.randn(n)
    return pd.DataFrame({"y": y, "x": x})


def test_auto_detect_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    assert type(res).__name__ == "OLSResults"
    est = GComputation(res)
    assert isinstance(est._compiled.adapter, LinearmodelsIVAdapter)


def test_gcomputation_predict_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    est = GComputation(res)

    pred = est.predict()
    native_pred = res.fitted_values
    assert np.isclose(
        float(pred.estimate), float(native_pred.mean().iloc[0]), rtol=1e-4
    )


def test_gcomputation_dydx_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    est = GComputation(res)

    slope = est.dydx("x")
    assert np.isclose(float(slope.estimate), float(res.params["x"]), rtol=2e-2)


def test_refit_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    adapter = LinearmodelsIVAdapter(res)

    resampled = ols_data.sample(n=len(ols_data), replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, LinearmodelsIVAdapter)
    assert len(new_adapter.coefficients()) == len(adapter.coefficients())


def test_bootstrap_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    est = GComputation(res, method="bootstrap", B=20, seed=42)
    pred = est.predict()
    assert np.isfinite(float(pred.estimate))
    assert np.isfinite(float(pred.std_error))


def test_custom_vcov_olsresults(ols_data):
    mod = IV2SLS.from_formula("y ~ 1 + x", data=ols_data)
    res = mod.fit()
    est = GComputation(res, vcov="robust")
    pred = est.predict()
    assert np.isfinite(float(pred.estimate))
    assert np.isfinite(float(pred.std_error))
