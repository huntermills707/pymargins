"""Regression tests for weighted estimands (Phase 0 of survey implementation)."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pytest

from pymargins import Margins


def test_dydx_applies_session_weights():
    """Weighted dydx must differ from unweighted and match hand computation."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * df.x)))).astype(int)
    w = rng.uniform(0.3, 3.0, n)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    # Hand-compute weighted average marginal effect
    b0, b1 = np.asarray(fit.params)
    p = 1 / (1 + np.exp(-(b0 + b1 * df.x.values)))
    me = b1 * p * (1 - p)
    expected = float(np.average(me, weights=w))

    got = float(np.asarray(Margins(fit, weights=w).dydx("x").estimate).ravel()[0])
    unw = float(np.asarray(Margins(fit).dydx("x").estimate).ravel()[0])

    assert np.isclose(got, expected, rtol=1e-4), (got, expected)
    assert not np.isclose(got, unw, rtol=1e-4)


def test_predict_still_weighted():
    """predict() must continue to respect session weights."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * df.x)))).astype(int)
    w = rng.uniform(0.3, 3.0, n)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    unw = float(np.asarray(Margins(fit).predict().estimate).ravel()[0])
    wt = float(np.asarray(Margins(fit, weights=w).predict().estimate).ravel()[0])

    assert not np.isclose(unw, wt, rtol=1e-4)


def test_nan_weight_raises_at_construction():
    """Non-finite weights must raise at Margins construction, not mid-trace."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 0.5).astype(int)
    w = np.append(rng.uniform(0.5, 2.0, n - 1), np.nan)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    with pytest.raises(ValueError, match="finite"):
        Margins(fit, weights=w)


def test_zero_sum_weights_raises_at_construction():
    """Weights summing to zero must raise at construction."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 0.5).astype(int)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    with pytest.raises(ValueError, match="sum to zero"):
        Margins(fit, weights=np.zeros(n))


def test_negative_weight_raises_at_construction():
    """Negative weights must raise at construction."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 0.5).astype(int)
    w = rng.uniform(0.5, 2.0, n)
    w[0] = -1.0
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    with pytest.raises(ValueError, match="non-negative"):
        Margins(fit, weights=w)
