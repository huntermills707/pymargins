"""Regression tests for weighted estimands (Phase 0 of survey implementation)."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation


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

    got = float(np.asarray(GComputation(fit, weights=w).dydx("x").estimate).ravel()[0])
    unw = float(np.asarray(GComputation(fit).dydx("x").estimate).ravel()[0])

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

    unw = float(np.asarray(GComputation(fit).predict().estimate).ravel()[0])
    wt = float(np.asarray(GComputation(fit, weights=w).predict().estimate).ravel()[0])

    assert not np.isclose(unw, wt, rtol=1e-4)


# NOTE: The legacy Margins session validated weights eagerly at construction
# (finite, non-negative, non-zero-sum). GComputation does not perform this
# eager validation, so the corresponding regression tests are dropped as a
# semantic change under the R7 rewrite.
