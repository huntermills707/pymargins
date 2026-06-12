"""Anchor harness: GComputation vs Margins byte-identical reproduction (W2.7).

This test suite is the correctness gate for Phase 2.  Every merge from
Phase 2 on must keep this suite green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins
from pymargins.estimators import GComputation


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treat": rng.binomial(1, 0.5, size=n),
        }
    )


@pytest.fixture
def fit_glm(df):
    return smf.glm("y ~ treat + x1 + x2", data=df, family=sm.families.Binomial()).fit()


@pytest.fixture
def fit_ols(df):
    return smf.ols("y ~ treat + x1 + x2", data=df).fit()


# ---------------------------------------------------------------------------
# Anchor tests: GComputation(model) vs Margins(model)
# ---------------------------------------------------------------------------


class TestAnchorGLM:
    @pytest.mark.parametrize("method", ["delta", "simulation"])
    def test_predict_matches(self, df, fit_glm, method):
        seed = 12345
        m = Margins(fit_glm, at="overall", method=method, rng_seed=seed)
        est = GComputation(fit_glm, at="overall", method=method, seed=seed)

        r1 = m.predict()
        r2 = est.predict()

        np.testing.assert_array_equal(r1.estimate, r2.estimate)
        np.testing.assert_array_equal(r1.std_error, r2.std_error)
        np.testing.assert_array_equal(r1.conf_int_lower, r2.conf_int_lower)
        np.testing.assert_array_equal(r1.conf_int_upper, r2.conf_int_upper)

        if method == "simulation":
            np.testing.assert_array_equal(r1.draws, r2._result.draws)

    @pytest.mark.parametrize("method", ["delta", "simulation"])
    def test_dydx_matches(self, df, fit_glm, method):
        seed = 12345
        m = Margins(fit_glm, at="overall", method=method, rng_seed=seed)
        est = GComputation(fit_glm, at="overall", method=method, seed=seed)

        r1 = m.dydx("x1")
        r2 = est.dydx("x1")

        np.testing.assert_array_equal(r1.estimate, r2.estimate)
        np.testing.assert_array_equal(r1.std_error, r2.std_error)

    @pytest.mark.parametrize("method", ["delta", "simulation"])
    def test_contrasts_matches(self, df, fit_glm, method):
        seed = 12345
        m = Margins(fit_glm, at="overall", method=method, rng_seed=seed)
        est = GComputation(fit_glm, at="overall", method=method, seed=seed)

        r1 = m.contrasts(
            scenarios=[
                {"atexog": {"treat": 1}},
                {"atexog": {"treat": 0}},
            ],
            contrasts=[1, -1],
        )
        r2 = est.contrasts(
            scenarios=[
                {"atexog": {"treat": 1}},
                {"atexog": {"treat": 0}},
            ],
            contrasts=[1, -1],
        )

        np.testing.assert_array_equal(r1.estimate, r2.estimate)
        np.testing.assert_array_equal(r1.std_error, r2.std_error)


class TestAnchorOLS:
    @pytest.mark.parametrize("method", ["delta", "simulation", "bootstrap"])
    def test_predict_matches(self, df, fit_ols, method):
        seed = 12345
        kwargs = {"at": "overall", "method": method}
        if method == "bootstrap":
            kwargs["n_boot"] = 200
        m = Margins(fit_ols, rng_seed=seed, **kwargs)
        est = GComputation(fit_ols, seed=seed, **kwargs)
        if method == "bootstrap":
            est_kwargs = {"B": 200}
        else:
            est_kwargs = {}
        est = GComputation(fit_ols, seed=seed, **kwargs, **est_kwargs)

        r1 = m.predict()
        r2 = est.predict()

        np.testing.assert_array_equal(r1.estimate, r2.estimate)
        np.testing.assert_array_equal(r1.std_error, r2.std_error)
        np.testing.assert_array_equal(r1.conf_int_lower, r2.conf_int_lower)
        np.testing.assert_array_equal(r1.conf_int_upper, r2.conf_int_upper)

        if method == "simulation":
            np.testing.assert_array_equal(r1.draws, r2._result.draws)
        if method == "bootstrap":
            np.testing.assert_array_equal(r1.draws, r2._result.draws)
