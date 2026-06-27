"""Correctness tests for GEE and MixedLM adapters.

These tests verify that pymargins produces mathematically correct point
estimates by comparing against analytical ground truth and against the
behavior of equivalent GLM/OLS models.

Reference behavior (R marginaleffects)
--------------------------------------
- ``geepack::geeglm`` → GEE marginal effects on the same scale as GLM,
  using the robust sandwich covariance by default.
- ``lme4::lmer`` / ``nlme::lme`` → mixed effects marginal effects on the
  **population-average** scale by default (``re.form = NA`` for conditional,
  ``re.form = NULL`` for marginal).  For linear mixed models (identity link),
  PA and conditional coincide and equal ``X %*% fixef``.

Ground-truth checks implemented
-------------------------------
1. GEE (independent) vs GLM:  same mean structure => same AMEs.
2. Linear MixedLM:  ``dydx()`` for a continuous variable equals the fixed
   effect coefficient (for identity link, slope = coefficient).
3. MixedLM PA predictions:  ``predict()`` equals ``X @ fe_params``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation
from pymargins._adapters.statsmodels_gee import StatsmodelsGEEAdapter
from pymargins._adapters.statsmodels_mixedlm import StatsmodelsMixedLMAdapter

# ---------------------------------------------------------------------------
# 1. GEE vs GLM correctness
# ---------------------------------------------------------------------------


@pytest.fixture
def df_gee_glm():
    """Synthetic clustered data."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
            "group": np.repeat(np.arange(20), 10),
        }
    )
    # Binary outcome
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.8 * df["treatment"]
    df["y_bin"] = (rng.uniform(size=n) < (1 / (1 + np.exp(-eta)))).astype(float)
    # Count outcome
    df["y_count"] = rng.poisson(np.exp(eta))
    return df


def test_gee_logit_ame_matches_glm(df_gee_glm):
    """GEE with independence working correlation has same mean structure as GLM.

    The AME (average marginal effect) for a continuous covariate should be
    identical between GEE and GLM to numerical precision.
    """
    df = df_gee_glm
    fit_glm = smf.glm(
        "y_bin ~ x1 + x2 + treatment",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    fit_gee = smf.gee(
        "y_bin ~ x1 + x2 + treatment",
        groups="group",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()

    m_glm = GComputation(fit_glm)
    m_gee = GComputation(fit_gee)

    ame_glm = m_glm.dydx("x1")
    ame_gee = m_gee.dydx("x1")

    np.testing.assert_allclose(
        np.asarray(ame_glm.estimate),
        np.asarray(ame_gee.estimate),
        rtol=1e-10,
        err_msg="GEE logit AME should match GLM logit AME",
    )


def test_gee_poisson_ame_matches_glm(df_gee_glm):
    """GEE Poisson with independence working correlation => same AMEs as GLM Poisson."""
    df = df_gee_glm
    fit_glm = smf.glm(
        "y_count ~ x1 + x2 + treatment",
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    fit_gee = smf.gee(
        "y_count ~ x1 + x2 + treatment",
        groups="group",
        data=df,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()

    m_glm = GComputation(fit_glm)
    m_gee = GComputation(fit_gee)

    ame_glm = m_glm.dydx("x1")
    ame_gee = m_gee.dydx("x1")

    np.testing.assert_allclose(
        np.asarray(ame_glm.estimate),
        np.asarray(ame_gee.estimate),
        rtol=1e-10,
        err_msg="GEE Poisson AME should match GLM Poisson AME",
    )


def test_gee_logit_contrast_matches_glm(df_gee_glm):
    """Contrast (risk difference) for a binary covariate should match between GEE and GLM."""
    df = df_gee_glm
    fit_glm = smf.glm(
        "y_bin ~ x1 + x2 + treatment",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    fit_gee = smf.gee(
        "y_bin ~ x1 + x2 + treatment",
        groups="group",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()

    m_glm = GComputation(fit_glm)
    m_gee = GComputation(fit_gee)

    c_glm = m_glm.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    c_gee = m_gee.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )

    np.testing.assert_allclose(
        np.asarray(c_glm.estimate),
        np.asarray(c_gee.estimate),
        rtol=1e-10,
        err_msg="GEE logit contrast should match GLM logit contrast",
    )


# ---------------------------------------------------------------------------
# 2. MixedLM correctness
# ---------------------------------------------------------------------------


@pytest.fixture
def df_mixedlm():
    """Synthetic data with clusters for linear mixed models."""
    rng = np.random.default_rng(43)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
            "group": np.repeat(np.arange(20), 10),
        }
    )
    # Random intercept per group
    group_effect = rng.standard_normal(20)[df["group"].values]
    df["y"] = (
        1.0
        + 0.5 * df["x1"]
        - 0.3 * df["x2"]
        + 0.8 * df["treatment"]
        + group_effect
        + rng.standard_normal(n) * 0.5
    )
    return df


def test_mixedlm_dydx_equals_coefficient(df_mixedlm):
    """For a linear mixed model (identity link), the slope dydx equals the FE coeff."""
    df = df_mixedlm
    fit = smf.mixedlm("y ~ x1 + x2 + treatment", groups="group", data=df).fit()

    m = GComputation(fit)
    ame_x1 = m.dydx("x1")
    ame_x2 = m.dydx("x2")

    # For identity link, the AME for a continuous variable is exactly the
    # fixed-effect coefficient because d/dx (X @ beta) = beta_j.
    np.testing.assert_allclose(
        np.asarray(ame_x1.estimate),
        fit.fe_params["x1"],
        rtol=1e-10,
        err_msg="MixedLM dydx(x1) should equal fe_params['x1']",
    )
    np.testing.assert_allclose(
        np.asarray(ame_x2.estimate),
        fit.fe_params["x2"],
        rtol=1e-10,
        err_msg="MixedLM dydx(x2) should equal fe_params['x2']",
    )


def test_mixedlm_prediction_equals_manual_xbeta(df_mixedlm):
    """PA prediction should equal X @ fe_params for linear mixed models."""
    df = df_mixedlm
    fit = smf.mixedlm("y ~ x1 + x2 + treatment", groups="group", data=df).fit()
    adapter = StatsmodelsMixedLMAdapter(fit)

    X = adapter.design_matrix_from_df(df.iloc[:5])
    beta = adapter.coefficients()
    mu = adapter.predict(beta, X)

    expected = np.asarray(X) @ np.asarray(beta)
    np.testing.assert_allclose(np.asarray(mu), expected, rtol=1e-10)


def test_mixedlm_contrast_equals_coefficient(df_mixedlm):
    """For identity link, the contrast for a binary variable equals its coefficient."""
    df = df_mixedlm
    fit = smf.mixedlm("y ~ x1 + x2 + treatment", groups="group", data=df).fit()

    m = GComputation(fit)
    c = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )

    np.testing.assert_allclose(
        np.asarray(c.estimate),
        fit.fe_params["treatment"],
        rtol=1e-10,
        err_msg="MixedLM contrast(treatment) should equal fe_params['treatment']",
    )


# ---------------------------------------------------------------------------
# 3. Adapter-specific ground-truth tests
# ---------------------------------------------------------------------------


def test_gee_adapter_predict_matches_native(df_gee_glm):
    """StatsmodelsGEEAdapter.predict() should match statsmodels GEE.predict()."""
    df = df_gee_glm
    fit = smf.gee(
        "y_bin ~ x1 + x2 + treatment",
        groups="group",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    adapter = StatsmodelsGEEAdapter(fit)

    X = adapter.design_matrix_from_df(df.iloc[:5])
    beta = adapter.coefficients()
    mu_adapter = adapter.predict(beta, X)
    mu_native = fit.predict(df.iloc[:5])

    np.testing.assert_allclose(
        np.asarray(mu_adapter),
        mu_native.values,
        rtol=1e-10,
    )


def test_mixedlm_adapter_predict_matches_native_pa(df_mixedlm):
    """StatsmodelsMixedLMAdapter.predict() should match statsmodels predict on PA scale."""
    df = df_mixedlm
    fit = smf.mixedlm("y ~ x1 + x2 + treatment", groups="group", data=df).fit()
    adapter = StatsmodelsMixedLMAdapter(fit)

    X = adapter.design_matrix_from_df(df.iloc[:5])
    beta = adapter.coefficients()
    mu_adapter = adapter.predict(beta, X)
    mu_native = fit.predict(df.iloc[:5])

    np.testing.assert_allclose(
        np.asarray(mu_adapter),
        mu_native.values,
        rtol=1e-10,
    )
