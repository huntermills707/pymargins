"""Tests for design-based survey linearization covariance."""

import jax

jax.config.update("jax_enable_x64", True)

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, SurveyDesign


def test_survey_se_differs_from_default():
    """Survey design-based SEs must differ from the default (non-robust) SE."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "psu": rng.integers(0, 30, n),
        "strat": rng.integers(0, 3, n),
    })
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    df["w"] = rng.uniform(0.5, 2.0, n)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    d = SurveyDesign(weights=df.w.values, psu=df.psu.values, strata=df.strat.values)

    m = Margins(fit, survey_design=d, weights=df.w.values)
    r_survey = m.dydx("x")

    m_default = Margins(fit)
    r_default = m_default.dydx("x")

    assert not np.isclose(r_survey.std_error, r_default.std_error, rtol=1e-4)


def test_survey_self_consistency_with_cluster_cov():
    """Survey sandwich with equal weights and no strata must equal cluster
    covariance up to the statsmodels-specific (N-1)/(N-K) factor.
    """
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * df.x)))).astype(int)
    clusters = rng.integers(0, 10, n)

    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    d_simple = SurveyDesign(weights=np.ones(n), psu=clusters)
    m_simple = Margins(fit, survey_design=d_simple)

    fit_cluster = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": clusters}
    )

    survey_cov = np.asarray(m_simple._frozen_cov())
    cluster_cov = np.asarray(fit_cluster.cov_params())

    # The survey sandwich and statsmodels' cluster cov both carry the
    # G/(G-1) factor (one stratum of G PSUs), so it cancels; only the
    # statsmodels-specific (N-1)/(N-K) finite-sample factor remains.
    N = n
    K = 2  # intercept + x
    extra_factor = (N - 1) / (N - K)

    assert np.allclose(survey_cov * extra_factor, cluster_cov, rtol=1e-3)


def test_survey_ols_adapter():
    """OLS adapter must also produce design-based SEs."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "psu": rng.integers(0, 20, n),
        "strat": rng.integers(0, 3, n),
    })
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(size=n)
    df["w"] = rng.uniform(0.5, 2.0, n)
    fit = smf.ols("y ~ x", df).fit()
    d = SurveyDesign(weights=df.w.values, psu=df.psu.values, strata=df.strat.values)

    m = Margins(fit, survey_design=d, weights=df.w.values)
    r = m.dydx("x")

    m_default = Margins(fit)
    r_default = m_default.dydx("x")

    assert not np.isclose(r.std_error, r_default.std_error, rtol=1e-4)


def test_lonely_psu_raises():
    """A stratum with only one PSU must raise a clear error."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 0.5).astype(int)
    df["psu"] = np.arange(n)
    df["strat"] = np.zeros(n, dtype=int)
    df.loc[0, "strat"] = 1  # one stratum with only 1 PSU
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    d = SurveyDesign(weights=np.ones(n), psu=df.psu.values, strata=df.strat.values)

    with pytest.raises(ValueError, match="lonely PSU"):
        Margins(fit, survey_design=d).dydx("x")


def test_weighted_fit_avoids_double_counting():
    """When the model is fit with freq_weights, survey SE must not
    double-count — the adapter should detect fitting weights and pass
    unit weights to the linearization kernel."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "psu": rng.integers(0, 30, n),
        "strat": rng.integers(0, 3, n),
    })
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    df["w"] = rng.uniform(0.5, 2.0, n)

    # Fit WITH weights
    fit_w = smf.glm("y ~ x", df, family=sm.families.Binomial(),
                    freq_weights=df.w.values).fit()
    d = SurveyDesign(weights=df.w.values, psu=df.psu.values, strata=df.strat.values)

    # Fit weights == design weights → proportional → must NOT warn.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m_w = Margins(fit_w, survey_design=d, weights=df.w.values)
        se_weighted_fit = float(m_w.dydx("x").std_error)
    assert not any("not proportional" in str(x.message) for x in rec)

    fit_uw = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    m_uw = Margins(fit_uw, survey_design=d, weights=df.w.values)
    se_unweighted_fit = float(m_uw.dydx("x").std_error)

    assert se_weighted_fit > 0
    assert se_unweighted_fit > 0
    # The weighted-fit path uses the exact weighted bread, so it agrees
    # closely with the unweighted-fit path (≈1% on this fixture). A loose
    # tolerance would not catch a w̄- or w̄²-scale double-count.
    assert np.isclose(se_weighted_fit, se_unweighted_fit, rtol=0.05)


def test_weighted_fit_mismatched_weights_warns():
    """If the model's fit weights are not proportional to the survey design
    weights, the adapter must warn: the design-based variance uses the fit
    weights while the point estimate uses the design weights."""
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "psu": rng.integers(0, 30, n),
        "strat": rng.integers(0, 3, n),
    })
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    w_fit = rng.uniform(0.5, 2.0, n)
    w_design = rng.uniform(0.5, 2.0, n)  # independent draw → not proportional

    fit = smf.glm("y ~ x", df, family=sm.families.Binomial(),
                  freq_weights=w_fit).fit()
    d = SurveyDesign(weights=w_design, psu=df.psu.values, strata=df.strat.values)
    with pytest.warns(UserWarning, match="not proportional"):
        Margins(fit, survey_design=d, weights=w_design).dydx("x")


def test_fpc_fraction():
    """FPC supplied as a fraction must reduce variance relative to no FPC."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "psu": rng.integers(0, 30, n),
        "strat": rng.integers(0, 3, n),
    })
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    d_no_fpc = SurveyDesign(weights=np.ones(n), psu=df.psu.values, strata=df.strat.values)
    se_no_fpc = Margins(fit, survey_design=d_no_fpc).dydx("x").std_error

    # Small FPC fraction (0.1) → modest variance reduction
    fpc = np.full(n, 0.1)
    d_fpc = SurveyDesign(
        weights=np.ones(n), psu=df.psu.values, strata=df.strat.values,
        fpc=fpc, fpc_is_fraction=True
    )
    se_fpc = Margins(fit, survey_design=d_fpc).dydx("x").std_error

    assert se_fpc < se_no_fpc
