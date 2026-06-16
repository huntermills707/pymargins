"""Tests for alternative bootstrap CI methods."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation, steps


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    df["y"] = 0.5 + 0.8 * df["x1"] - 0.4 * df["x2"] + rng.standard_normal(n)
    return df


@pytest.fixture
def fit(df):
    return smf.ols("y ~ x1 + x2", data=df).fit()


# ---------------------------------------------------------------------------
# Percentile (default)
# ---------------------------------------------------------------------------


def test_percentile_default(fit):
    m = GComputation(fit, method="bootstrap", B=100, seed=42)
    res = m.dydx("x1")
    assert res.ci_method is None or res.ci_method == "percentile"
    assert res.estimate > res.conf_int_lower
    assert res.estimate < res.conf_int_upper


# ---------------------------------------------------------------------------
# Basic bootstrap
# ---------------------------------------------------------------------------


def test_basic_ci_differs_from_percentile(fit):
    m_basic = GComputation(
        fit,
        method="bootstrap",
        B=200,
        seed=42,
        ci="basic",
    )
    m_pct = GComputation(
        fit,
        method="bootstrap",
        B=200,
        seed=42,
        ci="percentile",
    )
    res_basic = m_basic.dydx("x1")
    res_pct = m_pct.dydx("x1")
    # Basic and percentile should generally differ
    assert not np.allclose(res_basic.conf_int_lower, res_pct.conf_int_lower)
    assert res_basic.ci_method == "basic"


def test_basic_ci_bounds_sensible(fit):
    m = GComputation(
        fit,
        method="bootstrap",
        B=100,
        seed=42,
        ci="basic",
    )
    res = m.dydx("x1")
    assert res.conf_int_lower < res.conf_int_upper
    assert np.isfinite(res.conf_int_lower)
    assert np.isfinite(res.conf_int_upper)


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


def test_bca_warns_without_acceleration():
    # Use a larger dataset so jackknife is skipped (n_obs > 200)
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": 0.5
            + 0.8 * rng.standard_normal(n)
            - 0.4 * rng.standard_normal(n)
            + rng.standard_normal(n),
        }
    )
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    with pytest.warns(UserWarning, match="BCa acceleration"):
        m = GComputation(
            fit,
            method="bootstrap",
            B=50,
            seed=42,
            ci="bca",
        )
        res = m.dydx("x1")
    assert res.ci_method == "bca"
    assert res.conf_int_lower < res.conf_int_upper


def test_bca_cluster_jackknife_auto(fit, df):
    # 10 clusters -> jackknife is automatic (n_clusters <= 200)
    clusters = df.index % 10
    inp = steps.input(df, cluster=clusters)
    m = GComputation(
        inp,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=42,
        ci="bca",
    )
    res = m.dydx("x1")
    assert res.ci_method == "bca"
    assert res.bootstrap_extras is not None
    # z0 always computed; a may be computed from jackknife
    assert "z0" in res.bootstrap_extras
    assert res.conf_int_lower < res.conf_int_upper


# ---------------------------------------------------------------------------
# Studentized bootstrap
# ---------------------------------------------------------------------------


def test_studentized_ci(fit):
    m = GComputation(
        fit,
        method="bootstrap",
        B=100,
        seed=42,
        ci="studentized",
    )
    res = m.dydx("x1")
    assert res.ci_method == "studentized"
    assert res.conf_int_lower < res.conf_int_upper
    assert np.isfinite(res.conf_int_lower)
    assert np.isfinite(res.conf_int_upper)
    assert res.bootstrap_extras is not None
    assert "t_star" in res.bootstrap_extras
    assert "se_hat" in res.bootstrap_extras


# ---------------------------------------------------------------------------
# Invalid ci_method
# ---------------------------------------------------------------------------


def test_invalid_ci_method_raises(fit):
    with pytest.raises(ValueError, match="ci="):
        GComputation(
            fit,
            method="bootstrap",
            B=10,
            ci="unknown",
        )


# ---------------------------------------------------------------------------
# Log-scale sessions with alternative CIs
# ---------------------------------------------------------------------------


def test_basic_ci_with_log_scale_raises(fit):
    with pytest.raises(ValueError, match="basic bootstrap"):
        m = GComputation(
            fit,
            scale="log",
            method="bootstrap",
            B=100,
            seed=42,
            ci="basic",
        )
        m.predict(atexog={"x1": 1.0})


def test_bca_ci_with_log_scale(fit):
    m = GComputation(
        fit,
        scale="log",
        method="bootstrap",
        B=100,
        seed=42,
        ci="bca",
    )
    res = m.predict(atexog={"x1": 1.0})
    assert res.ci_method == "bca"
    assert np.all(res.estimate > 0)
    assert np.all(res.conf_int_lower > 0)


def test_studentized_ci_with_log_scale_raises(fit):
    with pytest.raises(ValueError, match="studentized bootstrap"):
        m = GComputation(
            fit,
            scale="log",
            method="bootstrap",
            B=100,
            seed=42,
            ci="studentized",
        )
        m.predict(atexog={"x1": 1.0})
