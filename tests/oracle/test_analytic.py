"""Layer-1 analytic oracle tests.

Expected values are computed in-test from the fitted model with numpy/scipy
(no hardcoded floats, no pymargins call on the expected side). Standard
delta-method results: chain rule, e.g. Wooldridge (2010) §3.

These run against the current tree at R1 (baselining legacy) and against the
new engine from R6 on — same tests, no edits.

Design §4.1, req §7. Added in 0.4.0 (R1).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st

from pymargins import GComputation, Margins

from ._tolerances import TOL_ANALYTIC


def _ols_design(df):
    return np.column_stack([np.ones(len(df)), df["treat"], df["x1"], df["x2"]])


def test_ols_ame_equals_beta(fit_ols):
    """Linear model: dydx('x1') estimate == beta_x1, SE == sqrt(Sigma[j,j])."""
    m = GComputation(fit_ols, at="overall", method="delta")
    r = m.dydx("x1")
    assert np.asarray(r.estimate).shape == () or np.asarray(r.estimate).size == 1
    est = np.asarray(r.estimate).item()
    se = np.asarray(r.std_error).item()
    j = fit_ols.model.exog_names.index("x1")
    np.testing.assert_allclose(est, fit_ols.params.iloc[j], rtol=TOL_ANALYTIC)
    np.testing.assert_allclose(se, np.sqrt(fit_ols.cov_params().iloc[j, j]), rtol=TOL_ANALYTIC)


def test_ols_mean_prediction_equals_ybar(fit_ols):
    """predict() == mean(y); SE == sqrt(xbar' Sigma xbar)."""
    m = GComputation(fit_ols, at="overall", method="delta")
    r = m.predict()
    est = np.asarray(r.estimate).item()
    se = np.asarray(r.std_error).item()
    y = fit_ols.model.endog
    X = fit_ols.model.exog
    np.testing.assert_allclose(est, np.mean(y), rtol=TOL_ANALYTIC)
    xbar = np.mean(X, axis=0)
    np.testing.assert_allclose(se, np.sqrt(xbar @ fit_ols.cov_params().values @ xbar), rtol=TOL_ANALYTIC)


def test_logit_ame_closed_form(fit_logit):
    """AME_j = mean(p*(1-p)) * beta_j; delta SE via explicit gradient."""
    m = GComputation(fit_logit, at="overall", method="delta")
    r = m.dydx("x1")
    est = np.asarray(r.estimate).item()
    se = np.asarray(r.std_error).item()

    X = fit_logit.model.exog
    beta = fit_logit.params.values
    j = fit_logit.model.exog_names.index("x1")
    p = 1 / (1 + np.exp(-X @ beta))
    ame = np.mean(p * (1 - p)) * beta[j]
    np.testing.assert_allclose(est, ame, rtol=TOL_ANALYTIC)

    # gradient_k = mean(p(1-p) * (delta_jk + beta_j * (1-2p) * x_k))
    grad = np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        grad[k] = np.mean(
            p * (1 - p) * ((1.0 if k == j else 0.0) + beta[j] * (1 - 2 * p) * X[:, k])
        )
    expected_se = np.sqrt(grad @ fit_logit.cov_params().values @ grad)
    np.testing.assert_allclose(se, expected_se, rtol=TOL_ANALYTIC)


def test_poisson_ame_closed_form(fit_poisson):
    """AME_j = mean(exp(Xb)) * beta_j; delta SE via explicit gradient."""
    m = GComputation(fit_poisson, at="overall", method="delta")
    r = m.dydx("x1")
    est = np.asarray(r.estimate).item()
    se = np.asarray(r.std_error).item()

    X = fit_poisson.model.exog
    beta = fit_poisson.params.values
    j = fit_poisson.model.exog_names.index("x1")
    lam = np.exp(X @ beta)
    ame = np.mean(lam) * beta[j]
    np.testing.assert_allclose(est, ame, rtol=TOL_ANALYTIC)

    grad = np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        grad[k] = np.mean(lam * ((1.0 if k == j else 0.0) + beta[j] * X[:, k]))
    expected_se = np.sqrt(grad @ fit_poisson.cov_params().values @ grad)
    np.testing.assert_allclose(se, expected_se, rtol=TOL_ANALYTIC)


def test_logit_risk_difference_closed_form(fit_logit):
    """Counterfactual risk difference: mean(p1) - mean(p0)."""
    X = fit_logit.model.exog
    beta = fit_logit.params.values
    names = fit_logit.model.exog_names
    treat_idx = names.index("treat")

    X1 = X.copy()
    X0 = X.copy()
    X1[:, treat_idx] = 1.0
    X0[:, treat_idx] = 0.0
    p1 = 1 / (1 + np.exp(-X1 @ beta))
    p0 = 1 / (1 + np.exp(-X0 @ beta))
    rd = np.mean(p1) - np.mean(p0)

    m = GComputation(fit_logit, at="overall", method="delta")
    r = m.contrasts(
        scenarios=[{"atexog": {"treat": 1.0}}, {"atexog": {"treat": 0.0}}],
        contrasts=[1.0, -1.0],
    )
    est = np.asarray(r.estimate).item()
    np.testing.assert_allclose(est, rd, rtol=TOL_ANALYTIC)

    # gradient = mean(p1(1-p1) X1) - mean(p0(1-p0) X0)
    grad = np.mean((p1 * (1 - p1))[:, None] * X1, axis=0) - np.mean(
        (p0 * (1 - p0))[:, None] * X0, axis=0
    )
    expected_se = np.sqrt(grad @ fit_logit.cov_params().values @ grad)
    se = np.asarray(r.std_error).item()
    np.testing.assert_allclose(se, expected_se, rtol=TOL_ANALYTIC)


def test_weighted_logit_ame(fit_logit):
    """Weighted logit AME with normalized weights."""
    df = fit_logit.model.data.frame
    w = np.exp(np.random.default_rng(1).normal(0.0, 0.3, size=len(df)))
    w = w / w.sum()

    # TODO(R6): re-point at GComputation once weights= routing lands.
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    r = m.dydx("x1")
    est = np.asarray(r.estimate).item()

    X = fit_logit.model.exog
    beta = fit_logit.params.values
    j = fit_logit.model.exog_names.index("x1")
    p = 1 / (1 + np.exp(-X @ beta))
    ame = np.sum(w * p * (1 - p)) * beta[j]
    np.testing.assert_allclose(est, ame, rtol=TOL_ANALYTIC)


def test_hand_ols_micro():
    """n=4 hand example: full pipeline, zero library trust for expected values."""
    X = np.column_stack([np.ones(4), np.arange(4)])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    rss = np.sum((y - X @ beta) ** 2)
    sigma2 = rss / (len(y) - 2)
    Sigma = sigma2 * np.linalg.inv(X.T @ X)

    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"y": y, "x": X[:, 1]})
    fit = smf.ols("y ~ x", data=df).fit()
    m = GComputation(fit, at="overall", method="delta")

    pred = m.predict()
    np.testing.assert_allclose(np.asarray(pred.estimate).item(), beta[0] + beta[1] * np.mean(X[:, 1]), rtol=TOL_ANALYTIC)
    np.testing.assert_allclose(
        np.asarray(pred.std_error).item(),
        np.sqrt(np.array([1.0, np.mean(X[:, 1])]) @ Sigma @ np.array([1.0, np.mean(X[:, 1])])),
        rtol=TOL_ANALYTIC,
    )

    slope = m.dydx("x")
    np.testing.assert_allclose(np.asarray(slope.estimate).item(), beta[1], rtol=TOL_ANALYTIC)
    np.testing.assert_allclose(np.asarray(slope.std_error).item(), np.sqrt(Sigma[1, 1]), rtol=TOL_ANALYTIC)


def test_ci_convention_is_z(fit_logit):
    """Delta CI equals estimate +/- z_{level} * SE."""
    m = GComputation(fit_logit, at="overall", method="delta", level=0.95)
    r = m.dydx("x1")
    est = np.asarray(r.estimate).item()
    se = np.asarray(r.std_error).item()
    z = st.norm.ppf(0.975)
    np.testing.assert_allclose(np.asarray(r.conf_int_lower).item(), est - z * se, rtol=TOL_ANALYTIC)
    np.testing.assert_allclose(np.asarray(r.conf_int_upper).item(), est + z * se, rtol=TOL_ANALYTIC)
