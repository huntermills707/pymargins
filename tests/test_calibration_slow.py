"""Layer-5 calibration lane — weekly slow-marked statistical validation.

Anchored to the rewrite plan rev.2 §4 layer 5 ("sim/boot SE vs delta SE
agreement within Monte Carlo error on smooth cases; the req §9 coverage
simulations") and req §9 ("coverage simulations for each new estimator").
These are genuinely statistical checks: a failure is a *method* bug, not float
noise, so they live in the slow lane and are excluded from the default gate.

Methodology (textbook; no package-specific recipe is invented):

- **Coverage.** Draw ``R`` independent samples from a correctly-specified DGP.
  For each replicate the target is the estimand evaluated at the *true*
  coefficients on that replicate's design (the analytic AME identity from
  ``test_analytic.py`` — closed form, never a pymargins call), so the check
  isolates the β̂-estimation uncertainty the CI actually models. Empirical
  coverage of that target must sit within Monte Carlo error of the nominal
  level.
- **Calibration.** On a smooth (low-curvature) case where the delta method is
  trustworthy, the simulation and bootstrap SEs must agree with the analytic
  delta SE within Monte Carlo error.

Everything is seeded, so the empirical numbers are deterministic and the bands
below are not flaky; they are wide enough to pass a correct engine and narrow
enough to fail a gross miscalibration (e.g. a dropped vcov, ~0.80 coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation

LEVEL = 0.95


def _ols_dataset(rng, n, *, b0=1.0, b1=2.0, b2=-1.0, sigma=1.0):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = b0 + b1 * x1 + b2 * x2 + rng.normal(scale=sigma, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _logit_dataset(rng, n, *, b0=-0.3, b1=0.8, b2=-0.4):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _logit_true_ame(df, *, b0, b1, b2):
    """Analytic AME of x1 at the true coefficients on this design: the
    closed-form identity mean(p(1-p))*b1 (Wooldridge 2010 §3; layer-1)."""
    eta = b0 + b1 * df["x1"].to_numpy() + b2 * df["x2"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-eta))
    return float(np.mean(p * (1.0 - p)) * b1)


def _ci(result):
    lo = float(np.ravel(result.conf_int_lower)[0])
    hi = float(np.ravel(result.conf_int_upper)[0])
    return lo, hi


def _se(result):
    return float(np.ravel(result.std_error)[0])


# ---------------------------------------------------------------------------
# Coverage simulations (req §9)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_coverage_ols_ame_delta():
    """OLS dydx delta CIs cover the true slope at the nominal rate.

    The AME of a linear model is exactly ``b1`` on every design, so the target
    is known without approximation.
    """
    b1 = 2.0
    n_rep, n = 500, 300
    rng = np.random.default_rng(20260622)
    hits = 0
    for _ in range(n_rep):
        df = _ols_dataset(rng, n, b1=b1)
        fit = smf.ols("y ~ x1 + x2", data=df).fit()
        lo, hi = _ci(GComputation(fit, method="delta", level=LEVEL).dydx("x1"))
        hits += lo <= b1 <= hi
    cov = hits / n_rep
    assert 0.92 <= cov <= 0.98, f"OLS AME delta coverage {cov:.3f} ≠ nominal {LEVEL}"


@pytest.mark.slow
def test_coverage_ols_ame_simulation():
    """OLS dydx simulation (Krinsky–Robb) CIs cover the true slope nominally."""
    b1 = 2.0
    n_rep, n = 300, 300
    rng = np.random.default_rng(7)
    hits = 0
    for i in range(n_rep):
        df = _ols_dataset(rng, n, b1=b1)
        fit = smf.ols("y ~ x1 + x2", data=df).fit()
        est = GComputation(fit, method="simulation", n_sim=2000, seed=i, level=LEVEL)
        lo, hi = _ci(est.dydx("x1"))
        hits += lo <= b1 <= hi
    cov = hits / n_rep
    assert 0.91 <= cov <= 0.99, f"OLS AME sim coverage {cov:.3f} ≠ nominal {LEVEL}"


@pytest.mark.slow
def test_coverage_logit_ame_delta():
    """Logit dydx delta CIs cover the true (analytic) AME at the nominal rate.

    The per-replicate target is mean(p(1-p))*b1 at the *true* coefficients on
    that replicate's design — the closed-form identity, computed in numpy.
    """
    b0, b1, b2 = -0.3, 0.8, -0.4
    n_rep, n = 350, 1500
    rng = np.random.default_rng(99)
    hits = 0
    for _ in range(n_rep):
        df = _logit_dataset(rng, n, b0=b0, b1=b1, b2=b2)
        truth = _logit_true_ame(df, b0=b0, b1=b1, b2=b2)
        fit = smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()
        est = GComputation(fit, at="overall", scale="response", method="delta", level=LEVEL)
        lo, hi = _ci(est.dydx("x1"))
        hits += lo <= truth <= hi
    cov = hits / n_rep
    assert 0.92 <= cov <= 0.98, f"logit AME delta coverage {cov:.3f} ≠ nominal {LEVEL}"


# ---------------------------------------------------------------------------
# Calibration: sim/boot SE vs delta SE on smooth cases (plan §4 layer 5)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calibration_se_agreement_ols():
    """On a smooth OLS case, simulation and bootstrap SEs agree with the
    analytic (HC1) delta SE within Monte Carlo error. The case bootstrap and
    the HC1 sandwich both target the robust variance, so all three align."""
    rng = np.random.default_rng(2024)
    df = _ols_dataset(rng, n=2000)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    se_delta = _se(GComputation(fit, method="delta", vcov="HC1").dydx("x1"))
    se_sim = _se(
        GComputation(fit, method="simulation", vcov="HC1", n_sim=8000, seed=11).dydx("x1")
    )
    se_boot = _se(GComputation(fit, method="bootstrap", B=3000, seed=11).dydx("x1"))
    assert abs(se_sim - se_delta) / se_delta < 0.05, (se_sim, se_delta)
    assert abs(se_boot - se_delta) / se_delta < 0.10, (se_boot, se_delta)


@pytest.mark.slow
def test_calibration_se_agreement_logit():
    """On a smooth logit AME (low curvature), the simulation SE agrees with the
    delta SE within Monte Carlo error — delta is trustworthy here, so the
    Krinsky–Robb draws must reproduce it."""
    rng = np.random.default_rng(456)
    df = _logit_dataset(rng, n=3000)
    fit = smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()
    se_delta = _se(
        GComputation(fit, at="overall", scale="response", method="delta").dydx("x1")
    )
    se_sim = _se(
        GComputation(
            fit, at="overall", scale="response", method="simulation", n_sim=8000, seed=5
        ).dydx("x1")
    )
    assert abs(se_sim - se_delta) / se_delta < 0.05, (se_sim, se_delta)
