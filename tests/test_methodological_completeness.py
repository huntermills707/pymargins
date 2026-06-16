"""Tests for METHODOLOGICAL_COMPLETENESS_PLAN gaps G5–G7b.

Ported to the v0.4.0 GComputation noun.  Tests that depended on legacy
MarginsResult mechanics (G1–G4 composition, kappa propagation through
result-level arithmetic, and the compose_results helper) are dropped:
those APIs were removed with the Margins session.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation


@pytest.fixture
def df_logit():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )
    lp = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"] + 0.3 * (df["sex"] == "M")
    df["outcome"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "outcome ~ age + treatment + C(sex)",
        data=df_logit,
        family=sm.families.Binomial(),
    ).fit()


# ---------------------------------------------------------------------------
# G5 — Empirical quadratic-form joint test
# ---------------------------------------------------------------------------


def test_joint_test_empirical_kind_exists(fit_logit):
    """kind='empirical' is accepted and returns a TestResult."""
    est = GComputation(fit_logit, method="simulation", n_sim=200, seed=42)
    pred = est.predict(atexog={"treatment": [0, 1]})
    result = pred.joint_test(kind="empirical")
    assert result.method == "joint_empirical"
    assert np.isfinite(float(result.statistic))
    assert 0 <= float(result.pvalue) <= 1


def test_joint_test_empirical_vs_wald_on_gaussian_draws(fit_logit):
    """Under approximately Gaussian draws, empirical and wald p-values agree."""
    est = GComputation(fit_logit, method="simulation", n_sim=500, seed=42)
    pred = est.predict(atexog={"treatment": [0, 1]})
    wald = pred.joint_test(kind="wald")
    emp = pred.joint_test(kind="empirical")
    # p-values need not be identical, but both should be in [0, 1]
    assert 0 <= float(wald.pvalue) <= 1
    assert 0 <= float(emp.pvalue) <= 1
    # Under near-Gaussianity they should not diverge wildly.
    # With a small number of draws the empirical p-value can be exactly 0
    # when Q_obs exceeds every draw-Q; that is valid, not a bug.
    if float(emp.pvalue) > 0:
        ratio = float(emp.pvalue) / (float(wald.pvalue) + 1e-10)
        assert 0.01 < ratio < 100


def test_joint_test_invalid_kind_raises(fit_logit):
    est = GComputation(fit_logit, method="simulation", n_sim=100, seed=42)
    pred = est.predict(atexog={"treatment": [0, 1]})
    with pytest.raises(ValueError, match="kind must be 'wald' or 'empirical'"):
        pred.joint_test(kind="foobar")


# ---------------------------------------------------------------------------
# G6 — Simultaneous (sup-t) confidence bands
# ---------------------------------------------------------------------------


def test_simultaneous_ci_wider_than_pointwise_simulation(fit_logit):
    """sup-t bands are wider than per-component CIs for simulation results."""
    est = GComputation(fit_logit, method="simulation", n_sim=300, seed=42)
    pred = est.predict(atexog={"treatment": [0, 1]})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(correction="sup-t")
    # Simultaneous bands should be wider (or equal) for every component
    assert np.all(lo_sim <= lo_point)
    assert np.all(hi_sim >= hi_point)


def test_simultaneous_ci_wider_than_pointwise_delta(fit_logit):
    """sup-t bands are wider than per-component CIs for delta results."""
    est = GComputation(fit_logit, method="delta")
    pred = est.predict(atexog={"treatment": [0, 1]})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(correction="sup-t")
    assert np.all(lo_sim <= lo_point)
    assert np.all(hi_sim >= hi_point)


def test_simultaneous_ci_scalar_result(fit_logit):
    """sup-t on a scalar estimand is close to the ordinary CI (Monte Carlo noise)."""
    est = GComputation(fit_logit, method="simulation", n_sim=200, seed=42)
    pred = est.predict(atexog={"treatment": 1})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(correction="sup-t")
    # For a scalar, sup-t critical value ≈ percentile critical value,
    # but finite-sample Monte Carlo means they're not identical.
    # Check they're within ~5% relative tolerance.
    np.testing.assert_allclose(lo_sim, lo_point, rtol=0.05)
    np.testing.assert_allclose(hi_sim, hi_point, rtol=0.05)


# ---------------------------------------------------------------------------
# G7b — Multi-outcome selector & to_frame() ergonomics
# ---------------------------------------------------------------------------


def _make_multinomial_df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])
    return df


def test_multi_outcome_unsliced_to_frame_tiles_scenarios():
    """to_frame() on unsliced multi-outcome result tiles scenarios per outcome."""
    df = _make_multinomial_df()
    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    est = GComputation(fit, method="delta")
    pred = est.predict(atexog={"treatment": [0, 1]})

    frame = pred.to_frame()
    assert len(frame) == 6  # 2 scenarios × 3 outcomes
    assert "outcome" in frame.columns
    assert "treatment" in frame.columns
    # Scenario columns should be tiled per outcome
    assert list(frame["treatment"]) == [0, 0, 0, 1, 1, 1]
    assert list(frame["outcome"]) == ["0", "1", "2", "0", "1", "2"]


def test_multi_outcome_sliced_to_frame_works():
    """to_frame() on sliced multi-outcome result should work."""
    df = _make_multinomial_df()
    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    est = GComputation(fit, method="delta")
    pred = est.predict(atexog={"treatment": [0, 1]}, outcome=1)

    frame = pred.to_frame()
    assert len(frame) == 2  # two treatment values
    assert "estimate" in frame.columns


# ---------------------------------------------------------------------------
# §7 Acceptance criteria — quantitative verification (G5/G6/G7b only)
# ---------------------------------------------------------------------------


def test_g7b_outcome_then_test_correct():
    """.test() after .outcome() on a delta result must use the sliced gradient."""
    df = _make_multinomial_df()
    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    est = GComputation(fit, method="delta")
    pred = est.predict()

    sliced = pred.outcome(1)
    # This should work and produce a finite p-value
    tr = sliced.test(value=0.0)
    pval = float(np.asarray(tr.pvalue).ravel()[0])
    assert np.isfinite(pval)
    assert 0.0 <= pval <= 1.0


def test_outcome_then_conf_int_on_simulation_multi_outcome():
    """.conf_int() after .outcome() on simulation multi-outcome result."""
    df = _make_multinomial_df()
    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    est = GComputation(fit, method="simulation", n_sim=200, seed=42)
    pred = est.predict()
    sliced = pred.outcome(1)

    # Should not raise and should return finite bounds
    lo, hi = sliced.conf_int()
    assert np.isfinite(float(np.asarray(lo).ravel()[0]))
    assert np.isfinite(float(np.asarray(hi).ravel()[0]))


def test_outcome_then_test_on_bootstrap_multi_outcome():
    """.test() after .outcome() on bootstrap multi-outcome result."""
    df = _make_multinomial_df()
    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    est = GComputation(fit, method="bootstrap", B=50, seed=42)
    pred = est.predict()
    sliced = pred.outcome(1)

    tr = sliced.test(value=0.0)
    pval = float(np.asarray(tr.pvalue).ravel()[0])
    assert np.isfinite(pval)
    assert 0.0 <= pval <= 1.0
