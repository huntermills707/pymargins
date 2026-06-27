"""Correctness tests against independent reference implementations.

These tests replicate the analyses from demo/williams_2012_demo.py and
compare pymargins (delta method) against reference values computed with:
  - StatsModels get_margeff() (delta method)
  - R marginaleffects (delta method by default)

The delta method is used by default in the new engine; inference is
apples-to-apples with the reference implementations.

Reference values were verified to agree across both independent
implementations (within expected numerical tolerances).
"""

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation

# ---------------------------------------------------------------------------
# Fixture: exact data from Williams 2012 demo
# ---------------------------------------------------------------------------


@pytest.fixture
def df_williams():
    """Synthetic NHANES-like data from demo/williams_2012_demo.py."""
    rng = np.random.default_rng(42)
    n = 5000
    female = rng.binomial(1, 0.52, size=n)
    black = rng.binomial(1, 0.11, size=n)
    age = rng.integers(20, 75, size=n)
    agegrp = pd.cut(
        age, bins=[19, 29, 39, 49, 59, 69, 100], labels=[1, 2, 3, 4, 5, 6]
    ).astype(int)
    bmi = 22 + 0.15 * age + 1.5 * female + rng.normal(0, 4, size=n)
    bmi = np.clip(bmi, 15, 50)
    lp = (
        -4.0
        + 0.55 * black
        + 0.10 * female
        + 0.06 * age
        + 0.03 * bmi
        + 0.5 * (agegrp == 2).astype(float)
        + 0.9 * (agegrp == 3).astype(float)
        + 1.4 * (agegrp == 4).astype(float)
        + 2.0 * (agegrp == 5).astype(float)
        + 2.6 * (agegrp == 6).astype(float)
    )
    diabetes = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    bp = (
        110
        + 0.4 * age
        + 2.5 * black
        + 1.2 * female
        + 0.5 * bmi
        + rng.normal(0, 8, size=n)
    )
    return pd.DataFrame(
        {
            "diabetes": diabetes,
            "bp": bp,
            "black": black,
            "female": female,
            "age": age,
            "agegrp": agegrp,
            "bmi": bmi,
        }
    )


@pytest.fixture
def fit_logit(df_williams):
    return smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df_williams,
        family=sm.families.Binomial(),
    ).fit()


@pytest.fixture
def fit_ols(df_williams):
    return smf.ols(
        "bp ~ C(black) + C(female) + age + bmi",
        data=df_williams,
    ).fit()


# ---------------------------------------------------------------------------
# Reference values (verified against StatsModels + R marginaleffects)
# ---------------------------------------------------------------------------

# -- Predictions (point estimates only; CIs depend on inference scale) ----
APM_EXPECTED = {
    1: 0.5121,
    2: 0.5950,
    3: 0.6575,
    4: 0.7388,
    5: 0.8062,
    6: 0.9566,
}

AAP_EXPECTED = {
    1: 0.5152,
    2: 0.5770,
    3: 0.6251,
    4: 0.6915,
    5: 0.7521,
    6: 0.9272,
}

REPR_EXPECTED = {
    20: 0.2079,
    50: 0.7054,
    70: 0.9127,
}

# -- Slopes (delta method, now with proper x64 precision) -----------------
MEM_AGE_EXPECTED = 0.0166
AME_AGE_EXPECTED = 0.0106

# Reference CIs for slopes (R marginaleffects delta method, response scale)
MEM_AGE_CI = (0.0111, 0.0221)
AME_AGE_CI = (0.00698, 0.0142)

# -- Risk ratios from log_scale contrasts ---------------------------------
RR_BLACK_EXPECTED = 1.1415
RR_FEMALE_EXPECTED = 1.0357
RR_BLACK_FEMALE0_EXPECTED = 1.1440
RR_BLACK_FEMALE1_EXPECTED = 1.1394

# Reference CIs for risk ratios (R marginaleffects: ln-ratio + exp transform)
RR_BLACK_CI = (1.09, 1.20)
RR_FEMALE_CI = (0.982, 1.09)
RR_BLACK_FEMALE0_CI = (1.09, 1.20)
RR_BLACK_FEMALE1_CI = (1.09, 1.19)

# -- Direct ratio (same point estimate as RR, different inference) ---------
DIRECT_RATIO_BLACK_EXPECTED = 1.1415
DIRECT_RATIO_FEMALE_EXPECTED = 1.0232

# -- True lift = RR - 1 -----------------------------------------------------
LIFT_BLACK_EXPECTED = 0.1415
LIFT_FEMALE_EXPECTED = 0.0232
LIFT_BLACK_CI = (0.088, 0.197)
LIFT_FEMALE_CI = (-0.012, 0.059)

# -- OLS ------------------------------------------------------------------
OLS_AGE_COEF_EXPECTED = 0.3938
OLS_AGE_SE_EXPECTED = 0.0082
OLS_AGE_CI = (0.378, 0.41)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_point_estimate(result, expected, abs_tol=1e-4):
    assert float(result.estimate) == pytest.approx(expected, abs=abs_tol)


def _assert_ci(result, expected_lo, expected_hi, abs_tol=1e-3):
    assert float(result.conf_int_lower) == pytest.approx(expected_lo, abs=abs_tol)
    assert float(result.conf_int_upper) == pytest.approx(expected_hi, abs=abs_tol)


# ---------------------------------------------------------------------------
# 2. Adjusted Predictions at the Means (APM)
# ---------------------------------------------------------------------------


def test_apm_point_estimates(fit_logit):
    m = GComputation(fit_logit, at="typical", scale="log")
    apm = m.predict(atexog={"agegrp": list(range(1, 7))})

    assert apm.estimate.shape == (6,)
    for i, g in enumerate(range(1, 7)):
        assert float(apm.estimate[i]) == pytest.approx(APM_EXPECTED[g], abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Average Adjusted Predictions (AAP)
# ---------------------------------------------------------------------------


def test_aap_point_estimates(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    aap = m.predict(atexog={"agegrp": list(range(1, 7))})

    assert aap.estimate.shape == (6,)
    for i, g in enumerate(range(1, 7)):
        assert float(aap.estimate[i]) == pytest.approx(AAP_EXPECTED[g], abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Predictions at Representative Values
# ---------------------------------------------------------------------------


def test_repr_values_point_estimates(fit_logit):
    m = GComputation(fit_logit, at="typical", scale="log")
    repr_pred = m.predict(atexog={"age": [20, 50, 70]})

    assert repr_pred.estimate.shape == (3,)
    for i, a in enumerate([20, 50, 70]):
        assert float(repr_pred.estimate[i]) == pytest.approx(REPR_EXPECTED[a], abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Marginal Effects at the Means (MEM) — continuous
# ---------------------------------------------------------------------------


def test_mem_age_point_estimate_and_ci(fit_logit):
    m = GComputation(fit_logit, at="typical", scale="log")
    mem = m.dydx("age")

    _assert_point_estimate(mem, MEM_AGE_EXPECTED, abs_tol=1e-4)
    # CI on reporting scale; allow modest tolerance vs R reference
    _assert_ci(mem, MEM_AGE_CI[0], MEM_AGE_CI[1], abs_tol=1e-3)


# ---------------------------------------------------------------------------
# 6. Average Marginal Effects (AME) — continuous
# ---------------------------------------------------------------------------


def test_ame_age_point_estimate_and_ci(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    ame = m.dydx("age")

    _assert_point_estimate(ame, AME_AGE_EXPECTED, abs_tol=1e-4)
    _assert_ci(ame, AME_AGE_CI[0], AME_AGE_CI[1], abs_tol=1e-3)


# ---------------------------------------------------------------------------
# 7. Discrete changes for dummy variables (risk ratios on log scale)
# ---------------------------------------------------------------------------


def test_discrete_black_risk_ratio_and_ci(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    disc = m.contrasts(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        contrasts=[+1, -1],
    )
    _assert_point_estimate(disc, RR_BLACK_EXPECTED, abs_tol=1e-4)
    _assert_ci(disc, RR_BLACK_CI[0], RR_BLACK_CI[1], abs_tol=1e-2)


def test_discrete_female_at_typical_risk_ratio_and_ci(fit_logit):
    m = GComputation(fit_logit, at="typical", scale="log")
    disc = m.contrasts(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        contrasts=[+1, -1],
    )
    _assert_point_estimate(disc, RR_FEMALE_EXPECTED, abs_tol=1e-4)
    _assert_ci(disc, RR_FEMALE_CI[0], RR_FEMALE_CI[1], abs_tol=1e-2)


# ---------------------------------------------------------------------------
# 8. Marginal Effects at Representative Values (MER)
# ---------------------------------------------------------------------------


def test_mer_black_female0_risk_ratio_and_ci(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    mer = m.contrasts(
        scenarios=[
            {"atexog": {"black": 1, "female": 0}, "label": "black=1, female=0"},
            {"atexog": {"black": 0, "female": 0}, "label": "black=0, female=0"},
        ],
        contrasts=[+1, -1],
    )
    _assert_point_estimate(mer, RR_BLACK_FEMALE0_EXPECTED, abs_tol=1e-4)
    _assert_ci(mer, RR_BLACK_FEMALE0_CI[0], RR_BLACK_FEMALE0_CI[1], abs_tol=1e-2)


def test_mer_black_female1_risk_ratio_and_ci(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    mer = m.contrasts(
        scenarios=[
            {"atexog": {"black": 1, "female": 1}, "label": "black=1, female=1"},
            {"atexog": {"black": 0, "female": 1}, "label": "black=0, female=1"},
        ],
        contrasts=[+1, -1],
    )
    _assert_point_estimate(mer, RR_BLACK_FEMALE1_EXPECTED, abs_tol=1e-4)
    _assert_ci(mer, RR_BLACK_FEMALE1_CI[0], RR_BLACK_FEMALE1_CI[1], abs_tol=1e-2)


# ---------------------------------------------------------------------------
# 9. Direct ratio via evaluate() on linear_scale
# ---------------------------------------------------------------------------


def test_direct_ratio_black_matches_log_scale_point_estimate(fit_logit):
    """Direct ratio via evaluate() should match log_scale RR point estimate."""
    m = GComputation(fit_logit, at="overall", scale="identity")
    ratio = m.evaluate(
        scenarios=[
            {"atexog": {"black": 1}},
            {"atexog": {"black": 0}},
        ],
        compose=lambda p: p[0] / p[1],
    )
    _assert_point_estimate(ratio, DIRECT_RATIO_BLACK_EXPECTED, abs_tol=1e-4)


def test_direct_ratio_female_matches_log_scale_point_estimate(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="identity")
    ratio = m.evaluate(
        scenarios=[
            {"atexog": {"female": 1}},
            {"atexog": {"female": 0}},
        ],
        compose=lambda p: p[0] / p[1],
    )
    _assert_point_estimate(ratio, DIRECT_RATIO_FEMALE_EXPECTED, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# 10. True lift = RR - 1 (derived from log_scale)
# ---------------------------------------------------------------------------


def test_true_lift_black_from_log_scale(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    rr = m.contrasts(
        scenarios=[
            {"atexog": {"black": 1}},
            {"atexog": {"black": 0}},
        ],
        contrasts=[+1, -1],
    )
    lift_est = float(rr.estimate) - 1.0
    lo, hi = rr.conf_int()
    lift_ci = (float(lo) - 1.0, float(hi) - 1.0)

    assert lift_est == pytest.approx(LIFT_BLACK_EXPECTED, abs=1e-4)
    assert lift_ci[0] == pytest.approx(LIFT_BLACK_CI[0], abs=1e-3)
    assert lift_ci[1] == pytest.approx(LIFT_BLACK_CI[1], abs=1e-3)


def test_true_lift_female_from_log_scale(fit_logit):
    m = GComputation(fit_logit, at="overall", scale="log")
    rr = m.contrasts(
        scenarios=[
            {"atexog": {"female": 1}},
            {"atexog": {"female": 0}},
        ],
        contrasts=[+1, -1],
    )
    lift_est = float(rr.estimate) - 1.0
    lo, hi = rr.conf_int()
    lift_ci = (float(lo) - 1.0, float(hi) - 1.0)

    assert lift_est == pytest.approx(LIFT_FEMALE_EXPECTED, abs=1e-4)
    assert lift_ci[0] == pytest.approx(LIFT_FEMALE_CI[0], abs=1e-3)
    assert lift_ci[1] == pytest.approx(LIFT_FEMALE_CI[1], abs=1e-3)


# ---------------------------------------------------------------------------
# 11. Lift via evaluate() — direct delta method on (p1-p0)/p0
# ---------------------------------------------------------------------------

# Reference: manual computation of (mean(Y_1) - mean(Y_0)) / mean(Y_0)
EVALUATE_LIFT_BLACK_EXPECTED = 0.1415
EVALUATE_LIFT_BLACK_CI = (0.087, 0.196)
EVALUATE_LIFT_FEMALE_EXPECTED = 0.0232
EVALUATE_LIFT_FEMALE_CI = (-0.012, 0.058)


def test_evaluate_lift_black_matches_manual_and_rr_minus_one(fit_logit, df_williams):
    """evaluate() lift = (p1-p0)/p0 should match manual comp and RR-1."""
    # Manual ground truth
    tmp_b1 = df_williams.copy()
    tmp_b1["black"] = 1
    tmp_b0 = df_williams.copy()
    tmp_b0["black"] = 0
    p1 = fit_logit.predict(tmp_b1).mean()
    p0 = fit_logit.predict(tmp_b0).mean()
    manual_lift = (p1 - p0) / p0

    # evaluate() lift on linear_scale
    m = GComputation(fit_logit, at="overall", scale="identity")
    lift_eval = m.evaluate(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        compose=lambda p: (p[0] - p[1]) / p[1],
    )

    # Point estimate matches manual exactly
    assert float(lift_eval.estimate) == pytest.approx(manual_lift, abs=1e-6)
    assert float(lift_eval.estimate) == pytest.approx(
        EVALUATE_LIFT_BLACK_EXPECTED, abs=1e-4
    )

    # CI matches reference (direct delta method on ratio scale)
    _assert_ci(
        lift_eval, EVALUATE_LIFT_BLACK_CI[0], EVALUATE_LIFT_BLACK_CI[1], abs_tol=1e-3
    )

    # Also matches RR - 1 from log_scale (point estimate only)
    m_rr = GComputation(fit_logit, at="overall", scale="log")
    rr = m_rr.contrasts(
        scenarios=[
            {"atexog": {"black": 1}},
            {"atexog": {"black": 0}},
        ],
        contrasts=[+1, -1],
    )
    lift_from_rr = float(rr.estimate) - 1.0
    assert float(lift_eval.estimate) == pytest.approx(lift_from_rr, abs=1e-6)


def test_evaluate_lift_female_matches_manual_and_rr_minus_one(fit_logit, df_williams):
    """evaluate() lift for female should match manual comp and RR-1."""
    tmp_f1 = df_williams.copy()
    tmp_f1["female"] = 1
    tmp_f0 = df_williams.copy()
    tmp_f0["female"] = 0
    pf1 = fit_logit.predict(tmp_f1).mean()
    pf0 = fit_logit.predict(tmp_f0).mean()
    manual_lift = (pf1 - pf0) / pf0

    m = GComputation(fit_logit, at="overall", scale="identity")
    lift_eval = m.evaluate(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        compose=lambda p: (p[0] - p[1]) / p[1],
    )

    assert float(lift_eval.estimate) == pytest.approx(manual_lift, abs=1e-6)
    assert float(lift_eval.estimate) == pytest.approx(
        EVALUATE_LIFT_FEMALE_EXPECTED, abs=1e-4
    )
    _assert_ci(
        lift_eval, EVALUATE_LIFT_FEMALE_CI[0], EVALUATE_LIFT_FEMALE_CI[1], abs_tol=1e-3
    )

    m_rr = GComputation(fit_logit, at="overall", scale="log")
    rr = m_rr.contrasts(
        scenarios=[
            {"atexog": {"female": 1}},
            {"atexog": {"female": 0}},
        ],
        contrasts=[+1, -1],
    )
    lift_from_rr = float(rr.estimate) - 1.0
    assert float(lift_eval.estimate) == pytest.approx(lift_from_rr, abs=1e-6)


# ---------------------------------------------------------------------------
# 13. OLS model
# ---------------------------------------------------------------------------


def test_ols_mem_age_matches_coefficient_and_ci(fit_ols):
    m = GComputation(fit_ols, at="typical", scale="identity")
    mem = m.dydx("age")
    _assert_point_estimate(mem, OLS_AGE_COEF_EXPECTED, abs_tol=1e-4)
    _assert_ci(mem, OLS_AGE_CI[0], OLS_AGE_CI[1], abs_tol=1e-2)
    assert float(mem.std_error) == pytest.approx(OLS_AGE_SE_EXPECTED, abs=1e-4)


def test_ols_ame_age_matches_coefficient_and_ci(fit_ols):
    m = GComputation(fit_ols, at="overall", scale="identity")
    ame = m.dydx("age")
    _assert_point_estimate(ame, OLS_AGE_COEF_EXPECTED, abs_tol=1e-4)
    _assert_ci(ame, OLS_AGE_CI[0], OLS_AGE_CI[1], abs_tol=1e-2)
    assert float(ame.std_error) == pytest.approx(OLS_AGE_SE_EXPECTED, abs=1e-4)
