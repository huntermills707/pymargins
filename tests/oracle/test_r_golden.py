"""Layer-2 R-golden comparisons.

Design §4.2, req §7. Added in 0.4.0 (R1).
"""

from __future__ import annotations

import numpy as np

from pymargins import GComputation, SurveyDesign, steps
from pymargins._adapters import auto_detect_adapter

from ._tolerances import TOL_VCOV
from .conftest import assert_coef_aligned, assert_matches_golden, load_golden


def _assert_vcov_matrix(fit, golden, groups=None):
    """Compare the adapter's resolved Σ̂ to the golden's vcov_matrix."""
    vcov = golden["quantities"].get("vcov_matrix")
    if vcov is None:
        return
    adapter = auto_detect_adapter(fit)
    spec = golden.get("vcov")
    if spec == "cluster(g)":
        spec = {"type": "cluster", "groups": groups}
    elif spec == "nonrobust":
        spec = None
    Sigma = np.asarray(adapter.covariance(spec))
    np.testing.assert_allclose(
        Sigma.ravel(), np.asarray(vcov), rtol=TOL_VCOV, atol=0.0
    )


def test_ols_predict_overall_nonrobust(fit_ols):
    g = load_golden("ols_predict_overall_nonrobust")
    assert_coef_aligned(fit_ols, g)
    r = GComputation(fit_ols, at="overall", method="delta").predict()
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_ols_ame_x1_nonrobust(fit_ols):
    g = load_golden("ols_ame_x1_nonrobust")
    assert_coef_aligned(fit_ols, g)
    r = GComputation(fit_ols, at="overall", method="delta").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_ols_ame_x1_hc1(fit_ols):
    g = load_golden("ols_ame_x1_hc1")
    assert_coef_aligned(fit_ols, g)
    r = GComputation(fit_ols, at="overall", method="delta", vcov="HC1").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_ols_ame_x1_cluster(fit_ols, oracle_df):
    g = load_golden("ols_ame_x1_cluster")
    assert_coef_aligned(fit_ols, g)
    _assert_vcov_matrix(fit_ols, g, groups=oracle_df["g"].values)
    r = GComputation(
        fit_ols,
        at="overall",
        method="delta",
        vcov={"type": "cluster", "groups": oracle_df["g"].values},
    ).dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_ols_contrast_treat_nonrobust(fit_ols):
    g = load_golden("ols_contrast_treat_nonrobust")
    assert_coef_aligned(fit_ols, g)
    r = GComputation(fit_ols, at="overall", method="delta").contrasts(
        scenarios=[{"atexog": {"treat": 1.0}}, {"atexog": {"treat": 0.0}}],
        contrasts=[1.0, -1.0],
    )
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_predict_overall_nonrobust(fit_logit):
    g = load_golden("logit_predict_overall_nonrobust")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(fit_logit, at="overall", method="delta").predict()
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_predict_at_treat1_nonrobust(fit_logit):
    g = load_golden("logit_predict_at_treat1_nonrobust")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(fit_logit, at="overall", method="delta").predict(
        atexog={"treat": 1.0}
    )
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_ame_x1_nonrobust(fit_logit):
    g = load_golden("logit_ame_x1_nonrobust")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(fit_logit, at="overall", method="delta").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_ame_x1_hc1(fit_logit):
    g = load_golden("logit_ame_x1_hc1")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(fit_logit, at="overall", method="delta", vcov="HC1").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_ame_x1_cluster(fit_logit, oracle_df):
    g = load_golden("logit_ame_x1_cluster")
    assert_coef_aligned(fit_logit, g)
    _assert_vcov_matrix(fit_logit, g, groups=oracle_df["g"].values)
    r = GComputation(
        fit_logit,
        at="overall",
        method="delta",
        vcov={"type": "cluster", "groups": oracle_df["g"].values},
    ).dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_contrast_treat_nonrobust(fit_logit):
    g = load_golden("logit_contrast_treat_nonrobust")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(fit_logit, at="overall", method="delta").contrasts(
        scenarios=[{"atexog": {"treat": 1.0}}, {"atexog": {"treat": 0.0}}],
        contrasts=[1.0, -1.0],
    )
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_logit_ame_x1_weighted_nonrobust(fit_logit, oracle_df):
    g = load_golden("logit_ame_x1_weighted_nonrobust")
    assert_coef_aligned(fit_logit, g)
    r = GComputation(
        fit_logit, at="overall", method="delta", weights=oracle_df["w"].values
    ).dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_probit_ame_x1_nonrobust(fit_probit):
    g = load_golden("probit_ame_x1_nonrobust")
    assert_coef_aligned(fit_probit, g)
    r = GComputation(fit_probit, at="overall", method="delta").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_poisson_predict_overall_nonrobust(fit_poisson):
    g = load_golden("poisson_predict_overall_nonrobust")
    assert_coef_aligned(fit_poisson, g)
    r = GComputation(fit_poisson, at="overall", method="delta").predict()
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_poisson_ame_x1_nonrobust(fit_poisson):
    g = load_golden("poisson_ame_x1_nonrobust")
    assert_coef_aligned(fit_poisson, g)
    r = GComputation(fit_poisson, at="overall", method="delta").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_poisson_ame_x1_hc1(fit_poisson):
    g = load_golden("poisson_ame_x1_hc1")
    assert_coef_aligned(fit_poisson, g)
    r = GComputation(fit_poisson, at="overall", method="delta", vcov="HC1").dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_survey_logit_predict_overall_linearized(fit_logit_weighted, oracle_df):
    g = load_golden("survey_logit_predict_overall_linearized")
    assert_coef_aligned(fit_logit_weighted, g)
    design = SurveyDesign(
        weights=oracle_df["w"].values,
        psu=oracle_df["psu"].values,
        strata=oracle_df["strata"].values,
    )
    r = GComputation(
        steps.input(oracle_df, design=design), outcome=fit_logit_weighted, method="delta"
    ).predict()
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)


def test_survey_logit_ame_x1_linearized(fit_logit_weighted, oracle_df):
    g = load_golden("survey_logit_ame_x1_linearized")
    assert_coef_aligned(fit_logit_weighted, g)
    design = SurveyDesign(
        weights=oracle_df["w"].values,
        psu=oracle_df["psu"].values,
        strata=oracle_df["strata"].values,
    )
    r = GComputation(
        steps.input(oracle_df, design=design), outcome=fit_logit_weighted, method="delta"
    ).dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error,
                          conf_low=r.conf_int_lower, conf_high=r.conf_int_upper)
