"""Numeric correctness test against R survey::svyglm + marginaleffects."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, SurveyDesign


def test_survey_ame_matches_R():
    """Weighted AME and design-based SE must match R survey reference."""
    df = pd.read_csv("tests/survey_fixture.csv")
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()

    d = SurveyDesign(
        weights=df.w.values,
        psu=df.psu.values,
        strata=df.strat.values,
    )
    m = Margins(fit, survey_design=d, weights=df.w.values)
    r = m.dydx("x")

    ref = pd.read_csv("tests/survey_reference.csv")
    expected_est = ref.loc[0, "estimate"]
    expected_se = ref.loc[0, "std_error"]

    assert np.isclose(float(r.estimate), expected_est, rtol=1e-4), (
        float(r.estimate),
        expected_est,
    )
    assert np.isclose(float(r.std_error), expected_se, rtol=1e-3), (
        float(r.std_error),
        expected_se,
    )
