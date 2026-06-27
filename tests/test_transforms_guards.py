"""Tests for structural guards on the new wiring surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation, steps
from pymargins._graph._compile import CompileError

# ---------------------------------------------------------------------------
# G4: survey_design still works alone
# ---------------------------------------------------------------------------


def test_survey_design_alone_unaffected():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    from pymargins.survey import SurveyDesign

    sd = SurveyDesign(weights=np.ones(n))
    est = GComputation(
        steps.input(df, design=sd),
        outcome=fit,
        method="bootstrap",
        B=10,
        seed=1,
    )
    r = est.predict()
    assert np.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# G5: matching + row-filter stage refused
# ---------------------------------------------------------------------------


class _FakeMatcher:
    def __init__(self, n):
        self.matched_data = pd.DataFrame({"x": range(n), "y": range(n)})
        self.cluster_ids = np.arange(n)
        self.population_note = "matched sample"

    def rematch(self, data):
        return data


def test_matching_plus_row_filter_refused():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    wiring = steps.drop_outliers(
        steps.match(steps.input(df), _FakeMatcher(n)),
        rule=lambda f: f["x"].abs() > 3,
    )
    with pytest.raises(CompileError, match=r"match \+ row-filter"):
        GComputation(wiring, outcome=fit, method="bootstrap")


# ---------------------------------------------------------------------------
# G6: BCa refused with a transform pipeline
# ---------------------------------------------------------------------------


def test_bca_with_transforms_raises():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    est = GComputation(
        steps.drop_outliers(steps.input(df), rule=lambda f: f["x"].abs() > 3),
        outcome=fit,
        method="bootstrap",
        B=10,
        seed=1,
        ci="bca",
    )
    with pytest.raises(ValueError, match="ci_method='bca' is not supported"):
        est.predict()
