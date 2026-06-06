"""Tests for structural guards (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import Margins, drop_outliers, reimpute
from pymargins._transforms import IdentityStage

# ---------------------------------------------------------------------------
# G4: survey_design + source_data override or row-altering stage
# ---------------------------------------------------------------------------


def test_survey_design_plus_reimpute_raises():
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

    with pytest.raises(ValueError, match="survey_design is not compatible"):
        Margins(
            fit,
            survey_design=sd,
            transforms=[reimpute(lambda d: d.fillna(0), incomplete=df)],
            method="bootstrap",
        )


def test_survey_design_plus_drop_raises():
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

    with pytest.raises(ValueError, match="survey_design is not compatible"):
        Margins(
            fit,
            survey_design=sd,
            transforms=[drop_outliers(lambda f: f["x"].abs() > 3)],
            method="bootstrap",
        )


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
    m = Margins(fit, survey_design=sd, method="bootstrap", n_boot=10, rng_seed=1)
    r = m.predict()
    assert np.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# G5: matching + transforms raises
# ---------------------------------------------------------------------------


class _FakeMatcher:
    def __init__(self, n):
        self.matched_data = pd.DataFrame({"x": range(n), "y": range(n)})
        self.cluster_ids = np.arange(n)

    def rematch(self, data):
        return data


def test_matching_plus_transforms_raises():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    with pytest.raises(
        ValueError, match="matching= and transforms= cannot be used together"
    ):
        Margins(
            fit,
            matching=_FakeMatcher(n),
            transforms=[IdentityStage()],
            method="bootstrap",
        )


# ---------------------------------------------------------------------------
# Strict mode: transforms is NOT required
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

    m = Margins(
        fit,
        transforms=[IdentityStage()],
        method="bootstrap",
        n_boot=10,
        rng_seed=1,
        bootstrap_config={"ci_method": "bca"},
    )
    with pytest.raises(ValueError, match="ci_method='bca' is not supported"):
        m.predict()


def test_strict_mode_transforms_not_required():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    # Should succeed without specifying transforms
    m = Margins(
        fit,
        strict=True,
        phi=None,
        phi_inv=None,
        vcov=None,
        weights=None,
        at="overall",
        level=0.95,
        method="delta",
        kappa_threshold=0.3,
        rng_seed=None,
        n_sim=4000,
        n_boot=1000,
        n_jobs=1,
        gradient_backend="autodiff",
        fd_step=1e-6,
        diagnostics=True,
        cluster=None,
        block_size=None,
        bootstrap_config=None,
        progress_bar=False,
        matching=None,
        formula=None,
        data=None,
    )
    assert m.transforms is None
