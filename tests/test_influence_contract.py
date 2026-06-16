"""Tests for the ModelAdapter.influence() contract (Phase 1, W1.3).

These tests validate:
1.  Tier-1 adapters produce ψ that reproduces the survey linearization path.
2.  Adapter-level ψ fed through a delta gradient matches GraphResult.influence().
3.  Tier-2/3 adapters return None for influence().
4.  ψ row count aligns to training_data length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, SurveyDesign
from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter
from pymargins._inference._linearization import linearization_meat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "stratum": rng.integers(0, 4, size=n),
            "psu": rng.integers(0, 3, size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )


@pytest.fixture
def design(df):
    return SurveyDesign(
        strata=df["stratum"].values,
        psu=df["psu"].values,
        weights=df["w"].values,
    )


# ---------------------------------------------------------------------------
# Test 1: ψ reproduces survey linearization
# ---------------------------------------------------------------------------

def _survey_covariance_from_psi(adapter, design):
    """Compute sandwich using adapter.influence() as the input to linearization_meat."""
    psi = np.asarray(adapter.influence())
    w = np.asarray(design.weights)
    psu = None if design.psu is None else np.asarray(design.psu)
    strata = None if design.strata is None else np.asarray(design.strata)
    fpc = None
    nest = design.nest

    w_mean = w.mean()
    if w_mean > 0:
        w = w / w_mean
    u = w[:, None] * psi
    meat = linearization_meat(u, psu, strata, fpc, nest)
    return meat


@pytest.mark.parametrize(
    "adapter_factory",
    [
        lambda df: StatsmodelsOLSAdapter(
            smf.ols("y ~ x1 + x2", data=df).fit()
        ),
        lambda df: StatsmodelsGLMAdapter(
            smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()
        ),
    ],
    ids=["ols", "glm"],
)
def test_influence_matches_survey_linearization(adapter_factory, df, design):
    adapter = adapter_factory(df)
    psi_based = _survey_covariance_from_psi(adapter, design)
    survey_based = np.asarray(adapter._survey_covariance(design))
    np.testing.assert_allclose(psi_based, survey_based, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Adapter ψ @ gradient matches GraphResult.influence()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fit_factory",
    [
        lambda df: smf.ols("y ~ x1 + x2", data=df).fit(),
        lambda df: smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit(),
        lambda df: smf.logit("y ~ x1 + x2", data=df).fit(disp=False),
        lambda df: smf.poisson("y ~ x1 + x2", data=df).fit(disp=False),
    ],
    ids=["ols", "glm", "logit", "poisson"],
)
def test_influence_matches_result_influence(fit_factory, df):
    fit = fit_factory(df)
    from pymargins._adapters import auto_detect_adapter
    adapter = auto_detect_adapter(fit)
    est = GComputation(fit, at="overall", method="delta")
    result = est.predict()

    psi_beta = np.asarray(adapter.influence())  # (n, p)
    g = np.asarray(result.gradient)  # (p,) or (k, p)

    if g.ndim == 1:
        psi_h = psi_beta @ g
    else:
        psi_h = psi_beta @ g.T

    result_infl = np.asarray(result.influence())
    np.testing.assert_allclose(psi_h, result_infl, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: Tier-2 adapters return None
# ---------------------------------------------------------------------------

def test_influence_none_on_tier2():
    # Any adapter without score_obs should inherit the default None.
    # We test by importing adapters that are known tier-2/3.
    from pymargins._adapters import auto_detect_adapter

    # lifelines adapter (if available)
    try:
        from lifelines import CoxPHFitter

        df_surv = pd.DataFrame(
            {
                "T": np.random.exponential(1, 50),
                "E": np.ones(50, dtype=int),
                "x": np.random.normal(size=50),
            }
        )
        fit = CoxPHFitter().fit(df_surv, duration_col="T", event_col="E")
        adapter = auto_detect_adapter(fit, data=df_surv)
        assert adapter.influence() is None
        methods = adapter.supported_inference_methods
        assert isinstance(methods, set)
    except ImportError:
        pytest.skip("lifelines not installed")

    # linearmodels adapter (if available)
    try:
        from linearmodels.panel import PooledOLS

        # PooledOLS needs a MultiIndex
        df_panel = pd.DataFrame(
            {
                "y": np.random.normal(size=60),
                "x": np.random.normal(size=60),
            },
            index=pd.MultiIndex.from_product(
                [range(10), range(6)], names=["entity", "time"]
            ),
        )
        fit = PooledOLS(df_panel.y, df_panel[["x"]]).fit()
        adapter = auto_detect_adapter(fit)
        assert adapter.influence() is None
        methods = adapter.supported_inference_methods
        assert isinstance(methods, set)
    except ImportError:
        pytest.skip("linearmodels not installed")


# ---------------------------------------------------------------------------
# Test 4: Index alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fit_factory",
    [
        lambda df: smf.ols("y ~ x1 + x2", data=df).fit(),
        lambda df: smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit(),
    ],
    ids=["ols", "glm"],
)
def test_influence_index_alignment(fit_factory, df):
    fit = fit_factory(df)
    from pymargins._adapters import auto_detect_adapter
    adapter = auto_detect_adapter(fit)
    psi = adapter.influence()
    assert psi is not None
    assert psi.shape[0] == len(adapter.training_data)
