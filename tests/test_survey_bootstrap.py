"""Tests for stratified survey bootstrap resampling."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, SurveyDesign, steps
from pymargins._inference._bootstrap import _generate_resample_indices


def test_stratified_resample_no_cross_contamination():
    """Resampled PSUs must stay within their stratum."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "psu": rng.integers(0, 30, n),
            "strat": rng.integers(0, 3, n),
        }
    )

    idx_list = _generate_resample_indices(
        rng_seed=1,
        n_boot=20,
        n_obs=n,
        cluster_ids=df.psu.values,
        strata=df.strat.values,
    )

    for idx in idx_list:
        for h in np.unique(df.strat.values):
            in_h = df.strat.values[idx] == h
            resampled_psus = df.psu.values[idx][in_h]
            original_psus = df.psu.values[df.strat.values == h]
            assert np.all(np.isin(resampled_psus, original_psus))


def test_bootstrap_se_close_to_linearization():
    """Bootstrap SE should be in the same ballpark as linearization SE."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "psu": rng.integers(0, 30, n),
            "strat": rng.integers(0, 3, n),
        }
    )
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    df["w"] = rng.uniform(0.5, 2.0, n)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    d = SurveyDesign(weights=df.w.values, psu=df.psu.values, strata=df.strat.values)
    node = steps.input(df, design=d)

    m_boot = GComputation(
        node,
        outcome=fit,
        method="bootstrap",
        B=400,
        seed=1,
        weights=df.w.values,
    )
    r_boot = m_boot.dydx("x")

    m_lin = GComputation(node, outcome=fit, weights=df.w.values)
    r_lin = m_lin.dydx("x")

    rel_diff = abs(r_boot.std_error - r_lin.std_error) / r_lin.std_error
    assert rel_diff < 0.3, (
        f"Bootstrap SE {r_boot.std_error} too far from linearization {r_lin.std_error}"
    )


def test_bootstrap_with_explicit_weights():
    """Bootstrap must work when both survey_design and weights are given."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "psu": rng.integers(0, 20, n),
            "strat": rng.integers(0, 2, n),
        }
    )
    df["y"] = (rng.random(n) < 0.5).astype(int)
    df["w"] = rng.uniform(0.5, 2.0, n)
    fit = smf.glm("y ~ x", df, family=sm.families.Binomial()).fit()
    d = SurveyDesign(weights=df.w.values, psu=df.psu.values, strata=df.strat.values)
    node = steps.input(df, design=d)

    m = GComputation(
        node,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=1,
        weights=df.w.values,
    )
    r = m.dydx("x")
    assert np.isfinite(r.std_error)
    assert r.std_error > 0
