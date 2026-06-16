"""Tests for matching support with the GComputation estimator noun.

See notes/MATCHING_API_DESIGN.md for the design specification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation, steps
from pymargins.matching import PysmatchClient

pysmatch = pytest.importorskip("pysmatch", reason="pysmatch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_matching():
    """Synthetic data suitable for propensity score matching."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treated": rng.binomial(1, 0.35, n),
        }
    )
    # Propensity score model
    eta_ps = -0.5 + 0.6 * df["x1"] - 0.4 * df["x2"]
    ps = 1 / (1 + np.exp(-eta_ps))
    df["treated"] = (rng.uniform(size=n) < ps).astype(int)
    # Outcome model
    eta_y = 0.3 + 0.2 * df["x1"] - 0.1 * df["x2"] + 0.5 * df["treated"]
    df["y"] = (rng.uniform(size=n) < (1 / (1 + np.exp(-eta_y)))).astype(float)
    return df


@pytest.fixture
def pysmatch_matcher(df_matching):
    """A fitted pysmatch Matcher on the synthetic data."""
    test = df_matching[df_matching["treated"] == 1].copy()
    control = df_matching[df_matching["treated"] == 0].copy()
    matcher = pysmatch.Matcher(
        test=test,
        control=control,
        yvar="treated",
        exclude=["y"],
    )
    matcher.fit_scores(balance=True, model_type="linear", nmodels=3, n_jobs=1)
    matcher.predict_scores()
    matcher.match(method="min", nmatches=1, threshold=0.3, replacement=False)
    return matcher


@pytest.fixture
def fitted_logit_matched(df_matching, pysmatch_matcher):
    """A statsmodels GLM fit on the matched sample only."""
    matched_df = pysmatch_matcher.matched_data
    fit = smf.glm(
        "y ~ x1 + x2 + treated",
        data=matched_df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": matched_df["match_id"]})
    return fit


# ---------------------------------------------------------------------------
# Protocol validation (now enforced by PysmatchClient / the graph compiler)
# ---------------------------------------------------------------------------


def test_pysmatch_client_requires_matched_data():
    class BadMatcher:
        pass

    with pytest.raises(ValueError, match="matched_data"):
        PysmatchClient(BadMatcher(), treatment_col="treated")


# ---------------------------------------------------------------------------
# Integration with PysmatchClient
# ---------------------------------------------------------------------------


def test_pysmatch_client_attributes(pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    assert hasattr(client, "matched_data")
    assert hasattr(client, "cluster_ids")
    assert len(client.matched_data) == len(client.cluster_ids)
    assert "match_id" in client.matched_data.columns


def test_pysmatch_client_rematch(pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    original = client.matched_data
    # Resample with replacement (bootstrap-like)
    resampled = original.sample(n=len(original), replace=True, random_state=42)
    rematched = client.rematch(resampled)
    assert isinstance(rematched, pd.DataFrame)
    assert "match_id" in rematched.columns


# ---------------------------------------------------------------------------
# Base data filtering
# ---------------------------------------------------------------------------


def test_matching_base_data_is_matched_data(
    fitted_logit_matched, pysmatch_matcher, df_matching
):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    est = GComputation(
        steps.match(steps.input(df_matching), client),
        outcome=fitted_logit_matched,
    )
    pd.testing.assert_frame_equal(est._compiled.base_data, client.matched_data)


def test_matching_at_overall_uses_matched_data(
    fitted_logit_matched, pysmatch_matcher, df_matching
):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    est = GComputation(
        steps.match(steps.input(df_matching), client),
        outcome=fitted_logit_matched,
    )
    pred = est.predict(atexog={"treated": 1})
    # The prediction should be a single scalar
    assert pred.estimate.shape == () or pred.estimate.shape == (1,)


# ---------------------------------------------------------------------------
# Regression: matched predictions must differ from unmatched
# ---------------------------------------------------------------------------


def test_matched_predictions_differ_from_unmatched(df_matching, pysmatch_matcher):
    """Matched analysis on matched data should differ from analysis on full data."""
    # Fit model on FULL data, no matching
    full_model = smf.glm(
        "y ~ x1 + x2 + treated",
        data=df_matching,
        family=sm.families.Binomial(),
    ).fit()
    full_est = GComputation(full_model)
    pred_full = full_est.predict(atexog={"treated": 1})

    # Fit model on MATCHED data, with matching supplied through the wiring graph
    matched_df = pysmatch_matcher.matched_data
    matched_model = smf.glm(
        "y ~ x1 + x2 + treated",
        data=matched_df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": matched_df["match_id"]})
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    matched_est = GComputation(
        steps.match(steps.input(df_matching), client),
        outcome=matched_model,
    )
    pred_matched = matched_est.predict(atexog={"treated": 1})

    # Predictions should differ because the base data differs
    assert not np.allclose(pred_full.estimate, pred_matched.estimate, atol=1e-8), (
        "Matched and unmatched predictions should differ when the matched "
        "sample has a different covariate distribution than the full sample."
    )


# ---------------------------------------------------------------------------
# End-to-end: delta inference on matched data
# ---------------------------------------------------------------------------


def test_end_to_end_delta_contrast(
    fitted_logit_matched, pysmatch_matcher, df_matching
):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    est = GComputation(
        steps.match(steps.input(df_matching), client),
        outcome=fitted_logit_matched,
    )
    rd = est.contrasts(
        scenarios=[
            {"atexog": {"treated": 1}},
            {"atexog": {"treated": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(rd.estimate).all()
    assert np.isfinite(rd.std_error).all()
    assert np.isfinite(rd.conf_int_lower).all()
    assert np.isfinite(rd.conf_int_upper).all()


# ---------------------------------------------------------------------------
# End-to-end: bootstrap with rematching
# ---------------------------------------------------------------------------


def test_end_to_end_bootstrap_contrast(
    fitted_logit_matched, pysmatch_matcher, df_matching
):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    est = GComputation(
        steps.match(steps.input(df_matching), client),
        outcome=fitted_logit_matched,
        method="bootstrap",
        B=20,
        seed=42,
    )
    rd = est.contrasts(
        scenarios=[
            {"atexog": {"treated": 1}},
            {"atexog": {"treated": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(rd.estimate).all()
    assert np.isfinite(rd.std_error).all()
    assert np.isfinite(rd.conf_int_lower).all()
    assert np.isfinite(rd.conf_int_upper).all()
    # Bootstrap should have draws
    assert rd.draws is not None
    assert len(rd.draws) == 20
