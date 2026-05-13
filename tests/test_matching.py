"""Tests for matching support in Margins.

See notes/MATCHING_API_DESIGN.md for the design specification.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins
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
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "treated": rng.binomial(1, 0.35, n),
    })
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
# Protocol validation
# ---------------------------------------------------------------------------

def test_matching_missing_matched_data_raises():
    class BadMatcher:
        cluster_ids = np.array([1, 2])
        def rematch(self, data):
            return data

    model = smf.glm("y ~ x", data=pd.DataFrame({"y": [0, 1], "x": [0, 1]}),
                    family=sm.families.Binomial()).fit()
    with pytest.raises(ValueError, match="matched_data"):
        Margins(model, matching=BadMatcher())


def test_matching_missing_cluster_ids_raises():
    class BadMatcher:
        matched_data = pd.DataFrame({"x": [0, 1]})
        def rematch(self, data):
            return data

    model = smf.glm("y ~ x", data=pd.DataFrame({"y": [0, 1], "x": [0, 1]}),
                    family=sm.families.Binomial()).fit()
    with pytest.raises(ValueError, match="cluster_ids"):
        Margins(model, matching=BadMatcher())


def test_matching_length_mismatch_raises(df_matching):
    model = smf.glm("y ~ x1 + x2 + treated", data=df_matching,
                    family=sm.families.Binomial()).fit()
    class ShortMatcher:
        matched_data = df_matching.iloc[:50].copy()
        cluster_ids = np.arange(50)
        def rematch(self, data):
            return data

    with pytest.raises(ValueError, match="matched_data length"):
        Margins(model, matching=ShortMatcher())


def test_matching_cluster_ids_wrong_length_raises(fitted_logit_matched, pysmatch_matcher):
    class BadMatcher:
        matched_data = pysmatch_matcher.matched_data.copy()
        cluster_ids = np.arange(len(pysmatch_matcher.matched_data) + 1)
        def rematch(self, data):
            return data

    with pytest.raises(ValueError, match="cluster_ids length"):
        Margins(fitted_logit_matched, matching=BadMatcher())


def test_matching_bootstrap_without_rematch_raises(fitted_logit_matched, pysmatch_matcher):
    class NoRematch:
        matched_data = pysmatch_matcher.matched_data.copy()
        cluster_ids = pysmatch_matcher.matched_data["match_id"].values

    with pytest.raises(ValueError, match="rematch"):
        Margins(fitted_logit_matched, matching=NoRematch(), method="bootstrap")


# ---------------------------------------------------------------------------
# Base data filtering
# ---------------------------------------------------------------------------

def test_matching_base_data_is_matched_data(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m = Margins(fitted_logit_matched, matching=client)
    base = m._get_base_data()
    pd.testing.assert_frame_equal(base, client.matched_data)


def test_matching_at_overall_uses_matched_data(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m = Margins(fitted_logit_matched, matching=client)
    pred = m.predict(atexog={"treated": 1})
    # The prediction should be a single scalar
    assert pred.estimate.shape == () or pred.estimate.shape == (1,)


# ---------------------------------------------------------------------------
# Vcov auto-derivation and warnings
# ---------------------------------------------------------------------------

def test_matching_auto_derives_cluster_vcov(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m = Margins(fitted_logit_matched, matching=client)
    assert isinstance(m.vcov_spec, dict)
    assert m.vcov_spec.get("type") == "cluster"


def test_matching_warns_non_cluster_vcov(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    with pytest.warns(UserWarning, match="not cluster-robust"):
        m = Margins(fitted_logit_matched, matching=client, vcov="HC3")
    assert m.vcov_spec == "HC3"


def test_matching_no_warning_when_user_supplies_ndarray(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    Sigma = np.eye(len(fitted_logit_matched.params))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = Margins(fitted_logit_matched, matching=client, vcov=Sigma)
    assert m.vcov_spec is Sigma


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
    m_full = Margins(full_model)
    pred_full = m_full.predict(atexog={"treated": 1})

    # Fit model on MATCHED data, with matching
    matched_df = pysmatch_matcher.matched_data
    matched_model = smf.glm(
        "y ~ x1 + x2 + treated",
        data=matched_df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": matched_df["match_id"]})
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m_matched = Margins(matched_model, matching=client)
    pred_matched = m_matched.predict(atexog={"treated": 1})

    # Predictions should differ because the base data differs
    assert not np.isclose(pred_full.estimate, pred_matched.estimate, atol=1e-8), (
        "Matched and unmatched predictions should differ when the matched "
        "sample has a different covariate distribution than the full sample."
    )


# ---------------------------------------------------------------------------
# End-to-end: delta inference on matched data
# ---------------------------------------------------------------------------

def test_end_to_end_delta_contrast(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m = Margins(fitted_logit_matched, matching=client)
    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treated": 1}},
            {"atexog": {"treated": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(rd.estimate)
    assert np.isfinite(rd.std_error)
    assert np.isfinite(rd.conf_int_lower)
    assert np.isfinite(rd.conf_int_upper)


# ---------------------------------------------------------------------------
# End-to-end: bootstrap with rematching
# ---------------------------------------------------------------------------

def test_end_to_end_bootstrap_contrast(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    m = Margins(
        fitted_logit_matched,
        matching=client,
        method="bootstrap",
        n_boot=20,
        rng_seed=42,
    )
    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treated": 1}},
            {"atexog": {"treated": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(rd.estimate)
    assert np.isfinite(rd.std_error)
    assert np.isfinite(rd.conf_int_lower)
    assert np.isfinite(rd.conf_int_upper)
    # Bootstrap should have draws
    assert rd.draws is not None
    assert len(rd.draws) == 20


# ---------------------------------------------------------------------------
# Weights alignment with matched data
# ---------------------------------------------------------------------------

def test_matching_weights_length_validation(fitted_logit_matched, pysmatch_matcher):
    client = PysmatchClient(pysmatch_matcher, treatment_col="treated")
    bad_weights = np.ones(len(pysmatch_matcher.matched_data) + 5)
    with pytest.raises(ValueError, match="weights length"):
        Margins(fitted_logit_matched, matching=client, weights=bad_weights)
