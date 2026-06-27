"""Tests for 2026-05-24 improvements."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import AdjustedResults, GComputation, GraphResult, adjust

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=n),
            "x2": rng.normal(0, 1, size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(0, 0.5, size=n)
    return df


@pytest.fixture
def fit_ols(df):
    return smf.ols("y ~ x1 + x2", data=df).fit()


@pytest.fixture
def fit_logit(df):
    df["y_bin"] = (df["y"] > df["y"].median()).astype(float)
    return smf.glm("y_bin ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()


# ---------------------------------------------------------------------------
# 1. GraphResult.contrast
# ---------------------------------------------------------------------------


def test_contrast_on_vector_result(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    assert len(r.estimate) == 2
    C = np.array([[1.0, -1.0]])
    diff = r.contrast(C, labels=["diff"])
    assert diff.estimate.shape == (1,)
    np.testing.assert_allclose(diff.estimate, r.estimate[0] - r.estimate[1], rtol=1e-5)
    assert diff.gradient is not None


def test_contrast_requires_delta(fit_ols):
    m = GComputation(fit_ols, at="mean", method="simulation")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    with pytest.raises(ValueError, match="delta-method"):
        r.contrast(np.array([[1.0, -1.0]]))


# ---------------------------------------------------------------------------
# 2. GraphResult.influence
# ---------------------------------------------------------------------------


def test_influence_bca_bootstrap(fit_ols):
    m = GComputation(
        fit_ols,
        at="mean",
        method="bootstrap",
        B=30,
        seed=0,
        ci="bca",
    )
    r = m.predict(atexog={"x1": 0.0})
    infl = r.influence()
    assert isinstance(infl, np.ndarray)
    n_obs = len(fit_ols.model.endog)
    assert infl.shape[0] == n_obs


def test_influence_requires_machinery(fit_ols):
    # Simulation results have no gradient and are not BCa -> unsupported.
    m = GComputation(fit_ols, at="mean", method="simulation")
    r = m.predict(atexog={"x1": 0.0})
    with pytest.raises(ValueError, match="Influence is not available"):
        r.influence()


def test_influence_delta_logit_works(fit_logit):
    m = GComputation(fit_logit, at="mean", method="delta")
    infl = m.dydx("x1").influence()
    n_obs = len(fit_logit.model.endog)
    assert infl.shape == (n_obs,)


@pytest.fixture
def df_small():
    rng = np.random.default_rng(7)
    n = 60
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["y"] = 1.0 + 0.5 * d["x1"] - 0.3 * d["x2"] + rng.normal(0, 0.5, size=n)
    return d


def test_influence_delta_matches_jackknife(df_small):
    fit = smf.ols("y ~ x1 + x2", data=df_small).fit()
    m_delta = GComputation(fit, at="mean", method="delta")
    infl_delta = m_delta.dydx("x1").influence()
    m_boot = GComputation(
        fit,
        at="mean",
        method="bootstrap",
        B=20,
        seed=0,
        ci="bca",
    )
    infl_jack = m_boot.dydx("x1").influence()
    assert infl_delta.shape == infl_jack.shape == (len(df_small),)
    # One-step (delta) vs exact LOO (jackknife) differ only by the leverage
    # factor 1/(1 - h_ii); they agree to first order.
    corr = np.corrcoef(infl_delta, infl_jack)[0, 1]
    assert corr > 0.99
    np.testing.assert_allclose(infl_delta, infl_jack, atol=2e-3, rtol=0.1)


def test_influence_reconstructs_robust_variance(df_small):
    fit = smf.ols("y ~ x1 + x2", data=df_small).fit()
    m = GComputation(fit, at="mean", method="delta")
    infl = m.dydx("x1").influence()
    # Sum of squared influence = HC0 sandwich variance of the estimand.
    m_hc0 = GComputation(fit, at="mean", method="delta", vcov="HC0")
    r_hc0 = m_hc0.dydx("x1")
    np.testing.assert_allclose(np.sum(infl**2), r_hc0.std_error**2, rtol=1e-4)


def test_influence_vector_estimand_shape(df_small):
    fit = smf.ols("y ~ x1 + x2", data=df_small).fit()
    m = GComputation(fit, at="mean", method="delta")
    r = m.predict(atexog={"x1": [0.0, 1.0, 2.0]})
    infl = r.influence()
    # GraphResult orients psi_h as (estimand_dim, n_obs); legacy MarginsResult
    # used (n_obs, estimand_dim). This is a category (c) semantic change.
    assert infl.shape == (3, len(df_small))


# ---------------------------------------------------------------------------
# 3. adjust
# ---------------------------------------------------------------------------


def test_adjust_single_result(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r1 = m.predict(atexog={"x1": 0.0})
    r2 = m.predict(atexog={"x1": 1.0})
    adj = adjust([r1, r2], method="bonferroni")
    assert isinstance(adj, AdjustedResults)
    assert len(adj.p_raw) == 2
    assert len(adj.p_adj) == 2
    assert adj.method == "bonferroni"


def test_adjust_dict_results(fit_ols):
    m = GComputation(fit_ols, at="mean")
    results = {
        "a": m.predict(atexog={"x1": 0.0}),
        "b": m.predict(atexog={"x1": 1.0}),
    }
    adj = adjust(results, method="holm")
    assert len(adj.p_raw) == 2
    assert adj.results is results


def test_adjust_to_frame(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    adj = adjust(r, method="fdr_bh")
    frame = adj.to_frame()
    assert "p_raw" in frame.columns
    assert "p_adj" in frame.columns


# ---------------------------------------------------------------------------
# 4. Elasticity sugar
# ---------------------------------------------------------------------------


def test_eyex_basic(fit_ols):
    m = GComputation(fit_ols, at="mean")
    e = m.eyex("x1")
    assert isinstance(e, GraphResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("eyex")


def test_eydx_basic(fit_ols):
    m = GComputation(fit_ols, at="mean")
    e = m.eydx("x1")
    assert isinstance(e, GraphResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("eydx")


def test_dyex_basic(fit_ols):
    m = GComputation(fit_ols, at="mean")
    e = m.dyex("x1")
    assert isinstance(e, GraphResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("dyex")


def test_elasticity_matches_manual_composition(fit_ols):
    m = GComputation(fit_ols, at="mean")
    e = m.dyex("x1")
    slope = m.dydx("x1")
    x_bar = fit_ols.model.exog[:, 1].mean()
    np.testing.assert_allclose(e.estimate, slope.estimate * x_bar, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. to_disk / from_disk
# ---------------------------------------------------------------------------


def test_to_disk_from_disk_roundtrip(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": 0.0})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.pkl"
        r.to_disk(path)
        loaded = GraphResult.from_disk(path)
    np.testing.assert_allclose(loaded.estimate, r.estimate)
    np.testing.assert_allclose(loaded.std_error, r.std_error)
    np.testing.assert_allclose(loaded.gradient, r.gradient)


# ---------------------------------------------------------------------------
# 6. rmst
# ---------------------------------------------------------------------------


def test_rmst_requires_time_aware_adapter(fit_ols):
    m = GComputation(fit_ols, at="mean")
    with pytest.raises(ValueError, match="time-aware"):
        m.rmst(horizon=10.0)


# ---------------------------------------------------------------------------
# 7. WTP
# ---------------------------------------------------------------------------


def test_wtp_basic(fit_ols):
    m = GComputation(fit_ols, at="mean")
    wtp = m.wtp("x1", "x2")
    assert isinstance(wtp, GraphResult)
    assert wtp.estimand_metadata.get("labels", [""])[0].startswith("WTP")
    # WTP = -(∂y/∂x1) / (∂y/∂x2)
    slope1 = m.dydx("x1")
    slope2 = m.dydx("x2")
    expected = -(slope1.estimate / slope2.estimate)
    np.testing.assert_allclose(wtp.estimate, expected, rtol=1e-5)


def test_wtp_honours_atexog(fit_ols):
    m = GComputation(fit_ols, at="mean")
    wtp = m.wtp("x1", "x2", atexog={"x1": 1.0})
    assert isinstance(wtp, GraphResult)


# ---------------------------------------------------------------------------
# 8. diff_matrix
# ---------------------------------------------------------------------------


def test_diff_matrix_reference():
    from pymargins import diff_matrix

    C = diff_matrix(3, kind="reference")
    assert C.shape == (2, 3)
    expected = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(C, expected)


def test_diff_matrix_pairwise():
    from pymargins import diff_matrix

    C = diff_matrix(3, kind="pairwise")
    assert C.shape == (3, 3)
    expected = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(C, expected)


def test_diff_matrix_invalid_kind():
    from pymargins import diff_matrix

    with pytest.raises(ValueError, match="kind must be"):
        diff_matrix(3, kind="invalid")


# ---------------------------------------------------------------------------
# 9. pairwise_contrasts
# ---------------------------------------------------------------------------


def test_pairwise_contrasts_basic(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0, 2.0]})
    pc = r.pairwise_contrasts()
    assert pc.estimate.shape == (3,)  # 3*(3-1)/2 = 3
    assert len(pc.estimand_metadata.get("labels", [])) == 3
    # Check the contrast matrix was applied correctly
    expected = [
        r.estimate[1] - r.estimate[0],
        r.estimate[2] - r.estimate[0],
        r.estimate[2] - r.estimate[1],
    ]
    np.testing.assert_allclose(pc.estimate, expected, rtol=1e-5)


def test_pairwise_contrasts_with_labels(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    pc = r.pairwise_contrasts(labels=["low", "high"])
    assert pc.estimand_metadata["labels"] == ["high - low"]


def test_pairwise_contrasts_requires_delta(fit_ols):
    m = GComputation(fit_ols, at="mean", method="simulation")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    with pytest.raises(ValueError, match="delta-method"):
        r.pairwise_contrasts()


def test_pairwise_contrasts_composes_with_adjust(fit_ols):
    m = GComputation(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0, 2.0]})
    pc = r.pairwise_contrasts()
    adj = adjust(pc, method="holm")
    assert len(adj.p_raw) == 3


# Note: A full rmst() test would require a survival adapter and lifelines.
# The adapter-level survival tests already exercise survival predictions,
# so this smoke test covers the session-level validation path.
