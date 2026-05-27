"""Tests for 2026-05-24 improvements."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import Margins, adjust, AdjustedResults
from pymargins._result import MarginsResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, size=n),
        "x2": rng.normal(0, 1, size=n),
    })
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
# 1. from_posterior
# ---------------------------------------------------------------------------

def test_from_posterior_basic(fit_ols):
    rng = np.random.default_rng(0)
    beta = fit_ols.params.values.astype(float)
    Sigma = fit_ols.cov_params().values.astype(float)
    draws = rng.multivariate_normal(beta, Sigma, size=500)
    m = Margins.from_posterior(fit_ols, draws)
    assert m.method == "simulation"
    assert m.n_sim == 500
    pred = m.predict(atexog={"x1": 0.0, "x2": 0.0})
    assert isinstance(pred, MarginsResult)


def test_from_posterior_point_estimate_override(fit_ols):
    rng = np.random.default_rng(1)
    beta = fit_ols.params.values.astype(float)
    Sigma = fit_ols.cov_params().values.astype(float)
    draws = rng.multivariate_normal(beta, Sigma, size=500)
    custom_pe = beta + 0.5
    m = Margins.from_posterior(fit_ols, draws, point_estimate=custom_pe)
    pred = m.predict(atexog={"x1": 0.0, "x2": 0.0})
    np.testing.assert_allclose(pred.estimate, custom_pe[0], rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. MarginsResult.contrast
# ---------------------------------------------------------------------------

def test_contrast_on_vector_result(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    assert len(r.estimate) == 2
    C = np.array([[1.0, -1.0]])
    diff = r.contrast(C, labels=["diff"])
    assert diff.estimate.shape == (1,)
    np.testing.assert_allclose(
        diff.estimate, r.estimate[0] - r.estimate[1], rtol=1e-5
    )
    assert diff.gradient is not None


def test_contrast_requires_delta(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean", method="simulation")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    with pytest.raises(ValueError, match="delta-method"):
        r.contrast(np.array([[1.0, -1.0]]))


# ---------------------------------------------------------------------------
# 3. MarginsResult.influence
# ---------------------------------------------------------------------------

def test_influence_bca_bootstrap(fit_ols):
    m = Margins.linear_scale(
        fit_ols,
        at="mean",
        method="bootstrap",
        n_boot=30,
        rng_seed=0,
        bootstrap_config={"ci_method": "bca"},
    )
    r = m.predict(atexog={"x1": 0.0})
    infl = r.influence()
    assert isinstance(infl, np.ndarray)
    n_obs = len(fit_ols.model.endog)
    assert infl.shape[0] == n_obs


def test_influence_requires_machinery(fit_logit):
    m = Margins.linear_scale(fit_logit, at="mean", method="delta")
    r = m.predict(atexog={"x1": 0.0})
    with pytest.raises(NotImplementedError):
        r.influence()


# ---------------------------------------------------------------------------
# 4. adjust
# ---------------------------------------------------------------------------

def test_adjust_single_result(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    r1 = m.predict(atexog={"x1": 0.0})
    r2 = m.predict(atexog={"x1": 1.0})
    adj = adjust([r1, r2], method="bonferroni")
    assert isinstance(adj, AdjustedResults)
    assert len(adj.p_raw) == 2
    assert len(adj.p_adj) == 2
    assert adj.method == "bonferroni"


def test_adjust_dict_results(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    results = {
        "a": m.predict(atexog={"x1": 0.0}),
        "b": m.predict(atexog={"x1": 1.0}),
    }
    adj = adjust(results, method="holm")
    assert len(adj.p_raw) == 2
    assert adj.results is results


def test_adjust_to_frame(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    r = m.predict(atexog={"x1": [0.0, 1.0]})
    adj = adjust(r, method="fdr_bh")
    frame = adj.to_frame()
    assert "p_raw" in frame.columns
    assert "p_adj" in frame.columns


# ---------------------------------------------------------------------------
# 5. Elasticity sugar
# ---------------------------------------------------------------------------

def test_eyex_basic(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    e = m.eyex("x1")
    assert isinstance(e, MarginsResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("eyex")


def test_eydx_basic(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    e = m.eydx("x1")
    assert isinstance(e, MarginsResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("eydx")


def test_dyex_basic(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    e = m.dyex("x1")
    assert isinstance(e, MarginsResult)
    assert e.estimand_metadata.get("labels", [""])[0].startswith("dyex")


def test_elasticity_matches_manual_composition(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    e = m.dyex("x1")
    slope = m.dydx("x1")
    x_bar = fit_ols.model.exog[:, 1].mean()
    np.testing.assert_allclose(e.estimate, slope.estimate * x_bar, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. to_disk / from_disk
# ---------------------------------------------------------------------------

def test_to_disk_from_disk_roundtrip(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    r = m.predict(atexog={"x1": 0.0})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.pkl"
        r.to_disk(path)
        loaded = MarginsResult.from_disk(path)
    np.testing.assert_allclose(loaded.estimate, r.estimate)
    np.testing.assert_allclose(loaded.std_error, r.std_error)
    assert loaded.gradient is None  # materialized


def test_from_disk_version_warning(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    r = m.predict(atexog={"x1": 0.0})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.pkl"
        r.to_disk(path)
        import pickle
        with open(path, "rb") as f:
            blob = pickle.load(f)
        blob["version"] = "0.0.0+old"
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        with pytest.warns(UserWarning, match="saved with pymargins"):
            loaded = MarginsResult.from_disk(path)
    np.testing.assert_allclose(loaded.estimate, r.estimate)


# ---------------------------------------------------------------------------
# 7. rmst
# ---------------------------------------------------------------------------

def test_rmst_requires_time_aware_adapter(fit_ols):
    m = Margins.linear_scale(fit_ols, at="mean")
    with pytest.raises(ValueError, match="time-aware"):
        m.rmst(horizon=10.0)


# Note: A full rmst() test would require a survival adapter and lifelines.
# The adapter-level survival tests already exercise survival predictions,
# so this smoke test covers the session-level validation path.
