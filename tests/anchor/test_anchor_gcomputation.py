"""Anchor harness: GComputation vs Margins byte-identical reproduction (W2.7).

This test suite is the correctness gate for Phase 2. Every merge from Phase 2
on must keep this suite green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, steps
from pymargins.estimators import GComputation


def assert_anchored(a, b, name):
    """Exact-array anchor assertion with a localization diagnostic."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        diff = np.max(np.abs(a - b)) if a.shape == b.shape else "shape-mismatch"
        raise AssertionError(
            f"[anchor:{name}] max|a-b|={diff} dtypes=({a.dtype},{b.dtype}) "
            f"shapes=({a.shape},{b.shape})"
        )


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n),
            "y_cont": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treat": rng.binomial(1, 0.5, size=n),
            "cluster": np.repeat(np.arange(40), 5),
        }
    )


@pytest.fixture
def fit_glm(df):
    return smf.glm("y ~ treat + x1 + x2", data=df, family=sm.families.Binomial()).fit()


@pytest.fixture
def fit_ols(df):
    return smf.ols("y_cont ~ treat + x1 + x2", data=df).fit()


@pytest.fixture
def fit_probit(df):
    return smf.probit("y ~ treat + x1 + x2", data=df).fit()


@pytest.fixture
def fit_poisson(df):
    return smf.glm(
        "y ~ treat + x1 + x2", data=df, family=sm.families.Poisson()
    ).fit()


# ---------------------------------------------------------------------------
# Anchor matrix: models × methods × queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fit_name",
    ["fit_ols", "fit_glm"],
)
@pytest.mark.parametrize("method", ["delta", "simulation"])
@pytest.mark.parametrize("query", ["predict", "dydx"])
def test_anchor_model_method_query(df, fit_name, method, query, request):
    fit = request.getfixturevalue(fit_name)
    seed = 12345
    m = Margins(fit, at="overall", method=method, rng_seed=seed)
    est = GComputation(fit, at="overall", method=method, seed=seed)

    if query == "predict":
        r1 = m.predict()
        r2 = est.predict()
    else:
        r1 = m.dydx("x1")
        r2 = est.dydx("x1")

    assert_anchored(r1.estimate, r2.estimate, f"{fit_name}/{method}/{query}/estimate")
    assert_anchored(r1.std_error, r2.std_error, f"{fit_name}/{method}/{query}/se")
    assert_anchored(
        r1.conf_int_lower, r2.conf_int_lower, f"{fit_name}/{method}/{query}/ci_low"
    )
    assert_anchored(
        r1.conf_int_upper, r2.conf_int_upper, f"{fit_name}/{method}/{query}/ci_high"
    )

    if method == "simulation":
        assert_anchored(r1.draws, r2._result.draws, f"{fit_name}/{method}/{query}/draws")


@pytest.mark.parametrize(
    "fit_name",
    ["fit_ols", "fit_glm"],
)
@pytest.mark.parametrize("query", ["predict", "dydx"])
def test_anchor_bootstrap(df, fit_name, query, request):
    fit = request.getfixturevalue(fit_name)
    seed = 12345
    m = Margins(fit, at="overall", method="bootstrap", n_boot=200, rng_seed=seed)
    est = GComputation(fit, at="overall", method="bootstrap", B=200, seed=seed)

    if query == "predict":
        r1 = m.predict()
        r2 = est.predict()
    else:
        r1 = m.dydx("x1")
        r2 = est.dydx("x1")

    assert_anchored(r1.estimate, r2.estimate, f"{fit_name}/bootstrap/{query}/estimate")
    assert_anchored(r1.std_error, r2.std_error, f"{fit_name}/bootstrap/{query}/se")
    assert_anchored(
        r1.conf_int_lower, r2.conf_int_lower, f"{fit_name}/bootstrap/{query}/ci_low"
    )
    assert_anchored(
        r1.conf_int_upper, r2.conf_int_upper, f"{fit_name}/bootstrap/{query}/ci_high"
    )


# ---------------------------------------------------------------------------
# Posture variations
# ---------------------------------------------------------------------------


def test_anchor_contrasts_ols(df, fit_ols):
    seed = 12345
    m = Margins(fit_ols, at="overall", method="delta", rng_seed=seed)
    est = GComputation(fit_ols, at="overall", method="delta", seed=seed)
    scenarios = [{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}]
    r1 = m.contrasts(scenarios=scenarios, contrasts=[1, -1])
    r2 = est.contrasts(scenarios=scenarios, contrasts=[1, -1])
    assert_anchored(r1.estimate, r2.estimate, "ols/contrasts/estimate")
    assert_anchored(r1.std_error, r2.std_error, "ols/contrasts/se")


def test_anchor_evaluate_ols(df, fit_ols):
    seed = 12345
    m = Margins(fit_ols, at="overall", method="delta", rng_seed=seed)
    est = GComputation(fit_ols, at="overall", method="delta", seed=seed)
    scenarios = [{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}]
    r1 = m.evaluate(scenarios=scenarios, compose=lambda x: x[1] - x[0])
    r2 = est.evaluate(scenarios=scenarios, compose=lambda x: x[1] - x[0])
    assert_anchored(r1.estimate, r2.estimate, "ols/evaluate/estimate")
    assert_anchored(r1.std_error, r2.std_error, "ols/evaluate/se")


@pytest.mark.xfail(
    strict=True,
    reason="D4: legacy cluster= declaration normalizes Σ̂ differently from the explicit vcov={'type':'cluster'} spec used by steps.input(cluster=).",
)
def test_anchor_cluster_ols(df, fit_ols):
    seed = 12345
    m = Margins(
        fit_ols,
        at="overall",
        method="delta",
        cluster=df["cluster"].values,
        rng_seed=seed,
    )
    est = GComputation(
        steps.input(df, cluster=df["cluster"].values),
        outcome=fit_ols,
        at="overall",
        method="delta",
        seed=seed,
    )
    r1 = m.predict()
    r2 = est.predict()
    assert_anchored(r1.estimate, r2.estimate, "ols/cluster/estimate")
    assert_anchored(r1.std_error, r2.std_error, "ols/cluster/se")


def test_anchor_weights_ols(df, fit_ols):
    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 1.5, size=len(df))
    seed = 12345
    m = Margins(fit_ols, at="overall", method="delta", weights=w, rng_seed=seed)
    est = GComputation(fit_ols, at="overall", method="delta", weights=w, seed=seed)
    r1 = m.predict()
    r2 = est.predict()
    assert_anchored(r1.estimate, r2.estimate, "ols/weights/estimate")
    assert_anchored(r1.std_error, r2.std_error, "ols/weights/se")
