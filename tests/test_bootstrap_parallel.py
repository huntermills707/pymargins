"""Tests for bootstrap parallelization.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import Margins


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "y": rng.standard_normal(n),
    })
    df["y"] = 0.5 + 0.8 * df["x1"] - 0.4 * df["x2"] + rng.standard_normal(n)
    return df


@pytest.fixture
def fit(df):
    return smf.ols("y ~ x1 + x2", data=df).fit()


# ---------------------------------------------------------------------------
# Reproducibility: serial vs parallel gives identical results
# ---------------------------------------------------------------------------

def test_parallel_reproducible(fit):
    m1 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42, n_jobs=1)
    m2 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42, n_jobs=2)
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)
    np.testing.assert_allclose(res1.std_error, res2.std_error)


def test_parallel_n_jobs_minus_one(fit):
    m = Margins(fit, method="bootstrap", n_boot=50, rng_seed=42, n_jobs=-1)
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Parallel with cluster bootstrap
# ---------------------------------------------------------------------------

def test_parallel_cluster_bootstrap(fit, df):
    m1 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42,
                 cluster=df.index % 10, n_jobs=1)
    m2 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42,
                 cluster=df.index % 10, n_jobs=2)
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Parallel with block bootstrap
# ---------------------------------------------------------------------------

def test_parallel_block_bootstrap(fit):
    m1 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42,
                 block_size=10, n_jobs=1)
    m2 = Margins(fit, method="bootstrap", n_boot=100, rng_seed=42,
                 block_size=10, n_jobs=2)
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Parallel with predictions and contrasts
# ---------------------------------------------------------------------------

def test_parallel_predictions(fit):
    m = Margins(fit, method="bootstrap", n_boot=50, rng_seed=42, n_jobs=2)
    res = m.predict(atexog={"x1": [0, 1]})
    assert np.all(np.isfinite(res.estimate))
    assert np.all(np.isfinite(res.conf_int_lower))
    assert np.all(np.isfinite(res.conf_int_upper))


def test_parallel_contrasts(fit):
    m = Margins(fit, method="bootstrap", n_boot=50, rng_seed=42, n_jobs=2)
    res = m.contrasts(
        scenarios=[
            {"atexog": {"x1": 1}},
            {"atexog": {"x1": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.conf_int_lower)
    assert np.isfinite(res.conf_int_upper)
