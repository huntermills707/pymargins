"""Tests for drop_outliers and trim stages (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import Margins, drop_outliers, reimpute, trim

# ---------------------------------------------------------------------------
# drop_outliers
# ---------------------------------------------------------------------------


def test_drop_outliers_re_derives_per_replicate():
    """A recording drop rule confirms it is re-applied every replicate."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    calls = [0]

    def rule(frame):
        calls[0] += 1
        return frame["x"].abs() > 3

    m = Margins(
        fit,
        transforms=[drop_outliers(rule)],
        method="bootstrap",
        n_boot=30,
        n_jobs=1,
        rng_seed=1,
    )
    _ = m.predict()
    assert calls[0] == 30


def test_drop_outliers_valid_under_delta():
    """drop_outliers has requires_resampling=False, so delta is allowed."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[drop_outliers(lambda f: f["x"].abs() > 3)],
        method="delta",
    )
    r = m.predict()
    assert np.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# trim
# ---------------------------------------------------------------------------


def test_trim_re_derives_per_replicate():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    calls = [0]
    orig_trim = trim

    def _rule(lower, upper, columns):
        # Re-use trim but count calls by wrapping
        stage = orig_trim(lower=lower, upper=upper, columns=columns)
        orig_prep = stage.prepare_resample

        def _counted_prep(data):
            calls[0] += 1
            return orig_prep(data)

        stage.prepare_resample = _counted_prep
        stage.prepare = _counted_prep
        return stage

    m = Margins(
        fit,
        transforms=[_rule(lower=-2, upper=2, columns=["x"])],
        method="bootstrap",
        n_boot=30,
        n_jobs=1,
        rng_seed=1,
    )
    _ = m.predict()
    assert calls[0] == 30


def test_trim_valid_under_delta():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[trim(lower=-2, upper=2, columns=["x"])],
        method="delta",
    )
    r = m.predict()
    assert np.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# Ordering matters
# ---------------------------------------------------------------------------


def test_drop_then_reimpute_not_equal_reimpute_then_drop():
    """On a crafted case, [drop, reimpute] ≠ [reimpute, drop].

    drop NaN rows first → smaller dataset; reimpute first → NaN are filled,
    then drop has nothing to remove, yielding a larger dataset and a
    different estimate.
    """
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": 1.0 + 0.5 * x + rng.normal(scale=0.3, size=n),
        }
    )
    # Inject MAR missingness
    missing = rng.uniform(size=n) < 0.25
    df_nan = df.copy()
    df_nan.loc[missing, "x"] = np.nan
    df_init = df_nan.fillna(df_nan.mean())
    fit = smf.ols("y ~ x", data=df_init).fit()

    def drop_na(f):
        return pd.isna(f["x"])

    def mean_imp(frame):
        return frame.fillna(frame.mean())

    m1 = Margins(
        fit,
        transforms=[drop_outliers(drop_na), reimpute(mean_imp, incomplete=df_nan)],
        method="bootstrap",
        n_boot=20,
        n_jobs=1,
        rng_seed=1,
    )
    r1 = m1.predict()

    m2 = Margins(
        fit,
        transforms=[reimpute(mean_imp, incomplete=df_nan), drop_outliers(drop_na)],
        method="bootstrap",
        n_boot=20,
        n_jobs=1,
        rng_seed=1,
    )
    r2 = m2.predict()

    # Standard errors must differ because the retained row sets differ
    assert not np.isclose(r1.std_error, r2.std_error), (
        f"Ordering should matter: SE {r1.std_error} vs {r2.std_error}"
    )


# ---------------------------------------------------------------------------
# 3b: weighted aggregation + row-altering stage raises
# ---------------------------------------------------------------------------


def test_weighted_plus_row_altering_raises():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    with pytest.raises(ValueError, match="weights= is not compatible"):
        Margins(
            fit,
            weights=np.ones(n),
            transforms=[drop_outliers(lambda f: f["x"].abs() > 3)],
            method="bootstrap",
        )


def test_weighted_plus_non_row_altering_ok():
    """A non-row-altering stage with weights is fine."""
    from pymargins._transforms import IdentityStage

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
        weights=np.ones(n),
        transforms=[IdentityStage()],
        method="bootstrap",
        n_boot=10,
        rng_seed=1,
    )
    r = m.predict()
    assert np.isfinite(r.estimate)
