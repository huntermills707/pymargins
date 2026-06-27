"""Tests for drop_outliers and trim stages (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymargins import GComputation, steps

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

    def rule(frame):
        return frame["x"].abs() > 3

    node = steps.drop_outliers(steps.input(df), rule)
    stage = node._payload
    prep_calls = [0]
    orig_prep_resample = stage.prepare_resample

    def _counted_prep_resample(data):
        prep_calls[0] += 1
        return orig_prep_resample(data)

    stage.prepare_resample = _counted_prep_resample

    df_prepared = node.collect()
    fit = smf.ols("y ~ x", data=df_prepared).fit()

    est = GComputation(
        node,
        outcome=fit,
        method="bootstrap",
        B=30,
        n_jobs=1,
        seed=1,
    )
    _ = est.predict()
    assert prep_calls[0] == 30


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

    node = steps.drop_outliers(steps.input(df), lambda f: f["x"].abs() > 3)
    df_prepared = node.collect()
    fit = smf.ols("y ~ x", data=df_prepared).fit()

    est = GComputation(
        node,
        outcome=fit,
        method="delta",
    )
    r = est.predict()
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

    node = steps.trim(steps.input(df), lower=-2, upper=2, columns=["x"])
    stage = node._payload
    prep_calls = [0]
    orig_prep_resample = stage.prepare_resample

    def _counted_prep_resample(data):
        prep_calls[0] += 1
        return orig_prep_resample(data)

    stage.prepare_resample = _counted_prep_resample

    df_prepared = node.collect()
    fit = smf.ols("y ~ x", data=df_prepared).fit()

    est = GComputation(
        node,
        outcome=fit,
        method="bootstrap",
        B=30,
        n_jobs=1,
        seed=1,
    )
    _ = est.predict()
    assert prep_calls[0] == 30


def test_trim_valid_under_delta():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )

    node = steps.trim(steps.input(df), lower=-2, upper=2, columns=["x"])
    df_prepared = node.collect()
    fit = smf.ols("y ~ x", data=df_prepared).fit()

    est = GComputation(
        node,
        outcome=fit,
        method="delta",
    )
    r = est.predict()
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

    def drop_na(f):
        return pd.isna(f["x"])

    def mean_imp(frame):
        return frame.fillna(frame.mean())

    base = steps.input(df_nan)
    node1 = steps.reimpute(steps.drop_outliers(base, drop_na), mean_imp)
    node2 = steps.drop_outliers(steps.reimpute(base, mean_imp), drop_na)

    # The new engine requires the model training data to match the wiring
    # output (template/wiring fingerprint check), so fit a separate model on
    # each prepared dataset.
    fit1 = smf.ols("y ~ x", data=node1.collect()).fit()
    fit2 = smf.ols("y ~ x", data=node2.collect()).fit()

    est1 = GComputation(
        node1,
        outcome=fit1,
        method="bootstrap",
        B=20,
        n_jobs=1,
        seed=1,
    )
    r1 = est1.predict()

    est2 = GComputation(
        node2,
        outcome=fit2,
        method="bootstrap",
        B=20,
        n_jobs=1,
        seed=1,
    )
    r2 = est2.predict()

    # Standard errors must differ because the retained row sets differ
    assert not np.isclose(r1.std_error, r2.std_error), (
        f"Ordering should matter: SE {r1.std_error} vs {r2.std_error}"
    )


# ---------------------------------------------------------------------------
# 3b: weighted aggregation + row-altering stage raises
# ---------------------------------------------------------------------------


def test_weighted_plus_non_row_altering_ok():
    """A non-row-altering wiring with weights is fine."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    est = GComputation(
        steps.input(df),
        outcome=fit,
        weights=np.ones(n),
        method="bootstrap",
        B=10,
        seed=1,
    )
    r = est.predict()
    assert np.isfinite(r.estimate)
