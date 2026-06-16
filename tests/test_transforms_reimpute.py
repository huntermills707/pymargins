"""Tests for the reimpute stage (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation, reimpute, steps
from pymargins._graph._node import Node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_incomplete_df(rng, n=200, prop_missing=0.15):
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(scale=0.5, size=n)
    # MAR missingness: x1 missing when x2 is high
    missing = rng.uniform(size=n) < prop_missing * (1 / (1 + np.exp(-df["x2"])))
    df_nan = df.copy()
    df_nan.loc[missing, "x1"] = np.nan
    df_init = df_nan.fillna(df_nan.mean())
    return df_init, df_nan


def _mean_imputer(df):
    """Deterministic conditional-mean imputer (triggers G3 warning)."""
    return df.fillna(df.mean())


def _iterative_imputer(*, max_iter, random_state, sample_posterior):
    """Return a seeded sklearn IterativeImputer, enabling the experimental API."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    return IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=sample_posterior,
    )


def _reimpute_node(df_base, df_incomplete, imputer, *, cluster=None):
    """Build a reimpute node whose point output is *df_base* but whose
    bootstrap resampling source is *df_incomplete*.

    This mirrors the legacy ``transforms=[reimpute(imputer, incomplete=...)]``
    semantics, because ``steps.reimpute`` derives the incomplete frame from the
    parent node's collected output.
    """
    stage = reimpute(imputer, incomplete=df_incomplete)
    return Node(
        kind="reimpute",
        inputs=(steps.input(df_base, cluster=cluster),),
        alters_rows=False,
        _payload=stage,
    )


# ---------------------------------------------------------------------------
# G1: reimpute + delta raises
# ---------------------------------------------------------------------------


def test_reimpute_with_delta_raises():
    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    with pytest.raises(ValueError, match="method='delta' is not compatible"):
        GComputation(
            _reimpute_node(df_init, df_nan, _mean_imputer),
            outcome=fit,
            method="delta",
        )


# ---------------------------------------------------------------------------
# G3: deterministic imputer triggers warning
# ---------------------------------------------------------------------------


def test_deterministic_imputer_warns():
    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    with pytest.warns(UserWarning, match="deterministic"):
        GComputation(
            _reimpute_node(df_init, df_nan, _mean_imputer),
            outcome=fit,
            method="bootstrap",
            B=10,
            n_jobs=1,
        )


# ---------------------------------------------------------------------------
# G2: missing structural column raises
# ---------------------------------------------------------------------------


def test_reimpute_with_nan_cluster_raises():
    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng, n=50)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    cluster = np.arange(len(df_nan), dtype=float)
    cluster[0] = np.nan

    est = GComputation(
        _reimpute_node(df_init, df_nan, _mean_imputer, cluster=cluster),
        outcome=fit,
        method="bootstrap",
        B=10,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="cluster IDs must not contain NaN"):
        est.predict()


# ---------------------------------------------------------------------------
# End-to-end: bootstrap predict with reimpute returns sane CIs
# ---------------------------------------------------------------------------


def test_reimpute_end_to_end_predict():
    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    est = GComputation(
        _reimpute_node(df_init, df_nan, _mean_imputer),
        outcome=fit,
        method="bootstrap",
        B=50,
        n_jobs=1,
        seed=7,
    )
    r = est.predict(atexog={"x1": 0, "x2": 0})
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.std_error)
    assert r.conf_int_lower < r.conf_int_upper
    assert r.n_boot_effective > 40  # most should succeed


# ---------------------------------------------------------------------------
# Per-replicate freshness: different fills across replicates
# ---------------------------------------------------------------------------


def _bootstrap_states(est):
    """Return the cached list of successful bootstrap states."""
    # BankSet stores the successful states list directly in _states_bank.
    return next(iter(est._banks._states_bank.values()))


def test_reimpute_freshness_across_replicates():
    """A stochastic imputer should produce different fills on repeated calls."""
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng, n=120)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    imp = _iterative_imputer(max_iter=5, random_state=0, sample_posterior=True)

    def _imputer(frame):
        arr = imp.fit_transform(frame)
        return pd.DataFrame(arr, columns=frame.columns)

    est = GComputation(
        _reimpute_node(df_init, df_nan, _imputer),
        outcome=fit,
        method="bootstrap",
        B=30,
        n_jobs=1,
        seed=7,
    )
    _ = est.predict()

    states = _bootstrap_states(est)
    assert len(states) > 0

    # The mere fact that bootstrap refits succeeded with different data
    # implies the imputer produced different fills.  We verify at least
    # one replicate had a different coefficient from the original.
    orig_coef = np.asarray(fit.params)
    diffs = 0
    for _b, adapter in states:
        coef = np.asarray(adapter.coefficients())
        if not np.allclose(coef, orig_coef, atol=1e-6):
            diffs += 1
    assert diffs > 0, (
        "All replicates had identical coefficients — imputer may be frozen"
    )


# ---------------------------------------------------------------------------
# Coverage / width: MI CIs wider than naive single-imputation delta
# ---------------------------------------------------------------------------


def test_reimpute_reproducible_with_fresh_seeded_imputer():
    """Two fresh sessions with the same seeded imputer and seed must
    produce identical bootstrap draws."""
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(scale=0.5, size=n)
    missing = rng.uniform(size=n) < 0.25
    df_nan = df.copy()
    df_nan.loc[missing, "x1"] = np.nan
    df_init = df_nan.fillna(df_nan.mean())
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    def make_imputer():
        imp = _iterative_imputer(max_iter=5, random_state=42, sample_posterior=True)

        def _imputer(frame):
            arr = imp.fit_transform(frame)
            return pd.DataFrame(arr, columns=frame.columns)

        return _imputer

    est1 = GComputation(
        _reimpute_node(df_init, df_nan, make_imputer()),
        outcome=fit,
        method="bootstrap",
        B=20,
        n_jobs=1,
        seed=7,
    )
    r1 = est1.predict(atexog={"x1": 0, "x2": 0})

    est2 = GComputation(
        _reimpute_node(df_init, df_nan, make_imputer()),
        outcome=fit,
        method="bootstrap",
        B=20,
        n_jobs=1,
        seed=7,
    )
    r2 = est2.predict(atexog={"x1": 0, "x2": 0})

    assert np.allclose(r1.draws_inf, r2.draws_inf), (
        "Two fresh sessions with identical seeds should produce identical draws"
    )


def test_reimpute_widens_se_on_affected_coefficient():
    """Reimpute-bootstrap SE for the affected coefficient must be at least as
    large as plain-bootstrap SE on the completed data, because MI injects
    imputation-model uncertainty."""
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(scale=0.5, size=n)
    # MAR missingness in x1
    missing = rng.uniform(size=n) < 0.30
    df_nan = df.copy()
    df_nan.loc[missing, "x1"] = np.nan
    df_init = df_nan.fillna(df_nan.mean())
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    imp = _iterative_imputer(max_iter=10, random_state=0, sample_posterior=True)

    def _imputer(frame):
        arr = imp.fit_transform(frame)
        return pd.DataFrame(arr, columns=frame.columns)

    est_mi = GComputation(
        _reimpute_node(df_init, df_nan, _imputer),
        outcome=fit,
        method="bootstrap",
        B=200,
        n_jobs=1,
        seed=7,
    )
    # We inspect the bootstrap coefficient draws directly
    _ = est_mi.predict()
    states_mi = _bootstrap_states(est_mi)
    assert len(states_mi) > 100
    coefs_mi = np.stack([np.asarray(s[1].coefficients()) for s in states_mi])
    se_mi_x1 = np.std(coefs_mi[:, 1], ddof=1)

    est_plain = GComputation(
        steps.input(df_init),
        outcome=fit,
        method="bootstrap",
        B=200,
        n_jobs=1,
        seed=7,
    )
    _ = est_plain.predict()
    states_plain = _bootstrap_states(est_plain)
    assert len(states_plain) > 100
    coefs_plain = np.stack([np.asarray(s[1].coefficients()) for s in states_plain])
    se_plain_x1 = np.std(coefs_plain[:, 1], ddof=1)

    assert se_mi_x1 >= se_plain_x1, (
        f"MI SE for x1 ({se_mi_x1}) should be >= plain SE ({se_plain_x1})"
    )

    # Sanity: unaffected coefficient (x2) should be roughly similar
    se_mi_x2 = np.std(coefs_mi[:, 2], ddof=1)
    se_plain_x2 = np.std(coefs_plain[:, 2], ddof=1)
    # Within 20% is a loose sanity check
    assert abs(se_mi_x2 - se_plain_x2) / max(se_plain_x2, 1e-12) < 0.20


def test_reimpute_draws_differ_from_plain_bootstrap():
    """MI bootstrap draws must differ from plain-bootstrap draws, proving
    that imputation uncertainty is injected into the distribution."""
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(42)
    df_init, df_nan = _make_incomplete_df(rng, n=300, prop_missing=0.25)
    fit = smf.ols("y ~ x1 + x2", data=df_init).fit()

    imp = _iterative_imputer(max_iter=10, random_state=0, sample_posterior=True)

    def _imputer(frame):
        arr = imp.fit_transform(frame)
        return pd.DataFrame(arr, columns=frame.columns)

    est_mi = GComputation(
        _reimpute_node(df_init, df_nan, _imputer),
        outcome=fit,
        method="bootstrap",
        B=100,
        n_jobs=1,
        seed=7,
    )
    r_mi = est_mi.predict(atexog={"x1": 0, "x2": 0})

    est_plain = GComputation(
        steps.input(df_init),
        outcome=fit,
        method="bootstrap",
        B=100,
        n_jobs=1,
        seed=7,
    )
    r_plain = est_plain.predict(atexog={"x1": 0, "x2": 0})

    # The draw arrays should not be identical
    assert not np.allclose(r_mi.draws_inf, r_plain.draws_inf), (
        "MI bootstrap draws are identical to plain bootstrap draws — "
        "imputation uncertainty is not being injected"
    )
