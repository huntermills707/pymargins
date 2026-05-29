"""Tests for session-level bootstrap and simulation cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins
from pymargins._result import compose_results

# ---------------------------------------------------------------------------
# Phase 2: bootstrap refit cache
# ---------------------------------------------------------------------------


def test_bootstrap_two_calls_share_refits():
    """Two bootstrap calls on the same session must reuse the cached refits.
    We verify this by monkeypatching _harvest_bootstrap_states to count calls."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    from pymargins._inference import _bootstrap as _boot_mod

    orig_harvest = _boot_mod._harvest_bootstrap_states
    harvest_calls = [0]

    def counting_harvest(*args, **kwargs):
        harvest_calls[0] += 1
        return orig_harvest(*args, **kwargs)

    _boot_mod._harvest_bootstrap_states = counting_harvest
    try:
        m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123)
        r1 = m.predict(atexog={"x1": 0})
        r2 = m.predict(atexog={"x1": 1})

        assert harvest_calls[0] == 1, f"Expected 1 harvest call, got {harvest_calls[0]}"
        assert r1.resample_bank_id == r2.resample_bank_id
        assert r1.n_boot_effective == 50
        assert r1.n_boot_failed == 0
    finally:
        _boot_mod._harvest_bootstrap_states = orig_harvest


def test_bootstrap_cache_shares_failures():
    """A flaky refit should produce the same failure count across two calls."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    call_count = [0]
    orig_refit = fit.model.__class__.fit

    def counting_refit(self, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 7 == 0:
            raise ValueError("flaky refit")
        return orig_refit(self, *args, **kwargs)

    # Monkeypatch the model class fit method via the instance's model class
    fit.model.__class__.fit = counting_refit
    try:
        m = Margins(fit, method="bootstrap", n_boot=20, n_jobs=1, rng_seed=1)
        r1 = m.predict(atexog={"x": 0})
        r2 = m.predict(atexog={"x": 1})

        assert r1.n_boot_failed == r2.n_boot_failed
        assert r1.n_boot_effective == r2.n_boot_effective
        assert r1.n_boot_failed > 0
    finally:
        fit.model.__class__.fit = orig_refit


def test_bootstrap_cache_applies_to_contrasts_and_slopes():
    """predict, dydx, and contrasts all reuse the same cached refits."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=7)
    r_pred = m.predict()
    r_slope = m.dydx("x1")
    r_contrast = m.contrasts(
        scenarios=[{"atexog": {"x1": 0}}, {"atexog": {"x1": 1}}],
        contrasts=[1, -1],
    )

    # All should report the same effective and failed counts
    assert (
        r_pred.n_boot_effective
        == r_slope.n_boot_effective
        == r_contrast.n_boot_effective
    )
    assert r_pred.n_boot_failed == r_slope.n_boot_failed == r_contrast.n_boot_failed


# ---------------------------------------------------------------------------
# Phase 2b: simulation draws bank
# ---------------------------------------------------------------------------


def test_simulation_two_calls_share_draws():
    """Two simulation calls on the same session must reuse the cached β* draws.
    We verify this by inspecting the session cache directly."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
        }
    )
    eta = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"]
    prob = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = (rng.uniform(size=n) < prob).astype(float)
    fit = smf.glm("y ~ age + treatment", data=df, family=sm.families.Binomial()).fit()

    m = Margins(fit, method="simulation", n_sim=100, rng_seed=42)
    m.predict(atexog={"treatment": 0})

    assert hasattr(m, "_simulation_draws_cache"), "Cache should exist after first call"
    cached_key, cached_draws = m._simulation_draws_cache

    m.predict(atexog={"treatment": 1})
    cached_key2, cached_draws2 = m._simulation_draws_cache

    assert cached_key == cached_key2
    assert cached_draws is cached_draws2, "Same draw array should be reused"


# ---------------------------------------------------------------------------
# Phase 3: freeze session parameters post-cache
# ---------------------------------------------------------------------------


def test_post_cache_mutation_raises():
    """Mutating a frozen attribute after the cache is materialized raises."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="bootstrap", n_boot=20, n_jobs=1, rng_seed=1)
    _ = m.predict()  # materialize cache

    with pytest.raises(RuntimeError, match="Cannot mutate"):
        m.n_boot = 30


def test_pre_cache_mutation_allowed():
    """Mutating a frozen attribute before the cache is materialized succeeds."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="bootstrap", n_boot=20, n_jobs=1, rng_seed=1)
    # No cache yet
    m.n_boot = 25  # should succeed
    _ = m.predict()
    assert m.n_boot == 25


def test_adapter_drift_detected():
    """Changing adapter.coefficients() after cache materialization raises."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="bootstrap", n_boot=20, n_jobs=1, rng_seed=1)
    _ = m.predict()  # materialize cache

    # Mutate coefficients
    orig_params = fit.params.copy()
    fit.params.iloc[0] += 1.0
    try:
        with pytest.raises(RuntimeError, match="has changed"):
            _ = m.predict()
    finally:
        fit.params.iloc[0] = orig_params.iloc[0]


def test_delta_session_no_cache_no_freeze():
    """Delta-method sessions do not materialize inference caches and allow
    mutation of bootstrap/simulation parameters."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="delta")
    _ = m.predict()
    # n_boot is not in use for delta, and no bootstrap/simulation cache
    # was materialized, so mutation should succeed.
    m.n_boot = 30
    assert m.n_boot == 30


# ---------------------------------------------------------------------------
# Phase 1: bootstrap_state default identity
# ---------------------------------------------------------------------------


def test_adapter_bootstrap_state_default_is_self():
    """The default bootstrap_state() returns self."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    m = Margins(fit, method="bootstrap", n_boot=10, rng_seed=1)
    adapter = m.adapter
    assert adapter.bootstrap_state() is adapter


# ---------------------------------------------------------------------------
# Joint inference / composition across cached calls
# ---------------------------------------------------------------------------


def test_bootstrap_composition_across_cached_calls():
    """Results from two cached bootstrap calls compose via _check_draws_match."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=7)
    r0 = m.predict(atexog={"x": 0})
    r1 = m.predict(atexog={"x": 1})

    # Arithmetic composition
    diff = r1 - r0
    assert diff.draws_inf is not None
    assert np.allclose(diff.draws_inf, r1.draws_inf - r0.draws_inf)

    # Nonlinear composition via compose_results
    ratio = compose_results([r1, r0], fn=lambda p: p[0] / p[1], label="ratio")
    assert ratio.draws_inf is not None


# ---------------------------------------------------------------------------
# Matching rematch through the cache
# ---------------------------------------------------------------------------


class _StubMatcher:
    """Minimal matcher that deterministically drops rows on rematch."""

    def __init__(self, data):
        self.matched_data = data.reset_index(drop=True)
        self.cluster_ids = np.arange(len(self.matched_data))

    def rematch(self, resampled):
        r = resampled.reset_index(drop=True)
        return r[r.index % 7 != 0].reset_index(drop=True)


def test_matching_cache_reuses_refits():
    """Matching with rematch: two calls should share the same harvested states."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "t": rng.binomial(1, 0.5, size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x"] - 0.4 * df["t"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x + t", data=df).fit()

    from pymargins._inference import _bootstrap as _boot_mod

    orig_harvest = _boot_mod._harvest_bootstrap_states
    harvest_calls = [0]

    def counting_harvest(*args, **kwargs):
        harvest_calls[0] += 1
        return orig_harvest(*args, **kwargs)

    _boot_mod._harvest_bootstrap_states = counting_harvest
    try:
        m = Margins(
            fit,
            matching=_StubMatcher(df),
            method="bootstrap",
            n_boot=50,
            n_jobs=1,
            rng_seed=7,
            at="overall",
        )
        r1 = m.predict()
        r2 = m.dydx("x")

        assert harvest_calls[0] == 1, f"Expected 1 harvest call, got {harvest_calls[0]}"
        assert r1.resample_bank_id == r2.resample_bank_id
    finally:
        _boot_mod._harvest_bootstrap_states = orig_harvest


# ---------------------------------------------------------------------------
# Survival adapter bootstrap cache
# ---------------------------------------------------------------------------


def test_lifelines_cox_bootstrap_cache():
    """Lifelines CoxPH adapter should transparently use the bootstrap states cache."""
    pytest.importorskip("lifelines")
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
        }
    )
    df["duration"] = rng.exponential(10, size=n)
    df["event"] = rng.binomial(1, 0.8, size=n)

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    from pymargins._inference import _bootstrap as _boot_mod

    orig_harvest = _boot_mod._harvest_bootstrap_states
    harvest_calls = [0]

    def counting_harvest(*args, **kwargs):
        harvest_calls[0] += 1
        return orig_harvest(*args, **kwargs)

    _boot_mod._harvest_bootstrap_states = counting_harvest
    try:
        m = Margins(cph, data=df, method="bootstrap", n_boot=30, n_jobs=1, rng_seed=7)
        r1 = m.predict(atexog={"treatment": 0})
        r2 = m.predict(atexog={"treatment": 1})

        assert harvest_calls[0] == 1, f"Expected 1 harvest call, got {harvest_calls[0]}"
        assert r1.resample_bank_id == r2.resample_bank_id
        assert r1.n_boot_effective == r2.n_boot_effective
    finally:
        _boot_mod._harvest_bootstrap_states = orig_harvest


def test_survival_multi_time_scenario_curve():
    """Survival adapter accepts per-scenario prediction_time and produces a
    multi-time counterfactual curve in a single bootstrap pass."""
    pytest.importorskip("lifelines")
    from lifelines import CoxPHFitter

    from pymargins._adapters.lifelines_coxph_survival import (
        LifelinesCoxPHSurvivalAdapter,
    )

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
        }
    )
    lp = -0.02 * (df["age"] - 50) - 0.7 * df["treatment"]
    df["duration"] = rng.exponential(np.exp(-lp) * 8.0)
    df["event"] = (df["duration"] < 20).astype(int)
    df["duration"] = df["duration"].clip(upper=20)
    cph = CoxPHFitter().fit(df, duration_col="duration", event_col="event")

    adapter = LifelinesCoxPHSurvivalAdapter(cph, training_data=df)
    m = Margins(
        cph,
        adapter=adapter,
        at="overall",
        method="bootstrap",
        n_boot=20,
        n_jobs=1,
        rng_seed=0,
    )

    times = [2.0, 8.0, 16.0]
    scens = [
        {"atexog": {"treatment": 1}, "prediction_time": t, "label": f"trt=1,t={t}"}
        for t in times
    ] + [
        {"atexog": {"treatment": 0}, "prediction_time": t, "label": f"trt=0,t={t}"}
        for t in times
    ]
    W = np.eye(len(scens))
    curve = m.contrasts(scenarios=scens, contrasts=W)

    est = np.asarray(curve.estimate).ravel()
    assert est.shape == (len(scens),)
    # All survival probabilities should lie in (0, 1).
    assert np.all((est > 0) & (est < 1))
    # Survival decreases with time for each arm.
    for arm_start in (0, len(times)):
        arm = est[arm_start : arm_start + len(times)]
        assert np.all(np.diff(arm) <= 1e-9), f"Non-monotonic curve: {arm}"


def test_survival_prediction_time_rejected_by_non_time_adapter():
    """A scenario with prediction_time against a non-time-aware adapter
    should raise a clear error rather than silently ignore the key."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(scale=0.3, size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(fit, method="delta")
    with pytest.raises(ValueError, match="prediction_time"):
        m.contrasts(
            scenarios=[{"atexog": {"x": 0}, "prediction_time": 5.0}],
            contrasts=[1.0],
        )
