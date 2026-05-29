"""Tests for bootstrap JAX compilation cache improvements (CODE_AUDIT §5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins
from pymargins._gradients import make_glm_jvp_wrapper

# ---------------------------------------------------------------------------
# GLM JVP wrapper cache
# ---------------------------------------------------------------------------


def test_glm_jvp_cache_reuses_wrapper():
    """make_glm_jvp_wrapper should return the identical callable for the same
    family link, avoiding per-adapter primitive creation."""
    family = sm.families.Binomial()
    w1 = make_glm_jvp_wrapper(family)
    w2 = make_glm_jvp_wrapper(family)
    assert w1 is w2


def test_glm_jvp_cache_differentiates_links():
    """Different link families should produce distinct wrappers."""
    w_logit = make_glm_jvp_wrapper(sm.families.Binomial())
    w_probit = make_glm_jvp_wrapper(
        sm.families.Binomial(link=sm.families.links.Probit())
    )
    assert w_logit is not w_probit


# ---------------------------------------------------------------------------
# Kernel detection on partial objects
# ---------------------------------------------------------------------------


def test_estimand_factories_return_marked_partials():
    """The high-level factories must return partials whose .func carries the
    __pymargins_kernel__ sentinel so the bootstrap path can short-circuit
    recompilation."""
    from functools import partial

    from pymargins._estimands import make_prediction_estimand

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter

    adapter = StatsmodelsGLMAdapter(fit)
    X = adapter.design_matrix_from_df(df)

    h = make_prediction_estimand(adapter, X)
    assert isinstance(h, partial)
    assert getattr(h.func, "__pymargins_kernel__", False) is True


# ---------------------------------------------------------------------------
# Bootstrap studentized CI with kernel path
# ---------------------------------------------------------------------------


def test_bootstrap_studentized_glm_uses_kernel_path():
    """Studentized bootstrap on a GLM should complete successfully and the
    kernel gradient path should be exercised (detected via coverage or by
    inspecting that the estimand is a marked partial)."""
    rng = np.random.default_rng(42)
    n = 100
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
    m = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        bootstrap_config={"ci_method": "studentized"},
    )
    result = m.predict(atexog={"treatment": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)


# ---------------------------------------------------------------------------
# Data-coupled (AME) fast path: at="overall"
# ---------------------------------------------------------------------------


def test_bootstrap_batched_matches_loop_ame_prediction():
    """Batched bootstrap with at='overall' must match the legacy loop engine."""
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

    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="overall"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict()
    result_l = m_loop.predict()

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.estimate, result_l.estimate, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_ame_glm():
    """Batched bootstrap at='overall' on a GLM must match the loop engine."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="overall"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict()
    result_l = m_loop.predict()

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_ame_studentized():
    """Studentized batched bootstrap at='overall' must match the loop engine."""
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

    m_batched = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="overall",
        bootstrap_config={"ci_method": "studentized"},
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="overall",
        bootstrap_config={"ci_method": "studentized", "engine": "loop"},
    )

    result_b = m_batched.predict()
    result_l = m_loop.predict()

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Data-coupled (AME) fast path under matching/rematch
# ---------------------------------------------------------------------------


class _StubMatcher:
    """Minimal matching client (no pysmatch) whose rematch() returns a
    *strict subset* of the resample, so the post-rematch design differs
    from data.iloc[all_idx[b]].  This is the configuration that exposed the
    data-coupled fast path skipping rematch()."""

    def __init__(self, data):
        self.matched_data = data.reset_index(drop=True)
        self.cluster_ids = np.arange(len(self.matched_data))

    def rematch(self, resampled):
        r = resampled.reset_index(drop=True)
        # Deterministically drop ~1/7 of rows; both engines must build the
        # design from this rematched subset (what beta_b is fit on).
        return r[r.index % 7 != 0].reset_index(drop=True)


def test_bootstrap_batched_matches_loop_ame_matching():
    """AME (at='overall') under a rematching matcher: the batched engine
    must build X_b from the post-rematch training data exactly like the
    loop, so draws/SE are numerically identical for a fixed seed."""
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

    m_batched = Margins(
        fit,
        matching=_StubMatcher(df),
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=7,
        at="overall",
    )
    m_loop = Margins(
        fit,
        matching=_StubMatcher(df),
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=7,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict()
    result_l = m_loop.predict()

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.estimate, result_l.estimate, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_ame_matching_studentized():
    """Studentized AME under rematching must also match the loop engine."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n).astype(float),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    m_batched = Margins(
        fit,
        matching=_StubMatcher(df),
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=7,
        at="overall",
        bootstrap_config={"ci_method": "studentized"},
    )
    m_loop = Margins(
        fit,
        matching=_StubMatcher(df),
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=7,
        at="overall",
        bootstrap_config={"ci_method": "studentized", "engine": "loop"},
    )

    result_b = m_batched.predict()
    result_l = m_loop.predict()

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_ame_all_replicates_fail_falls_back():
    """If every replicate fails to refit, the pure-AAP batched branch must
    fall back to the loop (which raises the failure-threshold error) rather
    than crashing on an empty jnp.stack()."""
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = rng.normal(size=n)
    fit = smf.ols("y ~ x", data=df).fit()

    class _AlwaysFail:
        """Adapter wrapper whose refit always raises, via a matcher that
        drops every row so the refit has no data."""

        def __init__(self, data):
            self.matched_data = data.reset_index(drop=True)
            self.cluster_ids = np.arange(len(self.matched_data))

        def rematch(self, resampled):
            return resampled.iloc[0:0]  # empty -> refit fails every time

    m = Margins(
        fit,
        matching=_AlwaysFail(df),
        method="bootstrap",
        n_boot=20,
        n_jobs=1,
        rng_seed=1,
        at="overall",
    )
    # The loop fallback raises the standard all-failed error; the key
    # assertion is that we do NOT get a bare "need at least one array to
    # stack" ValueError from the fast path.
    with pytest.raises(Exception) as excinfo:
        m.predict()
    assert "stack" not in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Generalized data-coupled coverage (audit MEDIUM #1): the batched engine
# must now cover predict(atexog) and contrasts at at="overall", and
# at="typical" with a free covariate — all previously loop-only — and stay
# numerically identical to the loop.
# ---------------------------------------------------------------------------


def _jit_counter():
    from pymargins._inference import _bootstrap

    state = {"n": 0, "orig": _bootstrap.jax.jit}

    def counting(*a, **k):
        state["n"] += 1
        return state["orig"](*a, **k)

    return _bootstrap, state, counting


@pytest.mark.parametrize(
    "at,kwargs,call",
    [
        ("overall", {}, dict(atexog={"x1": 1})),  # AME + atexog
        ("typical", {}, dict(atexog={"x1": 0})),  # typical, x2 free
    ],
)
def test_bootstrap_batched_matches_loop_predict_generalized(at, kwargs, call):
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    mod, state, counting = _jit_counter()
    mod.jax.jit = counting
    try:
        m_b = Margins(
            fit, method="bootstrap", n_boot=60, n_jobs=1, rng_seed=9, at=at, **kwargs
        )
        result_b = m_b.predict(**call)
        jits = state["n"]
    finally:
        mod.jax.jit = state["orig"]

    m_l = Margins(
        fit,
        method="bootstrap",
        n_boot=60,
        n_jobs=1,
        rng_seed=9,
        at=at,
        bootstrap_config={"engine": "loop"},
        **kwargs,
    )
    result_l = m_l.predict(**call)

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)
    # Proves the batched path was actually exercised (not a silent loop
    # fallback): a small constant number of compilations, not O(n_boot).
    assert 1 <= jits <= 3, f"expected the batched path (1-3 jits), got {jits}"


def test_bootstrap_batched_matches_loop_contrasts_ame():
    """Data-coupled (at='overall') contrasts go through linear_combination_kernel
    and must match the loop."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    sc = [{"atexog": {"x1": 0}}, {"atexog": {"x1": 1}}]

    m_b = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=9, at="overall"
    )
    m_l = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=9,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )
    rb = m_b.contrasts(scenarios=sc, contrasts=[1, -1])
    rl = m_l.contrasts(scenarios=sc, contrasts=[1, -1])
    assert np.allclose(rb.draws_inf, rl.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(rb.std_error, rl.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_slope_pinned_is_exact():
    """Data-INDEPENDENT slope (Xp/Xm byte-stable) is batched and must be
    numerically identical to the loop."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    m_b = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=9, at="typical"
    )
    m_l = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=9,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )
    rb = m_b.dydx("x1", atexog={"x1": 0.0, "x2": 0.0})
    rl = m_l.dydx("x1", atexog={"x1": 0.0, "x2": 0.0})
    assert np.allclose(rb.draws_inf, rl.draws_inf, rtol=1e-5, atol=1e-6)


def test_bootstrap_data_coupled_slope_routes_to_loop():
    """Data-coupled slope has an internal ~1e-6 finite difference that is
    catastrophically cancelling in float32; the batched engine must decline
    it and produce the loop result (not a >1% divergence)."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    m_b = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=9, at="overall"
    )
    m_l = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=9,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )
    rb = m_b.dydx("x1")
    rl = m_l.dydx("x1")
    assert np.allclose(rb.draws_inf, rl.draws_inf, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Robust data-independence detection (audit MEDIUM #2): a discrete free
# covariate whose mode can coincide for some resamples must not be
# misclassified as data-independent; batched must equal loop regardless.
# ---------------------------------------------------------------------------


def test_bootstrap_robust_probe_discrete_free_covariate():
    rng = np.random.default_rng(7)
    n = 150
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.integers(
                0, 3, size=n
            ),  # discrete; mode may repeat across resamples
        }
    )
    df["y"] = 0.5 + 0.4 * df["x"] - 0.2 * df["g"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x + C(g)", data=df).fit()

    # at="typical" pins x and g to representative values of each *resample*;
    # g's mode is data-dependent -> must be classified data-coupled.
    m_b = Margins(
        fit, method="bootstrap", n_boot=60, n_jobs=1, rng_seed=3, at="typical"
    )
    m_l = Margins(
        fit,
        method="bootstrap",
        n_boot=60,
        n_jobs=1,
        rng_seed=3,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )
    rb = m_b.predict(atexog={"x": 0})
    rl = m_l.predict(atexog={"x": 0})
    assert np.allclose(rb.draws_inf, rl.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(rb.std_error, rl.std_error, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Offset JIT safety (Wrinkle 2)
# ---------------------------------------------------------------------------


def test_bootstrap_offset_jit_safety():
    """GLM predictions with offset must be JIT-safe and numerically identical
    between batched and loop engines."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
        }
    )
    eta = -2.0 + 0.5 * df["x"]
    df["offset"] = np.log(rng.uniform(0.5, 2.0, size=n))
    prob = 1.0 / (1.0 + np.exp(-(eta + df["offset"])))
    df["y"] = (rng.uniform(size=n) < prob).astype(float)

    fit = smf.glm(
        "y ~ x", data=df, family=sm.families.Binomial(), offset=df["offset"]
    ).fit()

    # Fixed-X (data-independent) with offset
    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="typical"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict(atexog={"x": 0})
    result_l = m_loop.predict(atexog={"x": 0})
    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)

    # AME (data-coupled) with offset
    m_batched_ame = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="overall"
    )
    m_loop_ame = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="overall",
        bootstrap_config={"engine": "loop"},
    )

    result_b_ame = m_batched_ame.predict()
    result_l_ame = m_loop_ame.predict()
    assert np.allclose(
        result_b_ame.draws_inf, result_l_ame.draws_inf, rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# κ at β̂ reuse (Section 3.3)
# ---------------------------------------------------------------------------


def test_bootstrap_kappa_reuses_jitted_kernel():
    """When diagnostics=True, κ computation should use a jitted kernel and
    complete without error."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    from pymargins._inference import _bootstrap

    jit_count = [0]
    orig_jit = _bootstrap.jax.jit

    def counting_jit(*args, **kwargs):
        jit_count[0] += 1
        return orig_jit(*args, **kwargs)

    _bootstrap.jax.jit = counting_jit

    try:
        m = Margins(
            fit,
            method="bootstrap",
            n_boot=50,
            n_jobs=1,
            rng_seed=123,
            at="typical",
            diagnostics=True,
        )
        result = m.predict(atexog={"x": 0})
        # Should have at least 1 JIT for the bootstrap kernel and 1 for kappa
        assert jit_count[0] >= 1, (
            f"Expected at least 1 JIT compilation, got {jit_count[0]}"
        )
        assert result.kappa is not None
        assert np.isfinite(result.kappa)
    finally:
        _bootstrap.jax.jit = orig_jit


def test_bootstrap_studentized_slope_uses_kernel_path():
    """Studentized bootstrap on a slope estimand should exercise the kernel
    gradient path without errors."""
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
    m = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        bootstrap_config={"ci_method": "studentized"},
    )
    result = m.dydx("x1")

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert lower <= upper


def test_bootstrap_studentized_contrasts_uses_kernel_path():
    """Studentized bootstrap on contrasts should exercise the kernel path."""
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
    m = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        bootstrap_config={"ci_method": "studentized"},
    )
    result = m.contrasts(
        scenarios=[
            {"atexog": {"x1": 0}},
            {"atexog": {"x1": 1}},
        ],
        contrasts=[1, -1],
    )

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert lower <= upper


# ---------------------------------------------------------------------------
# Fast-path numerical identity vs legacy loop
# ---------------------------------------------------------------------------


def test_bootstrap_batched_matches_loop_glm_prediction():
    """Batched bootstrap engine must produce numerically identical results
    to the legacy loop engine for a fixed rng_seed."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    # Use at="typical" so the estimand is single-atom and data-independent,
    # which triggers the batched fast path.
    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="typical"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict(atexog={"x": 0})
    result_l = m_loop.predict(atexog={"x": 0})

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.estimate, result_l.estimate, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_ols_prediction():
    """Batched bootstrap on an OLS (LinearPredictionAdapter) prediction."""
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

    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="typical"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.predict(atexog={"x1": 0})
    result_l = m_loop.predict(atexog={"x1": 0})

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_slope():
    """Batched bootstrap on a slope estimand must match the loop engine."""
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

    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="typical"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.dydx("x1")
    result_l = m_loop.dydx("x1")

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)


def test_bootstrap_batched_matches_loop_studentized():
    """Batched studentized bootstrap must match the loop engine."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    m_batched = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"ci_method": "studentized"},
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"ci_method": "studentized", "engine": "loop"},
    )

    result_b = m_batched.predict(atexog={"x": 0})
    result_l = m_loop.predict(atexog={"x": 0})

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)
    assert np.allclose(result_b.std_error, result_l.std_error, rtol=1e-5, atol=1e-6)


def test_bootstrap_fast_path_jits_once():
    """The batched engine should compile the kernel a small constant number
    of times, not O(n_boot)."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()

    from pymargins._inference import _bootstrap

    jit_count = [0]
    orig_jit = _bootstrap.jax.jit

    def counting_jit(*args, **kwargs):
        jit_count[0] += 1
        return orig_jit(*args, **kwargs)

    _bootstrap.jax.jit = counting_jit

    try:
        m = Margins(
            fit, method="bootstrap", n_boot=200, n_jobs=1, rng_seed=123, at="typical"
        )
        result = m.predict(atexog={"x": 0})
        assert jit_count[0] <= 3, f"Expected ≤3 JIT compilations, got {jit_count[0]}"
        assert result.draws_inf.shape == (200,)
    finally:
        _bootstrap.jax.jit = orig_jit


def test_bootstrap_batched_matches_loop_contrasts():
    """Batched bootstrap on contrasts must match the loop engine."""
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

    m_batched = Margins(
        fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123, at="typical"
    )
    m_loop = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        at="typical",
        bootstrap_config={"engine": "loop"},
    )

    result_b = m_batched.contrasts(
        scenarios=[
            {"atexog": {"x1": 0}},
            {"atexog": {"x1": 1}},
        ],
        contrasts=[1, -1],
    )
    result_l = m_loop.contrasts(
        scenarios=[
            {"atexog": {"x1": 0}},
            {"atexog": {"x1": 1}},
        ],
        contrasts=[1, -1],
    )

    assert np.allclose(result_b.draws_inf, result_l.draws_inf, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# BCa jackknife with kernel path (no grad, just evaluation)
# ---------------------------------------------------------------------------


def test_bootstrap_bca_glm_completes():
    """BCa bootstrap should complete with the new kernel-based estimands."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.binomial(1, 0.5, size=n),
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    m = Margins(
        fit,
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=123,
        bootstrap_config={"ci_method": "bca"},
    )
    result = m.predict(atexog={"x": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)
