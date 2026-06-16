"""Tests for bootstrap JAX compilation cache improvements (CODE_AUDIT §5.1).

R7 port: legacy ``Margins`` comparisons against the removed ``engine="loop"``
option are dropped; the remaining tests exercise the same cache/kernel
behaviors through the ``GComputation`` noun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation
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

    from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
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
    kernel gradient path should be exercised."""
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
    est = GComputation(
        fit,
        method="bootstrap",
        B=50,
        n_jobs=1,
        seed=123,
        ci="studentized",
    )
    result = est.predict(atexog={"treatment": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)


# ---------------------------------------------------------------------------
# Offset JIT safety
# ---------------------------------------------------------------------------


def test_bootstrap_offset_jit_safety():
    """GLM predictions with offset must be JIT-safe through the bootstrap
    kernel."""
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
    est = GComputation(
        fit, method="bootstrap", B=50, n_jobs=1, seed=123, at="typical"
    )
    result = est.predict(atexog={"x": 0})
    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)

    # AME (data-coupled) with offset
    est_ame = GComputation(
        fit, method="bootstrap", B=50, n_jobs=1, seed=123, at="overall"
    )
    result_ame = est_ame.predict()
    assert result_ame.estimate is not None
    lower_ame, upper_ame = result_ame.conf_int()
    assert np.all(lower_ame <= upper_ame)


# ---------------------------------------------------------------------------
# κ at β̂ reuse
# ---------------------------------------------------------------------------


def test_bootstrap_kappa_reuses_jitted_kernel():
    """Bootstrap κ computation should use a jitted kernel and complete
    without error."""
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
        est = GComputation(
            fit,
            method="bootstrap",
            B=50,
            n_jobs=1,
            seed=123,
            at="typical",
        )
        result = est.predict(atexog={"x": 0})
        # Should have at least 1 JIT for the bootstrap kernel and 1 for kappa
        assert jit_count[0] >= 1, (
            f"Expected at least 1 JIT compilation, got {jit_count[0]}"
        )
        assert result.kappa is not None
        assert np.isfinite(result.kappa)
    finally:
        _bootstrap.jax.jit = orig_jit


# ---------------------------------------------------------------------------
# Studentized kernel path for slopes and contrasts
# ---------------------------------------------------------------------------


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
    est = GComputation(
        fit,
        method="bootstrap",
        B=50,
        n_jobs=1,
        seed=123,
        ci="studentized",
    )
    result = est.dydx("x1")

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
    est = GComputation(
        fit,
        method="bootstrap",
        B=50,
        n_jobs=1,
        seed=123,
        ci="studentized",
    )
    result = est.contrasts(
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
# Fast-path JIT count
# ---------------------------------------------------------------------------


def test_bootstrap_fast_path_jits_once():
    """The batched engine should compile the kernel a small constant number
    of times, not O(B)."""
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
        est = GComputation(
            fit, method="bootstrap", B=200, n_jobs=1, seed=123, at="typical"
        )
        result = est.predict(atexog={"x": 0})
        assert jit_count[0] <= 3, f"Expected ≤3 JIT compilations, got {jit_count[0]}"
        assert result.draws_inf.shape == (200,)
    finally:
        _bootstrap.jax.jit = orig_jit


# ---------------------------------------------------------------------------
# BCa jackknife with kernel path
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
    est = GComputation(
        fit,
        method="bootstrap",
        B=50,
        n_jobs=1,
        seed=123,
        ci="bca",
    )
    result = est.predict(atexog={"x": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)
