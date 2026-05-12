"""Tests for bootstrap JAX compilation cache improvements (CODE_AUDIT §5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins
from pymargins._gradients import make_glm_jvp_wrapper, _glm_jvp_cache
from pymargins._estimands import (
    prediction_kernel,
    slope_kernel,
    linear_combination_kernel,
    evaluate_kernel,
)


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
    w_probit = make_glm_jvp_wrapper(sm.families.Binomial(link=sm.families.links.Probit()))
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
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.binomial(1, 0.5, size=n),
    })
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
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treatment": rng.binomial(1, 0.5, size=n),
    })
    eta = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"]
    prob = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = (rng.uniform(size=n) < prob).astype(float)

    fit = smf.glm("y ~ age + treatment", data=df, family=sm.families.Binomial()).fit()
    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123,
                bootstrap_config={"ci_method": "studentized"})
    result = m.predict(atexog={"treatment": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)


def test_bootstrap_studentized_slope_uses_kernel_path():
    """Studentized bootstrap on a slope estimand should exercise the kernel
    gradient path without errors."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)

    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123,
                bootstrap_config={"ci_method": "studentized"})
    result = m.dydx("x1")

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert lower <= upper


def test_bootstrap_studentized_contrasts_uses_kernel_path():
    """Studentized bootstrap on contrasts should exercise the kernel path."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)

    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123,
                bootstrap_config={"ci_method": "studentized"})
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
# BCa jackknife with kernel path (no grad, just evaluation)
# ---------------------------------------------------------------------------

def test_bootstrap_bca_glm_completes():
    """BCa bootstrap should complete with the new kernel-based estimands."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.binomial(1, 0.5, size=n),
    })
    fit = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    m = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=123,
                bootstrap_config={"ci_method": "bca"})
    result = m.predict(atexog={"x": [0, 1]})

    assert result.estimate is not None
    lower, upper = result.conf_int()
    assert np.all(lower <= upper)
