"""Tests for the compile pipeline (W2.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins._graph._compile import CompileError, compile
from pymargins._graph._node import Node


def df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, 100),
            "x": rng.normal(size=100),
        }
    )


def test_template_mismatch_refusal():
    d1 = pd.DataFrame({"y": [0, 1, 0], "x": [1, 2, 3]})
    d2 = pd.DataFrame({"y": [0, 1, 0], "x": [4, 5, 6]})
    fit = smf.ols("y ~ x", data=d1).fit()
    wiring = Node(kind="input", _payload=d2)
    with pytest.raises(CompileError, match="template_mismatch"):
        compile(wiring, fit)


def test_auto_resolution_recorded():
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, report = compile(wiring, fit, method="auto")
    assert plan.method_resolved == "delta"
    assert "delta" in plan.method_resolution_reason


def test_plan_hash_insensitive_to_n_jobs():
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan1, _ = compile(wiring, fit, method="delta")
    plan2, _ = compile(wiring, fit, method="delta")
    assert plan1.hash == plan2.hash


def test_plan_hash_sensitive_to_method():
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan1, _ = compile(wiring, fit, method="delta")
    plan2, _ = compile(wiring, fit, method="simulation")
    assert plan1.hash != plan2.hash


def test_unhashable_callable_marked():
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _ = compile(wiring, fit, method="delta")
    # No custom phi => not unhashable
    assert plan.unhashable_callable is False


# ---------------------------------------------------------------------------
# Inference-budget invariant (n_sim >= 1, B >= 1) — R3 audit follow-up
#
# The legacy session validated n_sim >= 1 and n_boot >= 1 and defaulted them
# to 4000 / 1000. compile() must preserve that invariant so the Plan never
# carries a zero budget into the executor (R5), where n_sim=0 crashes the
# simulation path and B=0 crashes the bootstrap path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["delta", "simulation", "bootstrap"])
def test_compile_default_budget_is_positive(method):
    """Every compiled Plan carries a positive simulation and bootstrap budget,
    so the executor never sees the crashing n_sim=0 / B=0 defaults."""
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _ = compile(wiring, fit, method=method)
    assert plan.n_sim >= 1
    assert plan.B >= 1


@pytest.mark.parametrize("method", ["delta", "simulation", "bootstrap"])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_sim": 0}, "n_sim"),
        ({"n_sim": -5}, "n_sim"),
        ({"B": 0}, "B"),
        ({"B": -1}, "B"),
    ],
)
def test_compile_rejects_nonpositive_budget(method, kwargs, match):
    """A non-positive n_sim or B is refused at compile time, for every method,
    rather than surfacing as a downstream executor crash."""
    d = df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match=match):
        compile(wiring, fit, method=method, **kwargs)
