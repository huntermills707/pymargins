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
