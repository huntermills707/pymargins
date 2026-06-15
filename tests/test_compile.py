"""Tests for the compile pipeline (R5).

Design \u00a74.3/\u00a74.5/\u00a75.2, req \u00a73-\u00a74.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import steps
from pymargins._graph._compile import Compiled, CompileError, compile
from pymargins._graph._node import Node


def make_df(seed: int = 42, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, n),
            "x": rng.normal(size=n),
        }
    )


def test_invalid_ci_refuses():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match='ci="garbage" is not valid'):
        compile(wiring, fit, method="delta", ci="garbage")


def test_invalid_at_refuses():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match='at="bogus" is not valid'):
        compile(wiring, fit, at="bogus")


def test_constants_overrides_not_yet_supported():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match="constants_overrides is not supported"):
        compile(wiring, fit, constants_overrides=(("KAPPA_BORDERLINE", 0.5),))


@pytest.mark.parametrize("scale", ["response", "identity", "log", "logit", "probit"])
def test_valid_named_scales_compile(scale):
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _, _ = compile(wiring, fit, method="delta", scale=scale)
    assert plan.scale == scale


def test_invalid_scale_refuses():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match='scale="link" is not valid'):
        compile(wiring, fit, method="delta", scale="link")


def test_callable_at_fingerprints_in_plan():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    at_fn = lambda df: df.iloc[[0]]  # noqa: E731
    plan, _, _ = compile(wiring, fit, method="delta", at=at_fn)
    assert isinstance(plan.at, str)
    assert plan.at.startswith("callable:")


def test_unhashable_callable_at_sets_flag():
    """Two different un-introspectable at callables must set the honesty flag."""
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)

    class NoNameCallable:
        def __call__(self, df):
            return df.iloc[[0]]

    plan, _, _ = compile(wiring, fit, method="delta", at=NoNameCallable())
    assert plan.at == "callable:unhashable_callable"
    assert plan.unhashable_callable is True


def test_nested_unhashable_callable_at_sets_flag():
    """An un-introspectable callable nested inside an at= dict must set the honesty flag."""
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)

    class NoNameCallable:
        def __call__(self, df):
            return df.iloc[[0]]

    at = {"subgroup": NoNameCallable()}
    plan, _, _ = compile(wiring, fit, method="delta", at=at)
    assert "callable:unhashable_callable" in plan.at
    assert plan.unhashable_callable is True


def test_dict_at_with_non_json_values_fingerprints():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    at = {"x": np.array([0.0, 1.0])}
    plan, _, _ = compile(wiring, fit, method="delta", at=at)
    assert isinstance(plan.at, str)
    assert "0.0" in plan.at or "[0.0" in plan.at


def test_unknown_kwarg_typeerror():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(TypeError, match="kapa_threshold"):
        compile(wiring, fit, kapa_threshold=1)


def test_unknown_node_kind_refuses():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    weird = Node(kind="weird", inputs=(Node(kind="input", _payload=d),))
    with pytest.raises(CompileError, match="Unknown node kind"):
        compile(weird, fit)


def test_template_mismatch_refuses_names_both_fingerprints():
    d1 = pd.DataFrame({"y": [0, 1, 0], "x": [1, 2, 3]})
    d2 = pd.DataFrame({"y": [0, 1, 0], "x": [4, 5, 6]})
    fit = smf.ols("y ~ x", data=d1).fit()
    wiring = Node(kind="input", _payload=d2)
    with pytest.raises(CompileError, match="template_mismatch"):
        compile(wiring, fit)


def test_no_fingerprint_skip_path():
    """A wiring whose collect() raises must refuse, not silently skip."""

    class BadNode(Node):
        def collect(self):
            raise NotImplementedError("no collection")

    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = BadNode(kind="input", _payload=d)
    with pytest.raises(CompileError, match="Could not collect"):
        compile(wiring, fit)


def test_vcov_survey_conflict():
    from pymargins.survey import SurveyDesign

    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    design = SurveyDesign(weights=np.ones(len(d)))
    wiring = steps.input(d, design=design)
    with pytest.raises(CompileError, match="conflicts with the survey design"):
        compile(wiring, fit, vcov="HC1")


def test_vcov_cluster_without_cluster_refuses():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match='vcov="cluster" requires a cluster variable'):
        compile(wiring, fit, vcov="cluster")


def test_match_plus_filter_refused():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    matcher = type("M", (), {"matched_data": d, "cluster_ids": None, "population_note": "matched"})()
    wiring = steps.drop_outliers(steps.match(steps.input(d), matcher), rule=lambda df: df["x"] > 10)
    with pytest.raises(CompileError, match=r"match \+ row-filter"):
        compile(wiring, fit)


def test_transform_order_matches_wiring():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = steps.drop_outliers(
        steps.trim(steps.input(d), lower=-10, upper=10),
        rule=lambda df: df["x"] > 10,
    )
    plan, report, compiled = compile(wiring, fit)
    assert compiled.wiring_facts.transforms is not None
    names = [type(s).__name__ for s in compiled.wiring_facts.transforms]
    assert names == ["_TrimStage", "_DropOutliersStage"]


def test_auto_resolves_delta_low_kappa():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, report, compiled = compile(wiring, fit, method="auto")
    assert plan.method_resolved == "delta"
    assert "\u03ba" in plan.method_resolution_reason or "kappa" in plan.method_resolution_reason


def test_auto_resolves_simulation_high_kappa():
    """A near-boundary logit prediction should push posture \u03ba high."""
    rng = np.random.default_rng(4)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 2.0 + 5.0 * x1 - 3.0 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    fit = smf.glm("y ~ x1 + x2", data=d, family=sm.families.Binomial()).fit()
    wiring = Node(kind="input", _payload=d)
    plan, report, compiled = compile(wiring, fit, method="auto")
    assert plan.method_resolved == "simulation"
    assert "\u03ba" in plan.method_resolution_reason or "kappa" in plan.method_resolution_reason


def test_auto_reason_recorded():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, report, compiled = compile(wiring, fit, method="auto")
    assert plan.method_resolution_reason != ""
    assert plan.method_resolution_reason != "user-specified"


def test_ci_defaults_per_method():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan_delta, _, _ = compile(wiring, fit, method="delta", ci=None)
    plan_sim, _, _ = compile(wiring, fit, method="simulation", ci=None)
    plan_boot, _, _ = compile(wiring, fit, method="bootstrap", ci=None)
    assert plan_delta.ci == "wald"
    assert plan_sim.ci == "wald"
    assert plan_boot.ci == "percentile"


def test_plan_hash_golden():
    """A hand-built toy plan has a stable recipe-1 hash.

    If this test fails because the hash recipe changed, bump the recipe suffix
    in ``Plan.hash`` and update this constant in the same commit, documenting
    why the recipe changed.
    """
    d = make_df(n=10)
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _, _ = compile(wiring, fit, method="delta")
    assert plan.hash == "a8c10f7@1"


# ---------------------------------------------------------------------------
# Inference-budget invariant (n_sim >= 1, B >= 1) \u2014 R3 audit follow-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["delta", "simulation", "bootstrap"])
def test_compile_default_budget_is_positive(method):
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _, _ = compile(wiring, fit, method=method)
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
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    with pytest.raises(CompileError, match=match):
        compile(wiring, fit, method=method, **kwargs)


@pytest.mark.parametrize(
    ("method", "ci_in", "ci_out"),
    [
        ("bootstrap", "wald", "percentile"),
        ("bootstrap", "", "percentile"),
        ("bootstrap", "bca", "bca"),
        ("bootstrap", "basic", "basic"),
        ("delta", "wald", "wald"),
    ],
)
def test_compile_resolves_bootstrap_ci(method, ci_in, ci_out):
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _, _ = compile(wiring, fit, method=method, ci=ci_in)
    assert plan.ci == ci_out


# ---------------------------------------------------------------------------
# Plan / Compiled object shape
# ---------------------------------------------------------------------------


def test_compiled_returns_adapter_and_frozen_cov():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, report, compiled = compile(wiring, fit)
    assert isinstance(compiled, Compiled)
    assert compiled.adapter is not None
    assert compiled.frozen_cov is not None
    assert compiled.wiring_facts is not None


def test_plan_hash_insensitive_to_n_jobs():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan1, _, _ = compile(wiring, fit, method="delta")
    plan2, _, _ = compile(wiring, fit, method="delta")
    assert plan1.hash == plan2.hash


def test_plan_hash_sensitive_to_method():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan1, _, _ = compile(wiring, fit, method="delta")
    plan2, _, _ = compile(wiring, fit, method="simulation")
    assert plan1.hash != plan2.hash


def test_unhashable_callable_marked():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    plan, _, _ = compile(wiring, fit, method="delta")
    assert plan.unhashable_callable is False


def test_weights_fingerprint_in_plan():
    d = make_df()
    fit = smf.ols("y ~ x", data=d).fit()
    wiring = Node(kind="input", _payload=d)
    w = np.random.default_rng(0).uniform(0.5, 1.5, size=len(d))
    plan, _, compiled = compile(wiring, fit, weights=w)
    assert plan.weights_fingerprint is not None
    assert compiled.weights is not None
    assert np.allclose(compiled.weights, w)
