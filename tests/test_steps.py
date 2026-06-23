"""Tests for the steps module (W2.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from pymargins import steps
from pymargins._graph._compile import compile


def test_input_is_only_dependence_carrier():
    df = pd.DataFrame({"x": [1, 2, 3]})
    n = steps.input(df, design=None, cluster=None, block=None)
    assert n.kind == "input"
    assert n.params == ()


def test_input_carries_design():
    from pymargins.survey import SurveyDesign
    df = pd.DataFrame({"x": [1, 2, 3]})
    design = SurveyDesign(weights=df["x"].values)
    n = steps.input(df, design=design)
    assert n.params[0][0] == "design"
    assert n.params[0][1] is design


def test_trim_produces_contract_fields():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    n = steps.trim(steps.input(df), lower=2, upper=4)
    assert n.kind == "trim"
    assert n.alters_rows is True
    assert n.fan is None


def test_drop_outliers_produces_contract_fields():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    n = steps.drop_outliers(steps.input(df), rule=lambda d: d["x"] > 4)
    assert n.kind == "drop_outliers"
    assert n.alters_rows is True


def test_match_population_note():
    df = pd.DataFrame({"x": [1, 2, 3], "treat": [1, 0, 1]})
    matcher = type("FakeMatcher", (), {"matched_data": df, "cluster_ids": [1, 1, 2], "rematch": lambda self, d: d})()
    n = steps.match(steps.input(df), matcher)
    assert n.alters_rows is True
    assert n.population_note == "matched sample"



def test_reimpute_collect_returns_imputed_parent():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    n = steps.input(df)
    r = steps.reimpute(n, imputer=lambda d: d.fillna(0))
    out = r.collect()
    # prepare() applies the imputer to the point-execution output so the
    # template model can be fit on an imputed frame.
    assert out["x"].isna().sum() == 0


def test_impute_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="lands in 0.5.0/0.6.0"):
        steps.impute(steps.input(df), None, m=5)


def test_imputed_raises():
    with pytest.raises(NotImplementedError, match="lands in 0.5.0/0.6.0"):
        steps.imputed([])


def test_propensity_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="lands in 0.5.0/0.6.0"):
        steps.propensity(steps.input(df), "treat ~ x")


def test_transform_order_preserved():
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"y": [0, 1, 0, 1, 0], "x": [1, 2, 3, 4, 5]})
    # Use no-op transforms so the wiring output still matches the fit data.
    prep = steps.trim(steps.input(df), lower=-100, upper=100)
    prep = steps.drop_outliers(prep, rule=lambda d: d["x"] > 100)
    fit = smf.ols("y ~ x", data=df).fit()
    plan, report, compiled = compile(prep, fit)
    stages = compiled.wiring_facts.transforms
    assert len(stages) == 2
    assert type(stages[0]).__name__ == "_TrimStage"
    assert type(stages[1]).__name__ == "_DropOutliersStage"


def test_design_affects_plan_hash():
    import statsmodels.formula.api as smf

    from pymargins.survey import SurveyDesign
    df = pd.DataFrame({"y": [0, 1, 0], "x": [1, 2, 3]})
    fit = smf.ols("y ~ x", data=df).fit()
    d1 = SurveyDesign(weights=df["x"].values)
    d2 = SurveyDesign(weights=(df["x"] * 2).values)
    from pymargins._graph._compile import compile
    plan1, _, _ = compile(steps.input(df, design=d1), fit)
    plan2, _, _ = compile(steps.input(df, design=d2), fit)
    assert plan1.hash != plan2.hash


def test_scenario_reexports():
    assert callable(steps.at_levels)
    assert callable(steps.pairwise)
