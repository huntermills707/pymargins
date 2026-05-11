"""Tests for MarginsResult.to_frame() enriched output."""

import numpy as np
import pandas as pd
import pytest

import statsmodels.api as sm

from pymargins import Margins
from pymargins.scenarios import pairwise


@pytest.fixture
def df():
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    group = np.random.choice(["A", "B"], n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + np.random.randn(n) * 0.5
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})


@pytest.fixture
def m(df):
    mod = sm.OLS.from_formula("y ~ x1 + x2 + C(group)", data=df)
    res = mod.fit()
    return Margins(res)


# ---------------------------------------------------------------------------
# Core columns
# ---------------------------------------------------------------------------

def test_to_frame_has_core_columns(m):
    result = m.dydx("x1")
    df = result.to_frame()
    assert "estimate" in df.columns
    assert "std_error" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns
    assert "conf_level" in df.columns
    assert "n_obs" in df.columns
    assert "method" in df.columns
    assert "kind" in df.columns
    assert "term" in df.columns


def test_to_frame_has_p_value(m):
    result = m.dydx("x1")
    df = result.to_frame()
    assert "statistic" in df.columns
    assert "p_value" in df.columns
    assert df["p_value"].iloc[0] < 0.05  # x1 is significant


def test_to_frame_term_is_list_for_dydx(m):
    result = m.dydx("x1")
    df = result.to_frame()
    assert df["term"].iloc[0] == ["x1"]


def test_to_frame_term_is_list_for_multi_dydx(m):
    result = m.dydx(["x1", "x2"])
    df = result.to_frame()
    assert df["term"].iloc[0] == ["x1", "x2"]


# ---------------------------------------------------------------------------
# Scenario columns for predictions
# ---------------------------------------------------------------------------

def test_to_frame_prediction_has_scenario_columns(m):
    result = m.predict(atexog={"x1": [0, 1, 2]})
    df = result.to_frame()
    assert "x1" in df.columns
    assert list(df["x1"]) == [0, 1, 2]


def test_to_frame_prediction_single_atexog_has_scenario(m):
    result = m.predict(atexog={"x1": 0})
    df = result.to_frame()
    assert "x1" in df.columns
    assert df["x1"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Over columns
# ---------------------------------------------------------------------------

def test_to_frame_over_columns(m):
    result = m.dydx("x1", over="group")
    df = result.to_frame()
    assert "over" in df.columns
    assert "over_value" in df.columns
    assert set(df["over"]) == {"group"}
    assert set(df["over_value"]) == {"A", "B"}


def test_to_frame_over_with_prediction(m):
    result = m.predict(over="group")
    df = result.to_frame()
    assert "over" in df.columns
    assert "over_value" in df.columns
    assert set(df["over_value"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_to_frame_fallback_columns(m):
    result = m.dydx("x1")
    df = result.to_frame()
    assert "fallback_triggered" in df.columns
    assert "kappa" in df.columns
    assert df["fallback_triggered"].iloc[0] == False


# ---------------------------------------------------------------------------
# Contrasts and evaluate (minimal metadata)
# ---------------------------------------------------------------------------

def test_to_frame_contrasts_has_scenarios(m):
    scenarios, weights = pairwise("x1", [0, 1])
    result = m.contrasts(scenarios=scenarios, contrasts=weights)
    df = result.to_frame()
    assert "kind" in df.columns
    assert df["kind"].iloc[0] == "contrasts"
    assert "label" in df.columns
    # Scenarios from contrast should be present
    assert "scenarios" in result.estimand_metadata


def test_to_frame_evaluate_has_scenarios(m):
    result = m.evaluate(
        scenarios=[
            {"atexog": {"x1": 1}},
            {"atexog": {"x1": 0}},
        ],
        compose=lambda p: p[0] - p[1],
    )
    df = result.to_frame()
    assert "kind" in df.columns
    assert df["kind"].iloc[0] == "evaluate"
    assert "scenarios" in result.estimand_metadata


# ---------------------------------------------------------------------------
# Row count matches
# ---------------------------------------------------------------------------

def test_to_frame_row_count_matches_grid(m):
    result = m.predict(atexog={"x1": [0, 1, 2], "x2": [10, 20]})
    df = result.to_frame()
    assert len(df) == 6  # 3 x 2 grid
    assert "x1" in df.columns
    assert "x2" in df.columns


def test_to_frame_row_count_matches_over(m):
    result = m.dydx("x1", over="group")
    df = result.to_frame()
    assert len(df) == 2  # A and B


# ---------------------------------------------------------------------------
# Type preservation
# ---------------------------------------------------------------------------

def test_to_frame_preserves_numeric_types(m):
    result = m.predict(atexog={"x1": [0.5, 1.5]})
    df = result.to_frame()
    assert pd.api.types.is_numeric_dtype(df["x1"])
    assert df["x1"].iloc[0] == 0.5
    assert df["x1"].iloc[1] == 1.5


def test_to_frame_preserves_string_types(m):
    result = m.predict(atexog={"group": ["A", "B"]})
    df = result.to_frame()
    assert df["group"].iloc[0] == "A"
    assert df["group"].iloc[1] == "B"
