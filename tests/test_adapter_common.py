"""Tests for shared adapter helpers in _common.py."""

import numpy as np
import pandas as pd
import pytest

from pymargins._adapters._common import (
    column_index_of_variable,
    build_variable_metadata,
    _infer_variable_type,
    design_matrix_from_df,
)
from pymargins._adapter import VariableInfo


def test_column_index_exact_match():
    exog_names = ["const", "x1", "x2"]
    meta = {
        "x1": VariableInfo(name="x1", var_type="continuous"),
        "x2": VariableInfo(name="x2", var_type="continuous"),
    }
    assert column_index_of_variable(exog_names, meta, "x1") == 1


def test_column_index_patsy_expanded():
    exog_names = ["Intercept", "C(region)[T.south]", "C(region)[T.east]"]
    meta = {"region": VariableInfo(name="region", var_type="categorical")}
    # Should NOT match because region is categorical (raises)
    with pytest.raises(ValueError, match="categorical"):
        column_index_of_variable(exog_names, meta, "region")


def test_column_index_startswith_not_substring():
    """Overlapping names must not match via substring."""
    exog_names = ["Intercept", "treatment", "treatment_time"]
    meta = {
        "treatment": VariableInfo(name="treatment", var_type="continuous"),
        "treatment_time": VariableInfo(name="treatment_time", var_type="continuous"),
    }
    assert column_index_of_variable(exog_names, meta, "treatment") == 1
    assert column_index_of_variable(exog_names, meta, "treatment_time") == 2


def test_column_index_interaction_prefix():
    """Interaction term x:z should be found when x is the variable name."""
    exog_names = ["Intercept", "z", "x:z"]
    meta = {"x": VariableInfo(name="x", var_type="continuous")}
    assert column_index_of_variable(exog_names, meta, "x") == 2


def test_column_index_interaction_suffix():
    """Interaction term z:x should be found when x is the variable name."""
    exog_names = ["Intercept", "z", "z:x"]
    meta = {"x": VariableInfo(name="x", var_type="continuous")}
    assert column_index_of_variable(exog_names, meta, "x") == 2


def test_column_index_i_transform():
    """I(x ** 2) term should be found for variable x."""
    exog_names = ["Intercept", "x", "I(x ** 2)"]
    meta = {"x": VariableInfo(name="x", var_type="continuous")}
    assert column_index_of_variable(exog_names, meta, "x") == 1


def test_column_index_i_transform_only():
    """When only I(x**2) is present and x is not, x should still resolve."""
    exog_names = ["Intercept", "I(x ** 2)"]
    meta = {"x": VariableInfo(name="x", var_type="continuous")}
    assert column_index_of_variable(exog_names, meta, "x") == 1


def test_infer_variable_type():
    s_binary = pd.Series([0, 1, 0, 1])
    assert _infer_variable_type(s_binary) == "binary"

    s_cat = pd.Series(["a", "b", "c"])
    assert _infer_variable_type(s_cat) == "categorical"

    s_cont = pd.Series([1.0, 2.0, 3.0])
    assert _infer_variable_type(s_cont) == "continuous"

    s_bool = pd.Series([True, False, True])
    assert _infer_variable_type(s_bool) == "binary"


def test_infer_variable_type_integer_is_continuous():
    """All integer series (except binary) are continuous; no discrete heuristic."""
    s_int = pd.Series([1, 2, 3, 4])
    assert _infer_variable_type(s_int) == "continuous"
    s_age = pd.Series(range(10))  # 10 distinct integer ages
    assert _infer_variable_type(s_age) == "continuous"


def test_build_variable_metadata():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "g": ["a", "b", "a"],
        "b": [0, 1, 0],
    })
    meta = build_variable_metadata(df)
    assert meta["x"].var_type == "continuous"
    assert meta["g"].var_type == "categorical"
    assert meta["b"].var_type == "binary"


def test_design_matrix_from_df_auto_injects_intercept():
    """Array-fit fallback should auto-inject const/Intercept if missing."""
    import statsmodels.api as sm
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [0.5, 1.5, 2.5],
        "y": [1.0, 2.0, 3.0],
    })
    X = sm.add_constant(df[["x1", "x2"]])
    y = df["y"].values
    fit = sm.OLS(y, X).fit()

    # Pass a df WITHOUT the const column
    new_df = pd.DataFrame({"x1": [4.0], "x2": [3.5]})
    X_new = design_matrix_from_df(fit, ["const", "x1", "x2"], new_df)
    assert X_new.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(X_new)[0, 0], 1.0)


def test_auto_detect_adapter_failure():
    """auto_detect_adapter should raise NotImplementedError for unsupported models."""
    from pymargins._adapter import auto_detect_adapter
    class UnsupportedModel:
        pass
    with pytest.raises(NotImplementedError, match="No adapter registered"):
        auto_detect_adapter(UnsupportedModel())
