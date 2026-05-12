"""Tests for shared adapter helpers in _common.py."""

import numpy as np
import pandas as pd
import pytest

from pymargins._adapters._common import (
    column_index_of_variable,
    build_variable_metadata,
    _infer_variable_type,
    design_matrix_from_df,
    validate_vcov_spec,
    extract_training_data,
)
from pymargins._adapter import VariableInfo, ModelAdapter, WrappedFDAdapter


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
    """auto_detect_adapter should raise TypeError for unsupported models."""
    from pymargins._adapter import auto_detect_adapter
    class UnsupportedModel:
        pass
    with pytest.raises(TypeError) as exc_info:
        auto_detect_adapter(UnsupportedModel())
    msg = str(exc_info.value)
    assert "No adapter registered" in msg
    assert "Currently registered adapters" in msg
    assert "StatsmodelsGLMAdapter" in msg
    assert "StatsmodelsOLSAdapter" in msg
    assert "To write a custom adapter" in msg


def test_auto_detect_adapter_failure_suggests_closest():
    """Error should suggest the closest adapter based on heuristics."""
    from pymargins._adapter import auto_detect_adapter

    # A statsmodels-like object with an unrecognized class name should
    # suggest statsmodels adapters.
    class FakeStatsmodelsResult:
        pass
    FakeStatsmodelsResult.__module__ = "statsmodels.something"
    FakeStatsmodelsResult.__name__ = "SomeResult"

    with pytest.raises(TypeError) as exc_info:
        auto_detect_adapter(FakeStatsmodelsResult())
    msg = str(exc_info.value)
    assert "StatsmodelsGLMAdapter" in msg or "StatsmodelsOLSAdapter" in msg
    assert "adapters:" in msg


def test_auto_detect_adapter_failure_suggests_ols():
    """A statsmodels WLS-like object should strongly suggest OLS adapter."""
    from pymargins._adapter import auto_detect_adapter

    class FakeWLSResult:
        pass
    FakeWLSResult.__module__ = "statsmodels.regression"
    FakeWLSResult.__name__ = "WLSResults"

    with pytest.raises(TypeError) as exc_info:
        auto_detect_adapter(FakeWLSResult())
    msg = str(exc_info.value)
    assert "Did you mean" in msg
    assert "StatsmodelsOLSAdapter" in msg


def test_auto_detect_adapter_failure_cls_only_weak_match():
    """A non-statsmodels class whose name contains 'GLM' should weakly suggest
    the GLM adapter (class-name hint without module prefix)."""
    from pymargins._adapter import auto_detect_adapter

    class FakeGLMResult:
        pass
    FakeGLMResult.__module__ = "sklearn.something"
    FakeGLMResult.__name__ = "FakeGLMResult"

    with pytest.raises(TypeError) as exc_info:
        auto_detect_adapter(FakeGLMResult())
    msg = str(exc_info.value)
    assert "Possibly related adapters" in msg
    assert "StatsmodelsGLMAdapter" in msg


# ---------------------------------------------------------------------------
# validate_vcov_spec unit tests (IMPLEMENTATION_GUIDE.md §2.3)
# ---------------------------------------------------------------------------

def test_validate_vcov_spec_none():
    """None is always valid."""
    validate_vcov_spec(None)  # no error


def test_validate_vcov_spec_ndarray():
    """A user-supplied ndarray is valid."""
    validate_vcov_spec(np.eye(3))


def test_validate_vcov_spec_hc_strings():
    """HC0-HC3 (case-insensitive) are valid."""
    for s in ("HC0", "hc1", "Hc2", "hC3"):
        validate_vcov_spec(s)


def test_validate_vcov_spec_unsupported_string():
    """Unsupported strings raise ValueError."""
    with pytest.raises(ValueError, match="does not support vcov='HAC'"):
        validate_vcov_spec("HAC")


def test_validate_vcov_spec_cluster_dict():
    """Cluster dict with groups is valid."""
    validate_vcov_spec({"type": "cluster", "groups": [1, 2, 3]})


def test_validate_vcov_spec_cluster_missing_groups():
    """Cluster dict without groups raises ValueError."""
    with pytest.raises(ValueError, match="cluster vcov requires 'groups'"):
        validate_vcov_spec({"type": "cluster"})


def test_validate_vcov_spec_unsupported_dict():
    """Dict with unsupported type raises ValueError."""
    with pytest.raises(ValueError, match="does not support vcov dict with type='hac'"):
        validate_vcov_spec({"type": "hac"})


def test_validate_vcov_spec_unsupported_type():
    """Lists, ints, etc. raise ValueError."""
    with pytest.raises(ValueError, match="does not support vcov spec of type list"):
        validate_vcov_spec([1, 2, 3])
    with pytest.raises(ValueError, match="does not support vcov spec of type int"):
        validate_vcov_spec(42)


def test_validate_vcov_spec_custom_adapter_name():
    """The adapter name is included in the error message."""
    with pytest.raises(ValueError, match="MyAdapter does not support vcov='foo'"):
        validate_vcov_spec("foo", adapter_name="MyAdapter")


def test_validate_vcov_spec_jax_array():
    """JAX arrays should be accepted as user-supplied vcov."""
    import jax.numpy as jnp
    validate_vcov_spec(jnp.eye(3))


def test_validate_vcov_spec_cluster_empty_groups():
    """Empty groups list/array should be rejected."""
    with pytest.raises(ValueError, match="groups' must not be empty"):
        validate_vcov_spec({"type": "cluster", "groups": []})
    import numpy as np
    with pytest.raises(ValueError, match="groups' must not be empty"):
        validate_vcov_spec({"type": "cluster", "groups": np.array([])})


# ---------------------------------------------------------------------------
# ModelAdapter ABC
# ---------------------------------------------------------------------------

def test_model_adapter_is_abc():
    """ModelAdapter should inherit from abc.ABC and be uninstantiable."""
    import abc
    assert issubclass(ModelAdapter, abc.ABC)
    with pytest.raises(TypeError):
        ModelAdapter()


def test_model_adapter_abstract_methods():
    """Subclasses missing abstract methods should not be instantiable."""
    class Incomplete(ModelAdapter):
        pass
    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------------
# WrappedFDAdapter offset rejection
# ---------------------------------------------------------------------------

def test_wrapped_fd_adapter_rejects_offset():
    """WrappedFDAdapter.predict should raise NotImplementedError when offset is passed."""
    class DummyAdapter(WrappedFDAdapter):
        def coefficients(self):
            return np.array([1.0])
        def covariance(self, vcov_spec=None):
            return np.eye(1)
        def native_predict(self, beta_np, X):
            return X @ beta_np
        def design_matrix_from_df(self, df):
            return np.asarray(df)
        def column_index_of_variable(self, name):
            return 0
        def variable_metadata(self):
            return {}

    adapter = DummyAdapter()
    with pytest.raises(ValueError, match="offset"):
        adapter.predict(np.array([1.0]), np.array([[1.0]]), offset=np.array([0.5]))


# ---------------------------------------------------------------------------
# design_matrix_from_df missing columns
# ---------------------------------------------------------------------------

def test_design_matrix_from_df_missing_columns_raises():
    """Missing required columns should raise ValueError, not silently produce NaN."""
    import statsmodels.api as sm
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    X = sm.add_constant(df[["x1"]])
    fit = sm.OLS(df["y"], X).fit()

    new_df = pd.DataFrame({"x1": [4.0]})
    with pytest.raises(ValueError, match="Missing columns"):
        design_matrix_from_df(fit, ["const", "x1", "x2"], new_df)


# ---------------------------------------------------------------------------
# extract_training_data None frame handling
# ---------------------------------------------------------------------------

def test_extract_training_data_none_frame():
    """If results.model.data.frame exists but is None, should raise ValueError."""
    class FakeData:
        frame = None
    class FakeModel:
        data = FakeData()
    class FakeResults:
        model = FakeModel()

    with pytest.raises(ValueError, match="training_data must be provided"):
        extract_training_data(FakeResults(), None)


# ---------------------------------------------------------------------------
# VariableType and _infer_variable_type no "discrete"
# ---------------------------------------------------------------------------

def test_infer_variable_type_never_returns_discrete():
    """_infer_variable_type should never emit 'discrete'."""
    s_int = pd.Series([1, 2, 3, 4])
    assert _infer_variable_type(s_int) == "continuous"

    s_float = pd.Series([1.0, 2.0, 3.0])
    assert _infer_variable_type(s_float) == "continuous"


def test_variable_type_literal_excludes_discrete():
    """VariableType should not include 'discrete'."""
    from typing import get_args
    from pymargins._adapter import VariableType
    assert "discrete" not in get_args(VariableType)


# ---------------------------------------------------------------------------
# validate_vcov_spec ndarray shape limitation
# ---------------------------------------------------------------------------

def test_validate_vcov_spec_accepts_any_ndarray_shape():
    """validate_vcov_spec cannot check parameter count, so any ndarray is accepted."""
    validate_vcov_spec(np.array([1.0, 2.0]))  # 1D array — accepted
    validate_vcov_spec(np.zeros((2, 3)))     # non-square — accepted
