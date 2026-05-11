"""Tests for PolarsTabular backend."""

import numpy as np
import pandas as pd
import pytest

polars = pytest.importorskip("polars")

from pymargins._tabular import (
    PandasTabular,
    PolarsTabular,
    TabularData,
    as_tabular,
    concat_tables,
    to_pandas_if_needed,
)


# ---------------------------------------------------------------------------
# Construction & introspection
# ---------------------------------------------------------------------------

def test_polars_tabular_columns():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    assert tab.columns == ["a", "b"]


def test_polars_tabular_shape():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    assert tab.shape == (3, 2)


def test_polars_tabular_dtypes():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    dtypes = tab.dtypes()
    assert dtypes["a"] == polars.Int64
    assert dtypes["b"] == polars.String


# ---------------------------------------------------------------------------
# Column access
# ---------------------------------------------------------------------------

def test_polars_tabular_getitem():
    df = polars.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    tab = PolarsTabular(df)
    a = tab["a"]
    assert isinstance(a, np.ndarray)
    np.testing.assert_array_equal(a, [1, 2, 3])


def test_polars_tabular_with_column_array():
    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    tab2 = tab.with_column("a", np.array([10, 20, 30]))
    np.testing.assert_array_equal(tab2["a"], [10, 20, 30])
    # Original unchanged
    np.testing.assert_array_equal(tab["a"], [1, 2, 3])


def test_polars_tabular_with_column_scalar():
    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    tab2 = tab.with_column("a", 99)
    np.testing.assert_array_equal(tab2["a"], [99, 99, 99])


def test_polars_tabular_with_column_new():
    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    tab2 = tab.with_column("b", np.array(["x", "y", "z"]))
    assert "b" in tab2.columns
    np.testing.assert_array_equal(tab2["b"], ["x", "y", "z"])


# ---------------------------------------------------------------------------
# Row slicing (iloc)
# ---------------------------------------------------------------------------

def test_polars_tabular_iloc_int_list():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    sub = tab.iloc([0, 2])
    assert sub.shape == (2, 2)
    np.testing.assert_array_equal(sub["a"], [1, 3])


def test_polars_tabular_iloc_bool_mask():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    mask = np.array([True, False, True])
    sub = tab.iloc(mask)
    assert sub.shape == (2, 2)
    np.testing.assert_array_equal(sub["a"], [1, 3])


def test_polars_tabular_iloc_scalar():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    sub = tab.iloc(1)
    assert sub.shape == (1, 2)
    np.testing.assert_array_equal(sub["a"], [2])


# ---------------------------------------------------------------------------
# Groupby
# ---------------------------------------------------------------------------

def test_polars_tabular_groupby_single_key():
    df = polars.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 40]})
    tab = PolarsTabular(df)
    groups = list(tab.groupby(["a"]))
    assert len(groups) == 2
    # Keys should be scalar (pandas compat), not tuple
    keys = [g for g, _ in groups]
    assert keys == [1, 2]
    # Group data
    g1 = groups[0][1]
    g2 = groups[1][1]
    assert g1.shape == (2, 2)
    assert g2.shape == (2, 2)
    np.testing.assert_array_equal(g1["b"], [10, 20])
    np.testing.assert_array_equal(g2["b"], [30, 40])


def test_polars_tabular_groupby_multi_key():
    df = polars.DataFrame({
        "a": [1, 1, 2, 2],
        "b": ["x", "y", "x", "y"],
        "c": [10, 20, 30, 40],
    })
    tab = PolarsTabular(df)
    groups = list(tab.groupby(["a", "b"]))
    assert len(groups) == 4
    keys = [g for g, _ in groups]
    # Multi-key groups should be tuples
    assert all(isinstance(k, tuple) for k in keys)
    assert set(keys) == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

def test_polars_tabular_concat():
    t1 = PolarsTabular(polars.DataFrame({"a": [1, 2]}))
    t2 = PolarsTabular(polars.DataFrame({"a": [3, 4]}))
    merged = PolarsTabular.concat([t1, t2])
    assert merged.shape == (4, 1)
    np.testing.assert_array_equal(merged["a"], [1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def test_polars_tabular_to_pandas():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    pdf = tab.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    np.testing.assert_array_equal(pdf["a"].values, [1, 2, 3])
    np.testing.assert_array_equal(pdf["b"].values, ["x", "y", "z"])


def test_polars_tabular_to_pandas_patsy_safe_dtypes():
    """Verify that to_pandas produces numpy dtypes patsy can handle."""
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tab = PolarsTabular(df)
    pdf = tab.to_pandas()
    # Should be numpy int64 and object, not pyarrow extension types
    assert str(pdf["a"].dtype) == "int64"
    assert str(pdf["b"].dtype) == "object"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_polars_tabular_is_runtime_checkable():
    tab = PolarsTabular(polars.DataFrame({"a": [1]}))
    assert isinstance(tab, TabularData)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_as_tabular_from_polars():
    df = polars.DataFrame({"a": [1, 2]})
    tab = as_tabular(df)
    assert isinstance(tab, PolarsTabular)


def test_as_tabular_from_pandas():
    df = pd.DataFrame({"a": [1, 2]})
    tab = as_tabular(df)
    assert isinstance(tab, PandasTabular)


def test_as_tabular_from_existing_wrapper():
    t1 = PolarsTabular(polars.DataFrame({"a": [1]}))
    t2 = as_tabular(t1)
    assert t2 is t1


def test_as_tabular_raises_on_bad_type():
    with pytest.raises(TypeError):
        as_tabular([1, 2, 3])


# ---------------------------------------------------------------------------
# concat_tables mixed backends
# ---------------------------------------------------------------------------

def test_concat_tables_polars_only():
    t1 = PolarsTabular(polars.DataFrame({"a": [1]}))
    t2 = PolarsTabular(polars.DataFrame({"a": [2]}))
    merged = concat_tables([t1, t2])
    assert isinstance(merged, PolarsTabular)
    np.testing.assert_array_equal(merged["a"], [1, 2])


def test_concat_tables_pandas_only():
    t1 = PandasTabular(pd.DataFrame({"a": [1]}))
    t2 = PandasTabular(pd.DataFrame({"a": [2]}))
    merged = concat_tables([t1, t2])
    assert isinstance(merged, PandasTabular)
    np.testing.assert_array_equal(merged["a"], [1, 2])


def test_concat_tables_mixed():
    t1 = PolarsTabular(polars.DataFrame({"a": [1]}))
    t2 = PandasTabular(pd.DataFrame({"a": [2]}))
    merged = concat_tables([t1, t2])
    # Mixed falls back to pandas
    assert isinstance(merged, PandasTabular)
    np.testing.assert_array_equal(merged["a"], [1, 2])


def test_concat_tables_empty_raises():
    with pytest.raises(ValueError, match="at least one table"):
        concat_tables([])


# ---------------------------------------------------------------------------
# to_pandas_if_needed
# ---------------------------------------------------------------------------

def test_to_pandas_if_needed_polars():
    tab = PolarsTabular(polars.DataFrame({"a": [1]}))
    pdf = to_pandas_if_needed(tab)
    assert isinstance(pdf, pd.DataFrame)


def test_to_pandas_if_needed_pandas():
    df = pd.DataFrame({"a": [1]})
    pdf = to_pandas_if_needed(df)
    assert pdf is df


# ---------------------------------------------------------------------------
# Escape hatches
# ---------------------------------------------------------------------------

def test_polars_tabular_copy():
    t1 = PolarsTabular(polars.DataFrame({"a": [1, 2]}))
    t2 = t1.copy()
    assert t2 is not t1
    np.testing.assert_array_equal(t2["a"], [1, 2])


def test_polars_tabular_len():
    tab = PolarsTabular(polars.DataFrame({"a": [1, 2, 3]}))
    assert len(tab) == 3


def test_polars_tabular_repr():
    tab = PolarsTabular(polars.DataFrame({"a": [1, 2, 3]}))
    assert "PolarsTabular" in repr(tab)
    assert "shape=(3, 1)" in repr(tab)
