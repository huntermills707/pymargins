"""Targeted tests for remaining gaps in statsmodels_zi adapter."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.count_model as sm_zi

from pymargins._adapters.statsmodels_zi import (
    StatsmodelsZIAdapter,
    _zi_model_cls,
)


def _make_df(n=50):
    np.random.seed(1)
    df = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "z1": np.random.randn(n),
        }
    )
    # Create zero-inflated data
    rate = np.exp(0.5 + 0.3 * df["x1"])
    df["y"] = np.random.poisson(rate, size=n)
    df["y"] = np.where(np.random.rand(n) < 0.2, 0, df["y"])
    return df


def test_zi_model_cls_unknown():
    """Cover _zi_model_cls unknown model class (line 56-57)."""
    with pytest.raises(ValueError, match="Unknown ZI model class"):
        _zi_model_cls("UnknownModel")


def test_adapter_array_fit_param_names():
    """Cover adapter array-fit param name synthesis (lines 91-96)."""
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    assert len(adapter._infl_param_names) > 0
    assert len(adapter._count_param_names) > 0


def test_adapter_attach_validates_vcov():
    """Cover attach validation of vcov_spec."""
    df = _make_df()
    fit = sm_zi.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", data=df, exog_infl=df[["z1"]]
    ).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    mock_session = MagicMock()
    mock_session.vcov_spec = "invalid"
    with pytest.raises(ValueError):
        adapter.attach(mock_session)


def test_covariance_cluster_groups_mismatch():
    """Cover cluster covariance groups length mismatch (lines 280-288)."""
    df = _make_df()
    fit = sm_zi.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", data=df, exog_infl=df[["z1"]]
    ).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    with pytest.raises(ValueError, match="groups length"):
        adapter.covariance({"type": "cluster", "groups": np.array([1, 2])})


def test_refit_array_fit_no_count_cols():
    """Cover refit array-fit no count columns (lines 370-374)."""
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    resampled = pd.DataFrame({"z1": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError, match="count param names"):
        adapter.refit(resampled)


def test_refit_array_fit_no_infl_cols():
    """Cover refit array-fit no inflation columns (lines 385-389)."""
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    resampled = pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "y": [0, 1]})
    with pytest.raises(ValueError, match="inflation names"):
        adapter.refit(resampled)


def test_refit_array_fit_add_count_const():
    """Cover refit array-fit adding count const (lines 376-382)."""
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    # Drop const from resampled data to trigger insertion
    resampled = df[["x1", "x2", "z1", "y"]].sample(
        n=len(df), replace=True, random_state=1
    )
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsZIAdapter)


def test_refit_array_fit_add_infl_const():
    """Cover refit array-fit adding inflation const (lines 391-395)."""
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    resampled = df[["x1", "x2", "z1", "y"]].sample(
        n=len(df), replace=True, random_state=1
    )
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsZIAdapter)


def test_build_inflation_matrix_missing_cols():
    """Cover _build_inflation_matrix missing columns error (lines 220-224)."""
    # Use array-fit to avoid patsy path
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    bad_df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Missing inflation columns"):
        adapter._build_infl_matrix(bad_df)


def test_build_count_matrix_missing_cols():
    """Cover _build_count_matrix missing columns error (lines 243-249)."""
    # Use array-fit to avoid patsy path
    df = _make_df()
    endog = df["y"].values
    exog = pd.DataFrame({"const": 1.0, "x1": df["x1"], "x2": df["x2"]})
    exog_infl = pd.DataFrame({"const": 1.0, "z1": df["z1"]})
    fit = sm_zi.ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    bad_df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Missing count columns"):
        adapter._build_count_matrix(bad_df)


def test_covariance_unsupported_string():
    """Cover covariance unsupported string (line 121)."""
    df = _make_df()
    fit = sm_zi.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", data=df, exog_infl=df[["z1"]]
    ).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.covariance("invalid")


def test_collect_original_fit_kwargs():
    """Cover _collect_original_fit_kwargs (lines 406-420)."""
    df = _make_df()
    offset = np.log(np.ones(len(df)) * 2)
    fit = sm_zi.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", data=df, exog_infl=df[["z1"]], offset=offset
    ).fit(disp=False)
    adapter = StatsmodelsZIAdapter(fit, training_data=df)
    kwargs = adapter._collect_original_fit_kwargs()
    assert "offset" in kwargs


from unittest.mock import MagicMock
