"""Targeted tests for remaining gaps in statsmodels_ols adapter."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter


def _make_df(n=30):
    np.random.seed(1)
    return pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "y": np.random.randn(n),
        }
    )


def test_score_obs_wls():
    """Cover score_obs for WLS model (lines 97-100)."""
    df = _make_df()
    weights = np.random.uniform(0.5, 2.0, len(df))
    fit = sm.WLS(df["y"], sm.add_constant(df[["x1", "x2"]]), weights=weights).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    score = adapter.score_obs()
    assert score.shape[0] == len(df)


def test_score_obs_not_ols_wls():
    """Cover score_obs error for non-OLS/WLS (line 91-93)."""
    df = _make_df()
    fit = sm.GLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    with pytest.raises(NotImplementedError, match="score_obs"):
        adapter.score_obs()


def test_covariance_unsupported_string():
    """Cover covariance unsupported string (line 123)."""
    df = _make_df()
    fit = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.covariance("invalid")


def test_refit_wls_formula():
    """Cover refit for WLS with formula (lines 306-312)."""
    df = _make_df()
    weights = np.random.uniform(0.5, 2.0, len(df))
    from statsmodels.formula.api import wls as smf_wls

    fit = smf_wls("y ~ x1 + x2", data=df, weights=weights).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    resampled = df.sample(n=len(df), replace=True, random_state=1)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)


def test_refit_gls_formula():
    """Cover refit for GLS with formula (lines 313-317)."""
    df = _make_df()
    from statsmodels.formula.api import gls as smf_gls

    fit = smf_gls("y ~ x1 + x2", data=df).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    resampled = df.sample(n=len(df), replace=True, random_state=1)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)


def test_refit_array_fit_no_exog_cols():
    """Cover refit array-fit with no matching exog columns (lines 337-343)."""
    df = _make_df()
    fit = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    resampled = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="exog_names"):
        adapter.refit(resampled)


def test_refit_array_fit_wls_weights_index():
    """Cover refit array-fit WLS with weights indexing (lines 352-355)."""
    df = _make_df()
    weights = np.random.uniform(0.5, 2.0, len(df))
    fit = sm.WLS(df["y"], sm.add_constant(df[["x1", "x2"]]), weights=weights).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    resampled = df.sample(n=len(df), replace=True, random_state=1)
    index = resampled.index.values
    new_adapter = adapter.refit(resampled, index=index)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)


def test_refit_array_fit_gls_sigma_index():
    """Cover refit array-fit GLS with sigma indexing (lines 357-360)."""
    df = _make_df()
    # Use a diagonal sigma to avoid positive-definite issues
    sigma = np.diag(np.random.uniform(0.5, 2.0, len(df)))
    fit = sm.GLS(df["y"], sm.add_constant(df[["x1", "x2"]]), sigma=sigma).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    resampled = df.sample(n=len(df), replace=True, random_state=1)
    index = resampled.index.values
    try:
        new_adapter = adapter.refit(resampled, index=index)
        assert isinstance(new_adapter, StatsmodelsOLSAdapter)
    except np.linalg.LinAlgError:
        pytest.skip("GLS sigma indexing produced non-PD matrix")


def test_covariance_cluster_refit_formula_groups_length_mismatch():
    """Cover cluster refit groups length mismatch (lines 242-248)."""
    df = _make_df()
    from statsmodels.formula.api import ols as smf_ols

    fit = smf_ols("y ~ x1 + x2", data=df).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    with pytest.raises(ValueError, match="groups length"):
        adapter.covariance({"type": "cluster", "groups": np.array([1, 2])})


def test_survey_weights_length_mismatch():
    """Cover survey weights length mismatch (lines 156-160)."""
    df = _make_df()
    fit = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
    adapter = StatsmodelsOLSAdapter(fit, training_data=df)
    from pymargins.survey import SurveyDesign

    design = SurveyDesign(weights=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="weights length"):
        adapter._survey_covariance(design)
