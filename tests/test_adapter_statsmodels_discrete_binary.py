"""Tests for StatsmodelsDiscreteBinaryAdapter."""

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_discrete_binary import (
    StatsmodelsDiscreteBinaryAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    lp = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"]
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return df


@pytest.fixture
def fit_logit_formula(df_binary):
    return smf.logit("y ~ age + treatment", data=df_binary).fit(disp=False)


@pytest.fixture
def fit_probit_formula(df_binary):
    return smf.probit("y ~ age + treatment", data=df_binary).fit(disp=False)


@pytest.fixture
def fit_logit_array(df_binary):
    X = df_binary[["age", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_binary["y"].values
    return sm.Logit(y, X).fit(disp=False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_auto_detect_logit(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    assert isinstance(adapter, StatsmodelsDiscreteBinaryAdapter)


def test_auto_detect_probit(fit_probit_formula):
    adapter = auto_detect_adapter(fit_probit_formula)
    assert isinstance(adapter, StatsmodelsDiscreteBinaryAdapter)


def test_adapter_coefficients(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    coef = adapter.coefficients()
    assert coef.shape == (3,)
    np.testing.assert_allclose(
        np.asarray(coef), fit_logit_formula.params.values, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_covariance_default(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    cov = adapter.covariance()
    assert cov.shape == (3, 3)


def test_covariance_hc3_via_refit(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    cov = adapter.covariance(vcov_spec="hc3")
    assert cov.shape == (3, 3)


def test_covariance_cluster_via_refit(fit_logit_formula, df_binary):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    groups = df_binary["treatment"].values
    cov = adapter.covariance(vcov_spec={"type": "cluster", "groups": groups})
    assert cov.shape == (3, 3)


def test_covariance_unsupported_string_raises(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov string"):
        adapter.covariance(vcov_spec="hac")


def test_covariance_unsupported_dict_raises(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov dict type"):
        adapter.covariance(vcov_spec={"type": "hac"})


def test_covariance_cluster_missing_groups_raises(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires"):
        adapter.covariance(vcov_spec={"type": "cluster"})


def test_covariance_unsupported_type_raises(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov_spec"):
        adapter.covariance(vcov_spec=123)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels_logit(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    beta = adapter.coefficients()
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    pred = adapter.predict(beta, X)
    np.testing.assert_allclose(
        np.asarray(pred),
        fit_logit_formula.predict(df[:5]),
        rtol=1e-5,
    )


def test_predict_matches_statsmodels_probit(fit_probit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_probit_formula)
    beta = adapter.coefficients()
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    pred = adapter.predict(beta, X)
    np.testing.assert_allclose(
        np.asarray(pred),
        fit_probit_formula.predict(df[:5]),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(fit_logit_formula, df_binary):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    p = len(fit_logit_formula.model.exog_names)
    assert X.shape[1] == p


def test_variable_metadata(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    meta = adapter.variable_metadata()
    assert "age" in meta
    assert meta["age"].var_type == "continuous"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_margins_predict_logit(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    m = GComputation(fit_logit_formula, adapter=adapter, scale="response")
    res = m.predict()
    assert res.estimate.size == 1
    assert 0 < float(res.estimate) < 1


def test_margins_dydx_logit(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    m = GComputation(fit_logit_formula, adapter=adapter, scale="response")
    res = m.dydx("age")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula_logit(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    new_adapter = adapter.refit(adapter.training_data)
    assert isinstance(new_adapter, StatsmodelsDiscreteBinaryAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array_logit(fit_logit_array, df_binary):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_array, training_data=df_binary)
    new_adapter = adapter.refit(df_binary)
    assert isinstance(new_adapter, StatsmodelsDiscreteBinaryAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_with_index_logit(fit_logit_formula, df_binary):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    idx = np.random.default_rng(7).choice(
        len(df_binary), size=len(df_binary), replace=True
    )
    new_adapter = adapter.refit(adapter.training_data.iloc[idx], index=idx)
    assert isinstance(new_adapter, StatsmodelsDiscreteBinaryAdapter)


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------


def test_attach_rejects_bad_vcov(fit_logit_formula):
    adapter = StatsmodelsDiscreteBinaryAdapter(fit_logit_formula)
    from unittest.mock import MagicMock

    session = MagicMock()
    session.vcov_spec = "HAC"
    with pytest.raises(ValueError, match="HAC"):
        adapter.attach(session)
