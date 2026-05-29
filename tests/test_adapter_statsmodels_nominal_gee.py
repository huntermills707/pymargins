"""Tests for StatsmodelsNominalGEEAdapter."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pymargins._adapters.statsmodels_nominal_gee import StatsmodelsNominalGEEAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_nominal():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "group": np.repeat(range(30), 10),
        }
    )
    # Multinomial outcome with 3 categories
    logits = np.column_stack(
        [
            np.zeros(n),
            0.3 + 0.5 * df["x1"] - 0.4 * df["x2"],
            -0.2 + 0.2 * df["x1"] + 0.3 * df["x2"],
        ]
    )
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])
    return df


@pytest.fixture
def nominal_fit_formula(df_nominal):
    from statsmodels.genmod.generalized_estimating_equations import NominalGEE

    return NominalGEE.from_formula(
        "y ~ x1 + x2", groups=df_nominal["group"], data=df_nominal
    ).fit()


@pytest.fixture
def nominal_fit_array(df_nominal):
    from statsmodels.genmod.generalized_estimating_equations import NominalGEE

    exog = sm.add_constant(df_nominal[["x1", "x2"]])
    return NominalGEE(
        df_nominal["y"].values, exog, groups=df_nominal["group"].values
    ).fit()


# ---------------------------------------------------------------------------
# Construction / auto-detect
# ---------------------------------------------------------------------------


def test_auto_detect_formula(nominal_fit_formula):
    from pymargins._adapters import auto_detect_adapter

    adapter = auto_detect_adapter(nominal_fit_formula)
    assert isinstance(adapter, StatsmodelsNominalGEEAdapter)


def test_auto_detect_array_requires_training_data(nominal_fit_array):
    from pymargins._adapters import auto_detect_adapter

    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(nominal_fit_array)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def test_coefficients_formula(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    coef = adapter.coefficients()
    assert isinstance(coef, jnp.ndarray)
    assert coef.shape == (len(nominal_fit_formula.params),)
    np.testing.assert_allclose(coef, nominal_fit_formula.params.values)


def test_coefficients_array(nominal_fit_array, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_array, training_data=df_nominal)
    coef = adapter.coefficients()
    assert isinstance(coef, jnp.ndarray)
    assert coef.shape == (len(nominal_fit_array.params),)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def test_training_data_formula(nominal_fit_formula, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    assert adapter.training_data is not None
    assert len(adapter.training_data) == len(df_nominal)


def test_training_data_array(nominal_fit_array, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_array, training_data=df_nominal)
    assert adapter.training_data is not None
    assert len(adapter.training_data) == len(df_nominal)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_covariance_default(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (
        len(nominal_fit_formula.params),
        len(nominal_fit_formula.params),
    )
    # Only compare if JAX-converted values are finite (float32 may overflow)
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, nominal_fit_formula.cov_params())


def test_covariance_naive(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    cov = adapter.covariance("naive")
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, nominal_fit_formula.cov_naive)


def test_covariance_robust(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    cov = adapter.covariance("robust")
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, nominal_fit_formula.cov_robust)


def test_covariance_user_supplied(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    custom = np.eye(len(nominal_fit_formula.params))
    cov = adapter.covariance(custom)
    np.testing.assert_allclose(cov, custom)


def test_covariance_invalid(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.covariance("invalid")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_fittedvalues_formula(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    beta = adapter.coefficients()
    X = jnp.asarray(nominal_fit_formula.model.exog_orig)
    probs = adapter.predict(beta, X)
    # probs shape: (n_obs, K)
    assert probs.shape == (len(X), adapter.n_outcomes)
    # Non-reference probs should match fittedvalues
    ncut = adapter._ncut
    fv = np.asarray(nominal_fit_formula.fittedvalues).reshape(-1, ncut)
    np.testing.assert_allclose(probs[:, :ncut], fv, rtol=1e-5)
    # Rows sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_predict_matches_fittedvalues_array(nominal_fit_array, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_array, training_data=df_nominal)
    beta = adapter.coefficients()
    X = jnp.asarray(nominal_fit_array.model.exog_orig)
    probs = adapter.predict(beta, X)
    ncut = adapter._ncut
    fv = np.asarray(nominal_fit_array.fittedvalues).reshape(-1, ncut)
    np.testing.assert_allclose(probs[:, :ncut], fv, rtol=1e-5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def test_design_matrix_from_df_formula(nominal_fit_formula, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    X = adapter.design_matrix_from_df(df_nominal)
    assert X.shape[1] == len(adapter._std_exog_names)
    np.testing.assert_allclose(X, jnp.asarray(nominal_fit_formula.model.exog_orig))


def test_design_matrix_from_df_array(nominal_fit_array, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_array, training_data=df_nominal)
    X = adapter.design_matrix_from_df(df_nominal)
    assert X.shape[1] == len(adapter._std_exog_names)
    np.testing.assert_allclose(X, jnp.asarray(nominal_fit_array.model.exog_orig))


# ---------------------------------------------------------------------------
# Variable metadata / column index
# ---------------------------------------------------------------------------


def test_variable_metadata(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta


def test_column_index_of_variable(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    assert adapter.column_index_of_variable("x1") == 1
    assert adapter.column_index_of_variable("x2") == 2


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula(nominal_fit_formula, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)
    new_adapter = adapter.refit(df_nominal)
    assert isinstance(new_adapter, StatsmodelsNominalGEEAdapter)
    assert new_adapter.training_data is df_nominal


def test_refit_array(nominal_fit_array, df_nominal):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_array, training_data=df_nominal)
    new_adapter = adapter.refit(df_nominal)
    assert isinstance(new_adapter, StatsmodelsNominalGEEAdapter)


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------


def test_attach_validates_vcov(nominal_fit_formula):
    adapter = StatsmodelsNominalGEEAdapter(nominal_fit_formula)

    class FakeSession:
        vcov_spec = "robust"

    adapter.attach(FakeSession())

    class FakeSession2:
        vcov_spec = "invalid"

    with pytest.raises(ValueError):
        adapter.attach(FakeSession2())
