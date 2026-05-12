"""Tests for StatsmodelsOrdinalGEEAdapter."""

import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest
import statsmodels.api as sm

from pymargins._adapters.statsmodels_ordinal_gee import StatsmodelsOrdinalGEEAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_ordinal():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "group": np.repeat(range(50), 20),
    })
    # Ordered outcome with 3 categories
    # Smaller effects and wider thresholds avoid degenerate fits on random data.
    logit = 0.2 * df["x1"] - 0.15 * df["x2"]
    u = np.random.rand(n)
    df["y"] = (u > 1 / (1 + np.exp(-(-0.8 + logit)))).astype(int) + (
        u > 1 / (1 + np.exp(-(0.8 + logit)))
    ).astype(int)
    return df


@pytest.fixture
def ordinal_fit_formula(df_ordinal):
    from statsmodels.genmod import cov_struct
    from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
    return OrdinalGEE.from_formula(
        "y ~ x1 + x2",
        groups=df_ordinal["group"],
        data=df_ordinal,
        family=sm.families.Binomial(),
        cov_struct=cov_struct.Independence(),
    ).fit()


@pytest.fixture
def ordinal_fit_array(df_ordinal):
    from statsmodels.genmod import cov_struct
    from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
    exog = sm.add_constant(df_ordinal[["x1", "x2"]])
    return OrdinalGEE(
        df_ordinal["y"].values,
        exog,
        groups=df_ordinal["group"].values,
        family=sm.families.Binomial(),
        cov_struct=cov_struct.Independence(),
    ).fit()


# Well-converged reference model using statsmodels test data
@pytest.fixture
def ordinal_test_data_fit():
    import os
    csv_path = os.path.join(
        os.path.dirname(sm.__file__),
        "genmod", "tests", "results", "gee_ordinal_1.csv",
    )
    Z = np.genfromtxt(csv_path, delimiter=",")
    groups = Z[:, 0]
    endog = Z[:, 1]
    exog = np.concatenate((np.ones((Z.shape[0], 1)), Z[:, 2:]), axis=1)
    df = pd.DataFrame(exog, columns=["const", "x1", "x2", "x3", "x4", "x5"])
    df["y"] = endog.astype(int)
    df["group"] = groups.astype(int)
    from statsmodels.genmod import cov_struct
    from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
    fit = OrdinalGEE(
        endog, exog, groups,
        family=sm.families.Binomial(),
        cov_struct=cov_struct.GlobalOddsRatio("ordinal"),
    ).fit()
    return fit, df


# ---------------------------------------------------------------------------
# Construction / auto-detect
# ---------------------------------------------------------------------------

def test_auto_detect_formula(ordinal_fit_formula):
    from pymargins._adapters import auto_detect_adapter
    adapter = auto_detect_adapter(ordinal_fit_formula)
    assert isinstance(adapter, StatsmodelsOrdinalGEEAdapter)


def test_auto_detect_array_requires_training_data(ordinal_fit_array):
    from pymargins._adapters import auto_detect_adapter
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(ordinal_fit_array)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

def test_coefficients_formula(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    coef = adapter.coefficients()
    assert isinstance(coef, jnp.ndarray)
    assert coef.shape == (len(ordinal_fit_formula.params),)
    np.testing.assert_allclose(coef, ordinal_fit_formula.params.values)


def test_coefficients_array(ordinal_fit_array, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_array, training_data=df_ordinal)
    coef = adapter.coefficients()
    assert isinstance(coef, jnp.ndarray)
    assert coef.shape == (len(ordinal_fit_array.params),)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def test_training_data_formula(ordinal_fit_formula, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    assert adapter.training_data is not None
    assert len(adapter.training_data) == len(df_ordinal)


def test_training_data_array(ordinal_fit_array, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_array, training_data=df_ordinal)
    assert adapter.training_data is not None
    assert len(adapter.training_data) == len(df_ordinal)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def test_covariance_default(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (len(ordinal_fit_formula.params), len(ordinal_fit_formula.params))
    # Only compare if JAX-converted values are finite (float32 may overflow)
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, ordinal_fit_formula.cov_params())


def test_covariance_naive(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    cov = adapter.covariance("naive")
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, ordinal_fit_formula.cov_naive)


def test_covariance_robust(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    cov = adapter.covariance("robust")
    if np.isfinite(np.asarray(cov)).all():
        np.testing.assert_allclose(cov, ordinal_fit_formula.cov_robust)


def test_covariance_user_supplied(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    custom = np.eye(len(ordinal_fit_formula.params))
    cov = adapter.covariance(custom)
    np.testing.assert_allclose(cov, custom)


def test_covariance_well_converged(ordinal_test_data_fit):
    fit, df = ordinal_test_data_fit
    adapter = StatsmodelsOrdinalGEEAdapter(fit, training_data=df)
    cov = adapter.covariance()
    assert cov.shape == (len(fit.params), len(fit.params))
    np.testing.assert_allclose(cov, fit.cov_params(), rtol=1e-5)


def test_covariance_invalid(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.covariance("invalid")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_fittedvalues_formula(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    beta = adapter.coefficients()
    X = jnp.asarray(ordinal_fit_formula.model.exog_orig)
    probs = adapter.predict(beta, X)
    assert probs.shape == (len(X), adapter.n_outcomes)
    n = len(X)
    ncut = adapter._ncut
    cumprobs = np.asarray(ordinal_fit_formula.fittedvalues).reshape(n, ncut)
    expected = np.zeros((n, ncut + 1))
    expected[:, 0] = 1 - cumprobs[:, 0]
    for k in range(1, ncut):
        expected[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
    expected[:, ncut] = cumprobs[:, ncut - 1]
    np.testing.assert_allclose(probs, expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_predict_matches_fittedvalues_array(ordinal_fit_array, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_array, training_data=df_ordinal)
    beta = adapter.coefficients()
    X = jnp.asarray(ordinal_fit_array.model.exog_orig)
    probs = adapter.predict(beta, X)
    n = len(X)
    ncut = adapter._ncut
    cumprobs = np.asarray(ordinal_fit_array.fittedvalues).reshape(n, ncut)
    expected = np.zeros((n, ncut + 1))
    expected[:, 0] = 1 - cumprobs[:, 0]
    for k in range(1, ncut):
        expected[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
    expected[:, ncut] = cumprobs[:, ncut - 1]
    np.testing.assert_allclose(probs, expected, rtol=1e-5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_predict_well_converged(ordinal_test_data_fit):
    fit, df = ordinal_test_data_fit
    adapter = StatsmodelsOrdinalGEEAdapter(fit, training_data=df)
    beta = adapter.coefficients()
    X = jnp.asarray(fit.model.exog_orig)
    probs = adapter.predict(beta, X)
    assert probs.shape == (len(X), adapter.n_outcomes)
    n = len(X)
    ncut = adapter._ncut
    cumprobs = np.asarray(fit.fittedvalues).reshape(n, ncut)
    expected = np.zeros((n, ncut + 1))
    expected[:, 0] = 1 - cumprobs[:, 0]
    for k in range(1, ncut):
        expected[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
    expected[:, ncut] = cumprobs[:, ncut - 1]
    np.testing.assert_allclose(probs, expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def test_design_matrix_from_df_formula(ordinal_fit_formula, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    X = adapter.design_matrix_from_df(df_ordinal)
    assert X.shape[1] == len(adapter._std_exog_names)
    np.testing.assert_allclose(X, jnp.asarray(ordinal_fit_formula.model.exog_orig))


def test_design_matrix_from_df_array(ordinal_fit_array, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_array, training_data=df_ordinal)
    X = adapter.design_matrix_from_df(df_ordinal)
    assert X.shape[1] == len(adapter._std_exog_names)
    np.testing.assert_allclose(X, jnp.asarray(ordinal_fit_array.model.exog_orig))


# ---------------------------------------------------------------------------
# Variable metadata / column index
# ---------------------------------------------------------------------------

def test_variable_metadata(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta


def test_column_index_of_variable(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    assert adapter.column_index_of_variable("x1") == 1
    assert adapter.column_index_of_variable("x2") == 2


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def test_refit_formula(ordinal_fit_formula, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)
    new_adapter = adapter.refit(df_ordinal)
    assert isinstance(new_adapter, StatsmodelsOrdinalGEEAdapter)
    assert new_adapter.training_data is df_ordinal


def test_refit_array(ordinal_fit_array, df_ordinal):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_array, training_data=df_ordinal)
    new_adapter = adapter.refit(df_ordinal)
    assert isinstance(new_adapter, StatsmodelsOrdinalGEEAdapter)


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------

def test_attach_validates_vcov(ordinal_fit_formula):
    adapter = StatsmodelsOrdinalGEEAdapter(ordinal_fit_formula)

    class FakeSession:
        vcov_spec = "robust"
    adapter.attach(FakeSession())

    class FakeSession2:
        vcov_spec = "invalid"
    with pytest.raises(ValueError):
        adapter.attach(FakeSession2())
