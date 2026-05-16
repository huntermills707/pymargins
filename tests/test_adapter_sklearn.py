"""Tests for Track C — sklearn BootstrapOnly adapter."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from pymargins import Margins
from pymargins._adapters.sklearn_bootstrap import SklearnBootstrapAdapter


@pytest.fixture
def df_sklearn():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treat": rng.binomial(1, 0.5, size=n),
    })
    df["age_sq"] = df["age"] ** 2
    df["y"] = (
        10.0
        + 0.2 * df["age"]
        + 0.001 * df["age_sq"]
        + 3.0 * df["treat"]
        + rng.normal(size=n)
    )
    return df


@pytest.fixture
def fit_sklearn_linear(df_sklearn):
    X = df_sklearn[["age", "treat", "age_sq"]]
    y = df_sklearn["y"]
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Constructor and basic properties
# ---------------------------------------------------------------------------

def test_adapter_coefficients_is_dummy(fit_sklearn_linear):
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    beta = adapter.coefficients()
    assert beta.shape == (1,)
    assert float(beta[0]) == 0.0


def test_adapter_predict_ignores_beta(fit_sklearn_linear, df_sklearn):
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    beta = adapter.coefficients()
    preds = adapter.predict(beta, np.asarray(X[:5]))
    expected = model.predict(X[:5])
    np.testing.assert_allclose(np.asarray(preds), expected, rtol=1e-5)


def test_adapter_design_matrix_column_selection(fit_sklearn_linear, df_sklearn):
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    X_mat = adapter.design_matrix_from_df(df_sklearn)
    np.testing.assert_allclose(np.asarray(X_mat), np.asarray(X), rtol=1e-5)


# ---------------------------------------------------------------------------
# Formula interface
# ---------------------------------------------------------------------------

def test_adapter_with_formula_correct_dydx(fit_sklearn_linear, df_sklearn):
    """B.6 Acceptance: sklearn with formula= yields correct dydx."""
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(
        model,
        formula="y ~ 0 + age + treat + I(age**2)",
        data=df_sklearn,
        target_name="y",
    )
    m = Margins(model, adapter=adapter, method="bootstrap", n_boot=30, rng_seed=42)
    slope = m.dydx("age")
    # Expected slope at mean age: 0.2 + 0.001 * 2 * mean(age)
    expected = 0.2 + 0.001 * 2 * df_sklearn["age"].mean()
    np.testing.assert_allclose(float(slope.estimate), expected, rtol=0.15)
    assert float(slope.std_error) > 0


def test_adapter_without_formula_raises_on_derived_terms(df_sklearn):
    """Without formula=, dydx() on derived terms raises ValueError."""
    # Fit a model on data with a patsy-style derived column name
    df = df_sklearn.copy()
    df["I(age ** 2)"] = df["age"] ** 2
    X = df[["age", "treat", "I(age ** 2)"]]
    y = df["y"]
    model = LinearRegression()
    model.fit(X, y)
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    m = Margins(model, adapter=adapter, method="bootstrap", n_boot=10, rng_seed=42)
    with pytest.raises(ValueError, match="derived terms"):
        m.dydx("age")


# ---------------------------------------------------------------------------
# Bootstrap inference
# ---------------------------------------------------------------------------

def test_bootstrap_predict(fit_sklearn_linear, df_sklearn):
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    m = Margins(model, adapter=adapter, method="bootstrap", n_boot=20, rng_seed=42)
    pred = m.predict(atexog={"age": 50, "treat": 1})
    assert np.isfinite(float(pred.estimate))
    assert np.isfinite(float(pred.std_error))
    lower, upper = pred.conf_int()
    assert lower < upper


# ---------------------------------------------------------------------------
# B.4 verification
# ---------------------------------------------------------------------------

def test_formula_verification_catches_intercept_mismatch(df_sklearn):
    """If model was trained WITH intercept but formula suppresses it, raise."""
    df = df_sklearn.copy()
    X = df[["age", "treat", "age_sq"]]
    X.insert(0, "Intercept", 1.0)
    y = df["y"]
    model = LinearRegression()
    model.fit(X, y)

    adapter = SklearnBootstrapAdapter(
        model,
        formula="y ~ 0 + age + treat + I(age**2)",
        data=df,
        target_name="y",
    )
    with pytest.raises(ValueError, match="Formula verification failed"):
        Margins(model, adapter=adapter, method="bootstrap", n_boot=10)


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def test_refit_produces_new_model(fit_sklearn_linear, df_sklearn):
    model, X, y = fit_sklearn_linear
    adapter = SklearnBootstrapAdapter(model, X_train=X, y_train=y)
    idx = np.random.choice(len(df_sklearn), size=len(df_sklearn), replace=True)
    new_adapter = adapter.refit(adapter.training_data, index=idx)
    assert new_adapter.model is not model
    preds = new_adapter.predict(new_adapter.coefficients(), np.asarray(X[:5]))
    assert preds.shape == (5,)
