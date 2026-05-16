"""Tests for Track B — Formula interface for formula-less adapters."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

from pymargins import Margins
from pymargins._formula import FormulaSpec, _has_derived_terms
from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_poly():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treat": rng.binomial(1, 0.5, size=n),
    })
    df["y"] = (
        10.0
        + 0.2 * df["age"]
        + 3.0 * df["treat"]
        + 0.01 * (df["age"] ** 2)
        + rng.standard_normal(n) * 2.0
    )
    return df


@pytest.fixture
def fit_array_ols_poly(df_poly):
    """Array-fit OLS with I(age**2) — the silently-wrong path without formula=."""
    df_poly["I(age ** 2)"] = df_poly["age"] ** 2
    X = pd.DataFrame({
        "Intercept": 1.0,
        "age": df_poly["age"],
        "treat": df_poly["treat"],
        "I(age ** 2)": df_poly["I(age ** 2)"],
    })
    y = df_poly["y"]
    return sm.OLS(y, X).fit()


@pytest.fixture
def fit_formula_ols_poly(df_poly):
    """Formula-fit OLS with I(age**2) — the correct reference path."""
    return smf.ols("y ~ age + treat + I(age**2)", data=df_poly).fit()


# ---------------------------------------------------------------------------
# FormulaSpec unit tests
# ---------------------------------------------------------------------------

def test_formula_spec_builds_and_round_trips(df_poly):
    spec = FormulaSpec("y ~ age + treat + I(age**2)", df_poly)
    assert "Intercept" in spec.exog_names
    assert "age" in spec.exog_names
    assert "I(age ** 2)" in spec.exog_names

    X = spec.get_model_matrix(df_poly)
    assert X.shape[1] == 4


def test_formula_spec_perturbation_propagates(df_poly):
    spec = FormulaSpec("y ~ age + treat + I(age**2)", df_poly)
    df_perturbed = df_poly.copy()
    df_perturbed["age"] = df_perturbed["age"] + 0.1

    X_orig = np.asarray(spec.get_model_matrix(df_poly))
    X_pert = np.asarray(spec.get_model_matrix(df_perturbed))

    # The I(age**2) column should change
    col_idx = spec.exog_names.index("I(age ** 2)")
    assert not np.allclose(X_orig[:, col_idx], X_pert[:, col_idx])


def test_formula_spec_verification_passes(fit_formula_ols_poly, df_poly):
    spec = FormulaSpec("y ~ age + treat + I(age**2)", df_poly)
    adapter = StatsmodelsOLSAdapter(fit_formula_ols_poly)
    # For formula-fit models, verification may skip because fittedvalues matches
    # The formula-spec should not raise
    spec.verify_against(adapter)


def test_formula_spec_verification_fails_on_wrong_formula(df_poly):
    spec = FormulaSpec("y ~ age + treat", df_poly)  # missing I(age**2)
    # Fit the full model
    fit = smf.ols("y ~ age + treat + I(age**2)", data=df_poly).fit()
    adapter = StatsmodelsOLSAdapter(fit)
    with pytest.raises(ValueError, match="Formula verification failed"):
        spec.verify_against(adapter)


# ---------------------------------------------------------------------------
# _has_derived_terms
# ---------------------------------------------------------------------------

def test_has_derived_terms_detects_interactions():
    assert _has_derived_terms(["age", "treat", "age:treat"])
    assert not _has_derived_terms(["age", "treat", "Intercept"])


def test_has_derived_terms_detects_polynomials():
    assert _has_derived_terms(["age", "I(age ** 2)"])
    assert not _has_derived_terms(["age"])


# ---------------------------------------------------------------------------
# Margins.from_formula constructor
# ---------------------------------------------------------------------------

def test_from_formula_classmethod(fit_array_ols_poly, df_poly):
    m = Margins.from_formula(
        fit_array_ols_poly,
        formula="y ~ age + treat + I(age**2)",
        data=df_poly,
    )
    assert m.adapter._formula_spec is not None


# ---------------------------------------------------------------------------
# dydx correctness: array-fit + formula should match formula-fit
# ---------------------------------------------------------------------------

def test_dydx_array_fit_with_formula_matches_formula_fit(
    fit_array_ols_poly, fit_formula_ols_poly, df_poly
):
    """B.6 Acceptance: array-fit model with formula= yields correct dydx."""
    m_array = Margins.linear_scale(
        fit_array_ols_poly,
        formula="y ~ age + treat + I(age**2)",
        data=df_poly,
        at="mean",
    )
    m_formula = Margins.linear_scale(fit_formula_ols_poly, at="mean")

    slope_array = m_array.dydx("age")
    slope_formula = m_formula.dydx("age")

    np.testing.assert_allclose(
        float(slope_array.estimate),
        float(slope_formula.estimate),
        rtol=1e-4,
    )


def test_dydx_array_fit_without_formula_warns_on_derived_terms(
    fit_array_ols_poly, df_poly
):
    """Column-selection fallback should warn when derived terms are present."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = Margins.linear_scale(fit_array_ols_poly, data=df_poly, at="mean")
        m.dydx("age")
        # The warning is raised during design_matrix_from_df inside dydx
        warn_msgs = [str(warning.message) for warning in w]
        assert any("column-selection fallback" in msg for msg in warn_msgs)


# ---------------------------------------------------------------------------
# predict() / evaluate() arity validation
# ---------------------------------------------------------------------------

def test_predict_transform_arity_validation(fit_formula_ols_poly):
    m = Margins.linear_scale(fit_formula_ols_poly, at="mean")
    with pytest.raises(ValueError, match="transform must accept at least"):
        m.predict(transform=lambda: 1.0)


def test_evaluate_compose_arity_validation(fit_formula_ols_poly):
    m = Margins.linear_scale(fit_formula_ols_poly, at="mean")
    with pytest.raises(ValueError, match="compose must accept at least"):
        m.evaluate(
            scenarios=[{"atexog": {"age": 50}}],
            compose=lambda: 1.0,
        )


# ---------------------------------------------------------------------------
# joint_test / run_test negative tests
# ---------------------------------------------------------------------------

def test_joint_test_neither_gradient_nor_draws(fit_formula_ols_poly):
    m = Margins.linear_scale(fit_formula_ols_poly, at="mean")
    pred = m.predict(atexog={"age": [40, 60]})
    # materialize drops gradient and draws
    mat = pred.materialize()
    with pytest.raises(ValueError, match="neither"):
        mat.joint_test()


def test_run_test_neither_gradient_nor_draws(fit_formula_ols_poly):
    m = Margins.linear_scale(fit_formula_ols_poly, at="mean")
    pred = m.predict()
    mat = pred.materialize()
    with pytest.raises(ValueError, match="Cannot run test"):
        mat.test()


# ---------------------------------------------------------------------------
# Track B wrinkles
# ---------------------------------------------------------------------------

def test_formula_spec_stateful_transform_center():
    """B.5 Wrinkle 1: centered terms reuse training mean, not recompute."""
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"x": rng.normal(100, 10, size=50)})
    train["y"] = train["x"] + rng.normal(size=50)
    spec = FormulaSpec("y ~ center(x)", train)

    # New data with different mean
    new_df = pd.DataFrame({"x": rng.normal(200, 10, size=10)})
    X_new = np.asarray(spec.get_model_matrix(new_df))
    # The centered column should be x - mean(train_x), not x - mean(new_x)
    expected_center = new_df["x"].values - train["x"].mean()
    col_idx = spec.exog_names.index("center(x)")
    np.testing.assert_allclose(X_new[:, col_idx], expected_center, rtol=1e-5)


def test_formula_spec_pins_categorical_levels():
    """B.5 Wrinkle 2: FormulaSpec keeps full training level set on subset data."""
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "cat": rng.choice(["A", "B", "C"], size=100),
        "y": rng.normal(size=100),
    })
    spec = FormulaSpec("y ~ C(cat)", train)
    # Full design has 3 columns: Intercept + 2 dummies (patsy drops one level)
    X_full = np.asarray(spec.get_model_matrix(train))
    n_cols_full = X_full.shape[1]
    assert n_cols_full >= 2

    # Subset missing level "C"
    subset = train[train["cat"] != "C"].copy()
    X_sub = np.asarray(spec.get_model_matrix(subset))
    # Should still have the same number of columns as full design
    assert X_sub.shape[1] == n_cols_full
    # If C was not the reference level, its dummy column should be all zeros
    c_dummy_name = "C(cat)[T.C]"
    if c_dummy_name in spec.exog_names:
        c_col = spec.exog_names.index(c_dummy_name)
        assert np.all(X_sub[:, c_col] == 0)
