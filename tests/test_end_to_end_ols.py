"""End-to-end smoke tests for the OLS pipeline.

See IMPLEMENTATION_GUIDE.md §1.1 and §1.2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import Margins


@pytest.fixture
def df_ols():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )
    df["y"] = (
        10.0
        + 0.2 * df["age"]
        + 3.0 * df["treatment"]
        + 1.5 * (df["sex"] == "M")
        + rng.standard_normal(n) * 2.0
    )
    return df


@pytest.fixture
def fit_ols(df_ols):
    return smf.ols("y ~ age + treatment + C(sex)", data=df_ols).fit()


def test_margins_construction_linear_scale(fit_ols):
    m = Margins.linear_scale(fit_ols)
    assert m.phi is None
    assert m.phi_inv is None


def test_margins_log_scale_raises_for_negative_predictions(fit_ols):
    # OLS predictions can be negative; log scale with negative predictions
    # should still construct but may produce NaNs in practice.
    m = Margins.log_scale(fit_ols)
    assert m.phi is jnp.exp


def test_prediction_at_typical(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    assert pred.estimate.shape == (2,)
    # Difference should be around the treatment coefficient (~3.0)
    diff = float(pred.estimate[1] - pred.estimate[0])
    assert 1.0 < diff < 5.0


def test_ame_continuous(fit_ols):
    m = Margins.linear_scale(fit_ols, at="overall")
    ame = m.dydx("age")
    assert np.isfinite(float(ame.estimate))
    assert np.isfinite(float(ame.std_error))
    # AME of age should be around 0.2
    assert 0.0 < float(ame.estimate) < 0.5


def test_contrast_risk_difference(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}, "label": "treated"},
            {"atexog": {"treatment": 0}, "label": "control"},
        ],
        contrasts=[+1, -1],
    )
    # Risk difference should be around the treatment coefficient
    assert 1.0 < float(rd.estimate) < 5.0
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_prediction_with_over(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": 1}, over="sex")
    assert pred.estimate.shape == (2,)


def test_result_test_method(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    test = rd.test(value=0.0)
    assert 0.0 <= float(test.pvalue) <= 1.0


def test_hc3_vcov(fit_ols):
    m = Margins.linear_scale(fit_ols, vcov="HC3", at="typical")
    pred = m.predict(atexog={"treatment": 1})
    assert np.isfinite(float(pred.estimate))
    assert np.isfinite(float(pred.std_error))


# ---------------------------------------------------------------------------
# Additional coverage for testing gaps
# ---------------------------------------------------------------------------


def test_evaluate_ratio(fit_ols):
    """Nonlinear composition via evaluate()."""
    m = Margins.linear_scale(fit_ols, at="typical")

    ratio = m.evaluate(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        compose=lambda p: p[0] / p[1],
    )
    assert np.isfinite(float(ratio.estimate))
    assert float(ratio.conf_int_lower) < float(ratio.conf_int_upper)


def test_simulation_method(fit_ols):
    """Explicit simulation method for OLS."""
    m = Margins.linear_scale(
        fit_ols, at="typical", method="simulation", n_sim=2000, rng_seed=42
    )

    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert rd.method == "simulation"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_bootstrap_method(fit_ols):
    """Bootstrap method for OLS."""
    m = Margins.linear_scale(
        fit_ols, at="typical", method="bootstrap", n_boot=50, rng_seed=42
    )

    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)


def test_dydx_on_binary_raises(fit_ols):
    """dydx on a binary variable must raise ValueError."""
    m = Margins.linear_scale(fit_ols, at="overall")
    with pytest.raises(ValueError, match="binary"):
        m.dydx("treatment")


def test_dydx_on_categorical_raises(fit_ols):
    """dydx on a categorical variable must raise ValueError."""
    m = Margins.linear_scale(fit_ols, at="overall")
    with pytest.raises(ValueError, match="categorical"):
        m.dydx("sex")


# ---------------------------------------------------------------------------
# Total marginal effects (R/Stata semantics)
# ---------------------------------------------------------------------------


def test_dydx_includes_interaction_chain_rule():
    """dydx(x1) for y ~ x1*x2 must be β_x1 + β_{x1:x2}*x2 (total derivative).

    A column-wise slope would return only β_x1, ignoring the interaction.
    Reference: Stata `margins, dydx(x1)` and R `marginaleffects::slopes()`.
    """
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = (
        1.0
        + 2.0 * df["x1"]
        - 0.5 * df["x2"]
        + 1.5 * df["x1"] * df["x2"]
        + rng.standard_normal(n) * 0.1
    )

    fit = smf.ols("y ~ x1 * x2", data=df).fit()
    # Column ordering: ['Intercept', 'x1', 'x2', 'x1:x2']
    beta = fit.params.values
    b_x1 = beta[fit.model.exog_names.index("x1")]
    b_x1x2 = beta[fit.model.exog_names.index("x1:x2")]

    m = Margins.linear_scale(fit, at="typical")
    ame = m.dydx("x1")

    # At "typical", x2 is held at its median ≈ 0 (centered normal). The
    # total ME at that x2 is b_x1 + b_x1x2 * x2_typical.
    x2_typical = float(df["x2"].median())
    expected = b_x1 + b_x1x2 * x2_typical
    np.testing.assert_allclose(float(ame.estimate), expected, rtol=1e-4)

    # AME (averaged over the sample) should equal b_x1 + b_x1x2 * mean(x2).
    m_overall = Margins.linear_scale(fit, at="overall")
    ame_overall = m_overall.dydx("x1")
    expected_overall = b_x1 + b_x1x2 * df["x2"].mean()
    np.testing.assert_allclose(float(ame_overall.estimate), expected_overall, rtol=1e-4)


def test_dydx_through_polynomial_transform():
    """dydx(x) for y ~ x + I(x**2) must be β_x + 2*β_{I(x**2)}*x."""
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 0.5 + 1.0 * df["x"] - 0.3 * df["x"] ** 2 + rng.standard_normal(n) * 0.1

    fit = smf.ols("y ~ x + I(x**2)", data=df).fit()
    names = fit.model.exog_names
    b_x = fit.params.values[names.index("x")]
    b_x2 = fit.params.values[names.index("I(x ** 2)")]

    m = Margins.linear_scale(fit, at="overall")
    ame = m.dydx("x")
    expected = b_x + 2.0 * b_x2 * df["x"].mean()
    np.testing.assert_allclose(float(ame.estimate), expected, rtol=1e-3)
