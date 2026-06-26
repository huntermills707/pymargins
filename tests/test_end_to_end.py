"""End-to-end smoke tests for the GComputation pipeline.

See IMPLEMENTATION_GUIDE.md §0.4 and §0.5.
"""

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_logit():
    """Synthetic data for a logit model."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )
    eta = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"] + 0.3 * (df["sex"] == "M")
    prob = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = (rng.uniform(size=n) < prob).astype(float)
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "y ~ age + treatment + C(sex)",
        data=df_logit,
        family=sm.families.Binomial(),
    ).fit()


# ---------------------------------------------------------------------------
# 0.4 — Wire up GComputation to the adapter
# ---------------------------------------------------------------------------


def test_gcomputation_construction_log_scale(fit_logit):
    est = GComputation(fit_logit, scale="log")
    assert est.plan is not None
    assert est.plan.scale == "log"
    assert est.plan.method_resolved == "delta"


def test_gcomputation_summary(fit_logit):
    est = GComputation(fit_logit, scale="log", vcov="HC3")
    result = est.predict()
    summary = result.summary()
    assert "Graph Result" in summary
    assert est.plan.scale == "log"
    assert est.plan.vcov == "HC3"


def test_gcomputation_diagnose(fit_logit):
    est = GComputation(fit_logit, scale="log", method="auto")
    desc = est.plan.describe()
    assert "log" in desc
    assert est.plan.method_resolved in ("delta", "simulation")


# ---------------------------------------------------------------------------
# 0.5 — Smoke test: relative risk via log_scale
# ---------------------------------------------------------------------------


def test_relative_risk_contrast(fit_logit):
    """Compute a relative risk: exp(log(p_treat=1) - log(p_treat=0))."""
    est = GComputation(fit_logit, scale="log")

    rr = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}, "label": "treated"},
            {"atexog": {"treatment": 0}, "label": "control"},
        ],
        contrasts=[+1, -1],
    )

    # Result structure
    assert float(rr.estimate) > 0.0
    assert float(rr.std_error) >= 0.0
    assert float(rr.conf_int_lower) > 0.0
    assert float(rr.conf_int_upper) > float(rr.conf_int_lower)
    assert rr.level == 0.95

    # On log scale, RR should be around exp(0.8) ≈ 2.2
    assert 1.0 < float(rr.estimate) < 5.0


def test_prediction_at_typical(fit_logit):
    """Predict at representative values."""
    est = GComputation(fit_logit, at="typical")

    pred = est.predict(atexog={"treatment": [0, 1]})

    # Should produce 2 rows (one per treatment value)
    assert pred.estimate.shape == (2,)
    # Probabilities should be in (0, 1)
    assert np.all(pred.estimate > 0.0)
    assert np.all(pred.estimate < 1.0)


def test_ame_continuous(fit_logit):
    """Average marginal effect of age."""
    est = GComputation(fit_logit, at="overall")

    ame = est.dydx("age")

    assert np.isfinite(float(ame.estimate))
    assert np.isfinite(float(ame.std_error))
    assert float(ame.conf_int_lower) < float(ame.conf_int_upper)


def test_prediction_with_over(fit_logit):
    """Subgroup predictions by sex."""
    est = GComputation(fit_logit, at="typical")

    pred = est.predict(atexog={"treatment": 1}, over="sex")

    # Should produce 2 rows (M and F)
    assert pred.estimate.shape == (2,)


def test_contrast_vector_named(fit_logit):
    """Multiple named contrasts with joint inference."""
    est = GComputation(fit_logit, at="typical")

    result = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "sex": "M"}, "label": "TM"},
            {"atexog": {"treatment": 0, "sex": "M"}, "label": "CM"},
            {"atexog": {"treatment": 1, "sex": "F"}, "label": "TF"},
            {"atexog": {"treatment": 0, "sex": "F"}, "label": "CF"},
        ],
        contrasts={
            "treatment_effect_male": [+1, -1, 0, 0],
            "treatment_effect_female": [0, 0, +1, -1],
        },
    )

    # Vector result with 2 rows
    assert result.estimate.shape == (2,)
    assert len(result.estimand_metadata.get("labels", [])) == 2


def test_result_test_method(fit_logit):
    """Test a null hypothesis on a result."""
    est = GComputation(fit_logit, scale="log")

    rr = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )

    test = rr.test(value=1.0)  # H0: RR = 1
    assert test.method == "wald"
    assert np.isfinite(float(test.statistic))
    assert 0.0 <= float(test.pvalue) <= 1.0


def test_result_joint_test(fit_logit):
    """Joint test on a vector result."""
    est = GComputation(fit_logit, at="typical")

    result = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "sex": "M"}},
            {"atexog": {"treatment": 0, "sex": "M"}},
            {"atexog": {"treatment": 1, "sex": "F"}},
            {"atexog": {"treatment": 0, "sex": "F"}},
        ],
        contrasts={
            "male": [+1, -1, 0, 0],
            "female": [0, 0, +1, -1],
        },
    )

    jt = result.joint_test(value=np.zeros(2))
    assert jt.method == "joint_wald"
    assert np.isfinite(float(jt.statistic))
    assert 0.0 <= float(jt.pvalue) <= 1.0
    assert jt.df == 2


# ---------------------------------------------------------------------------
# Additional coverage for testing gaps
# ---------------------------------------------------------------------------


def test_evaluate_nnt(fit_logit):
    """Nonlinear composition: NNT = 1/(p_control - p_treated)."""
    est = GComputation(fit_logit, at="typical")

    nnt = est.evaluate(
        scenarios=[
            {"atexog": {"treatment": 0}, "label": "control"},
            {"atexog": {"treatment": 1}, "label": "treated"},
        ],
        compose=lambda p: 1.0 / (p[0] - p[1]),
    )
    assert np.isfinite(float(nnt.estimate))
    assert float(nnt.conf_int_lower) < float(nnt.conf_int_upper)


def test_simulation_method(fit_logit):
    """Explicit simulation method should produce valid CIs."""
    est = GComputation(
        fit_logit, at="typical", method="simulation", n_sim=2000, seed=42
    )

    rr = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert rr.method == "simulation"
    assert np.isfinite(float(rr.estimate))
    assert float(rr.conf_int_lower) < float(rr.conf_int_upper)
    assert rr.draws is not None
    assert rr.draws.shape == (2000,)


def test_bootstrap_method(fit_logit):
    """Bootstrap method should produce valid CIs via refit."""
    est = GComputation(fit_logit, at="typical", method="bootstrap", B=50, seed=42)

    rd = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


def test_predict_with_transform(fit_logit):
    """Per-row transform applied before aggregation."""
    est = GComputation(fit_logit, at="overall")

    pred = est.predict(atexog={"treatment": 1}, transform=lambda mu: mu**2)
    assert np.isfinite(float(pred.estimate))
    # Squared probabilities should be smaller than raw probabilities
    pred_raw = est.predict(atexog={"treatment": 1})
    assert float(pred.estimate) < float(pred_raw.estimate)


def test_dydx_on_binary_raises(fit_logit):
    """dydx on a binary variable must raise ValueError."""
    est = GComputation(fit_logit, at="overall")
    with pytest.raises(ValueError, match="binary"):
        est.dydx("treatment")


def test_dydx_on_categorical_raises(fit_logit):
    """dydx on a categorical variable must raise ValueError."""
    est = GComputation(fit_logit, at="overall")
    with pytest.raises(ValueError, match="categorical"):
        est.dydx("sex")


def test_scaled_transforms_result(fit_logit):
    """scaled() should multiply estimate and CI bounds by the given factor."""
    est = GComputation(fit_logit, at="overall")
    pred = est.predict(atexog={"age": [40, 60]})
    scaled = pred.scaled(by=100, units="%")
    np.testing.assert_allclose(scaled.estimate, pred.estimate * 100)
    np.testing.assert_allclose(scaled.conf_int_lower, pred.conf_int_lower * 100)
    np.testing.assert_allclose(scaled.conf_int_upper, pred.conf_int_upper * 100)
    assert scaled.estimand_metadata.get("units") == "%"


def test_dydx_with_over(fit_logit):
    """dydx with over= should produce one row per group."""
    est = GComputation(fit_logit, at="overall")
    result = est.dydx("age", over="sex")
    assert result.estimate.shape == (2,)


def test_empty_training_data_raises():
    """Empty training data should raise a clear error during model fitting or in pymargins."""
    df_empty = pd.DataFrame(
        {
            "x": pd.Series([], dtype=float),
            "y": pd.Series([], dtype=float),
        }
    )
    with pytest.raises(ValueError):
        smf.glm("y ~ x", data=df_empty, family=sm.families.Binomial()).fit()


def test_single_observation_ci_collapses():
    """With a single observation, bootstrap should produce zero SE and collapsed CIs."""
    df_one = pd.DataFrame({"x": [1.0], "y": [2]})
    fit = smf.glm("y ~ x", data=df_one, family=sm.families.Poisson()).fit()
    est = GComputation(fit, method="bootstrap", B=20, seed=42)
    pred = est.predict()
    assert np.isclose(float(pred.std_error), 0.0, atol=1e-12)
    assert np.isclose(float(pred.conf_int_lower), float(pred.estimate))
    assert np.isclose(float(pred.conf_int_upper), float(pred.estimate))
