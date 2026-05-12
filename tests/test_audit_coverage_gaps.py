"""Tests for CODE_AUDIT coverage gaps."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins


@pytest.fixture
def df_logit():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treatment": rng.binomial(1, 0.5, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })
    lp = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"] + 0.3 * (df["sex"] == "M")
    df["outcome"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "outcome ~ age + treatment + C(sex)",
        data=df_logit,
        family=sm.families.Binomial(),
    ).fit()


@pytest.fixture
def df_ols():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treatment": rng.binomial(1, 0.5, size=n),
        "group": rng.choice(["A", "B", "C"], size=n),
    })
    df["y"] = (
        10.0
        + 0.2 * df["age"]
        + 3.0 * df["treatment"]
        + 1.5 * (df["group"] == "A")
        + rng.standard_normal(n) * 2.0
    )
    return df


@pytest.fixture
def fit_ols(df_ols):
    return smf.ols("y ~ age + treatment + C(group)", data=df_ols).fit()


# ---------------------------------------------------------------------------
# Materialize tests
# ---------------------------------------------------------------------------

def test_materialized_to_frame_has_no_p_value(fit_ols):
    m = Margins.linear_scale(fit_ols)
    pred = m.predict(atexog={"treatment": [0, 1]})
    mat = pred.materialize()
    frame = mat.to_frame()
    assert "statistic" not in frame.columns
    assert "p_value" not in frame.columns


def test_materialized_test_raises(fit_ols):
    m = Margins.linear_scale(fit_ols)
    pred = m.predict(atexog={"treatment": [0, 1]})
    mat = pred.materialize()
    with pytest.raises(ValueError):
        mat.test()


# ---------------------------------------------------------------------------
# scaled() test
# ---------------------------------------------------------------------------

def test_scaled_multiplies_estimate_and_ci(fit_ols):
    m = Margins.linear_scale(fit_ols)
    pred = m.predict(atexog={"treatment": [0, 1]})
    scaled = pred.scaled(by=100, units="%")
    np.testing.assert_array_almost_equal(
        scaled.estimate, pred.estimate * 100
    )
    np.testing.assert_array_almost_equal(
        scaled.conf_int_lower, pred.conf_int_lower * 100
    )
    np.testing.assert_array_almost_equal(
        scaled.conf_int_upper, pred.conf_int_upper * 100
    )


# ---------------------------------------------------------------------------
# One-sided test alternatives
# ---------------------------------------------------------------------------

def test_test_one_sided_alternatives(fit_ols):
    m = Margins.linear_scale(fit_ols)
    pred = m.predict()
    two_sided = pred.test(alternative="two-sided")
    greater = pred.test(alternative="greater")
    less = pred.test(alternative="less")
    # One-sided p-values should satisfy: greater + less ≈ 1 (for z-tests)
    # when the estimate is away from zero. At minimum they must differ.
    assert float(greater.pvalue) != float(less.pvalue)
    # All p-values in [0, 1]
    assert 0 <= float(greater.pvalue) <= 1
    assert 0 <= float(less.pvalue) <= 1
    # For a positive estimate, greater p-value <= two-sided, less >= two-sided
    est = float(pred.estimate)
    if est > 0:
        assert float(greater.pvalue) <= float(two_sided.pvalue)
        assert float(less.pvalue) >= float(two_sided.pvalue)


# ---------------------------------------------------------------------------
# dydx with over
# ---------------------------------------------------------------------------

def test_dydx_with_over_produces_group_rows(df_ols, fit_ols):
    m = Margins.linear_scale(fit_ols)
    result = m.dydx("age", over="group")
    assert result.estimate.size == 3  # one per group
    frame = result.to_frame()
    assert "over" in frame.columns
    assert "over_value" in frame.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_observation_ci_collapses():
    df = pd.DataFrame({
        "x": [1.0],
        "y": [2.0],
    })
    model = smf.ols("y ~ x", data=df).fit()
    m = Margins.linear_scale(model)
    pred = m.predict()
    # With 1 observation, covariance is singular (infinite SE).
    # The library should handle this gracefully without crashing.
    assert np.isinf(float(pred.std_error)) or np.isnan(float(pred.std_error))
    assert np.isinf(float(pred.conf_int_lower)) or np.isnan(float(pred.conf_int_lower))
    assert np.isinf(float(pred.conf_int_upper)) or np.isnan(float(pred.conf_int_upper))


# ---------------------------------------------------------------------------
# Contrasts with ndarray
# ---------------------------------------------------------------------------

def test_contrasts_with_ndarray_matrix(fit_ols):
    m = Margins.linear_scale(fit_ols)
    contrasts = np.array([[1, -1, 0], [1, 0, -1]])
    result = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 0}},
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0.5}},
        ],
        contrasts=contrasts,
    )
    assert result.estimate.size == 2


# ---------------------------------------------------------------------------
# diagnose() with non-DataFrame training data
# ---------------------------------------------------------------------------

def test_diagnose_non_dataframe_raises(fit_ols):
    m = Margins.linear_scale(fit_ols)
    # Mock adapter.training_data to return a non-DataFrame
    from unittest.mock import patch
    with patch.object(m.adapter.__class__, "training_data", new_callable=lambda: property(lambda self: np.array([1, 2, 3]))):
        with pytest.raises(TypeError, match="diagnose.*requires base data"):
            m.diagnose()


# ---------------------------------------------------------------------------
# Fallback triggered appears in to_frame
# ---------------------------------------------------------------------------

def test_fallback_triggered_in_to_frame(fit_logit):
    # log_scale on a logit model creates a highly non-linear composition
    # that triggers the delta-method fallback even with kappa_threshold=0.0.
    m = Margins.log_scale(fit_logit, at="typical", kappa_threshold=0.0)
    pred = m.predict(atexog={"treatment": 1})
    assert pred.fallback_triggered
    frame = pred.to_frame()
    assert "fallback_triggered" in frame.columns
    assert "fallback_reason" in frame.columns
    assert frame["fallback_triggered"].iloc[0]


# ---------------------------------------------------------------------------
# Multi-outcome result to_frame
# ---------------------------------------------------------------------------

def test_multi_outcome_to_frame():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    # Three-category outcome
    lp0 = 1.0 + 0.5 * df["x1"] + 0.3 * df["x2"]
    lp1 = 0.5 + 0.2 * df["x1"] - 0.1 * df["x2"]
    denom = 1 + np.exp(lp0) + np.exp(lp1)
    p0 = 1 / denom
    p1 = np.exp(lp0) / denom
    p2 = np.exp(lp1) / denom
    probs = np.column_stack([p0, p1, p2])
    df["y"] = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    fit = smf.mnlogit("y ~ x1 + x2", data=df).fit(disp=False)
    m = Margins.linear_scale(fit)
    pred = m.predict()
    frame = pred.to_frame()
    assert len(frame) == 3  # one row per outcome
    assert "label" in frame.columns


# ---------------------------------------------------------------------------
# outcome() end-to-end with real multi-outcome model
# ---------------------------------------------------------------------------

def test_outcome_end_to_end():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    lp0 = 1.0 + 0.5 * df["x1"] + 0.3 * df["x2"]
    lp1 = 0.5 + 0.2 * df["x1"] - 0.1 * df["x2"]
    denom = 1 + np.exp(lp0) + np.exp(lp1)
    p0 = 1 / denom
    p1 = np.exp(lp0) / denom
    p2 = np.exp(lp1) / denom
    probs = np.column_stack([p0, p1, p2])
    df["y"] = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    fit = smf.mnlogit("y ~ x1 + x2", data=df).fit(disp=False)
    m = Margins.linear_scale(fit)
    pred = m.predict()
    assert pred.estimate.size == 3

    sub = pred.outcome(0)
    assert sub.estimate.size == 1
    assert "0" in sub.estimand_metadata["labels"][0]


# ---------------------------------------------------------------------------
# joint_test() on non-identity scale
# ---------------------------------------------------------------------------

def test_joint_test_non_identity_scale(fit_logit):
    m = Margins.log_scale(fit_logit, at="typical")
    result = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "age": 40}},
            {"atexog": {"treatment": 0, "age": 40}},
            {"atexog": {"treatment": 1, "age": 60}},
            {"atexog": {"treatment": 0, "age": 60}},
        ],
        contrasts={
            "age40": [+1, -1, 0, 0],
            "age60": [0, 0, +1, -1],
        },
    )
    # log_scale: phi=exp, phi_inv=log. The natural "no effect" point on the
    # reporting scale is 1 (since exp(0)=1). Pass value=1 explicitly.
    jt = result.joint_test(value=np.array([1.0, 1.0]), null_scale="reporting")
    assert jt.method == "joint_wald"
    assert np.isfinite(float(jt.statistic))
    assert np.isfinite(float(jt.pvalue))


# ---------------------------------------------------------------------------
# evaluate() with non-JAX-differentiable compose
# ---------------------------------------------------------------------------

def test_evaluate_non_jax_differentiable_compose(fit_logit):
    m = Margins.linear_scale(fit_logit, at="typical", method="simulation", n_sim=200, rng_seed=42)
    result = m.evaluate(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        compose=lambda p: p[0] / p[1] if p[1] > 0.5 else p[0] / (p[1] + 0.01),
    )
    # Non-differentiable compose auto-routes to simulation
    assert result.method == "simulation"
    assert np.isfinite(float(result.estimate))


# ---------------------------------------------------------------------------
# predict() with non-JAX-differentiable transform
# ---------------------------------------------------------------------------

def test_predict_non_jax_differentiable_transform(fit_logit):
    m = Margins.linear_scale(fit_logit, at="typical", method="simulation", n_sim=200, rng_seed=42)
    result = m.predict(
        atexog={"treatment": [0, 1]},
        transform=lambda x: x + 0.01 if x < 0.5 else x,
    )
    assert result.method == "simulation"
    assert np.all(np.isfinite(result.estimate))


# ---------------------------------------------------------------------------
# contrasts() with outcome parameter on multi-outcome model
# ---------------------------------------------------------------------------

def test_contrasts_with_outcome():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    lp0 = 1.0 + 0.5 * df["x1"] + 0.3 * df["x2"]
    lp1 = 0.5 + 0.2 * df["x1"] - 0.1 * df["x2"]
    denom = 1 + np.exp(lp0) + np.exp(lp1)
    p0 = 1 / denom
    p1 = np.exp(lp0) / denom
    p2 = np.exp(lp1) / denom
    probs = np.column_stack([p0, p1, p2])
    df["y"] = np.array([rng.choice(3, p=probs[i]) for i in range(n)])

    fit = smf.mnlogit("y ~ x1 + x2", data=df).fit(disp=False)
    m = Margins.linear_scale(fit)
    result = m.contrasts(
        scenarios=[
            {"atexog": {"x1": 1}},
            {"atexog": {"x1": 0}},
        ],
        contrasts=[1, -1],
        outcome=0,
    )
    assert result.estimate.size == 1
    assert np.isfinite(float(result.estimate.item()))


# ---------------------------------------------------------------------------
# TestResult.to_frame() multi-row
# ---------------------------------------------------------------------------

def test_testresult_to_frame_multi_row():
    from pymargins._result import TestResult
    tr = TestResult(
        statistic=np.array([2.0, 3.0]),
        pvalue=np.array([0.05, 0.01]),
        df=2,
        null_value=np.array([0.0, 0.0]),
        alternative="two-sided",
        method="joint_wald",
        estimand_metadata={},
    )
    frame = tr.to_frame()
    assert len(frame) == 2
    assert "statistic" in frame.columns
    assert "p_value" in frame.columns


# ---------------------------------------------------------------------------
# Empty training data
# ---------------------------------------------------------------------------

def test_empty_training_data_raises_clear_error(fit_logit):
    m = Margins.linear_scale(fit_logit, at="typical")
    # Mock training_data to be empty
    from unittest.mock import patch
    empty_df = pd.DataFrame({"age": [], "treatment": [], "sex": [], "outcome": []})
    with patch.object(m.adapter.__class__, "training_data", new=empty_df):
        with pytest.raises(ValueError):
            m.predict()


# ---------------------------------------------------------------------------
# All-NaN column in training data
# ---------------------------------------------------------------------------

def test_all_nan_column_raises_clear_error():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n) + 2.0 * rng.normal(size=n),
    })
    df["bad"] = np.nan
    fit = smf.ols("y ~ x", data=df).fit()
    m = Margins.linear_scale(fit, at="typical")
    # The adapter's training_data contains the original df with the all-NaN column.
    # predict() calls expand_scenario → aggregation_resolver → _summarize_column
    # on the all-NaN column, which should raise a clear ValueError.
    with pytest.raises(ValueError):
        m.predict()
