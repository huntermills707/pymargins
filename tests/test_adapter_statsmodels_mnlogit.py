"""Tests for StatsmodelsMNLogitAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_mnlogit import StatsmodelsMNLogitAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_mnlogit():
    """Synthetic data for multinomial logit."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
        }
    )
    # Generate a 3-category outcome
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])
    return df


@pytest.fixture
def mnlogit_fit_array(df_mnlogit):
    """Array-fit MNLogit."""
    return sm.MNLogit(
        df_mnlogit["y"], sm.add_constant(df_mnlogit[["x1", "x2", "treatment"]])
    ).fit(disp=False)


@pytest.fixture
def mnlogit_fit_formula(df_mnlogit):
    """Formula-fit MNLogit."""
    return smf.mnlogit("y ~ x1 + x2 + treatment + C(region)", data=df_mnlogit).fit(
        disp=False
    )


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_formula(mnlogit_fit_formula):
    adapter = auto_detect_adapter(mnlogit_fit_formula)
    assert isinstance(adapter, StatsmodelsMNLogitAdapter)


def test_auto_detect_array_requires_training_data(mnlogit_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(mnlogit_fit_array)


# ---------------------------------------------------------------------------
# Coefficients and covariance
# ---------------------------------------------------------------------------


def test_coefficients_shape(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    p = len(mnlogit_fit_formula.model.exog_names)
    K = mnlogit_fit_formula.model.J
    assert adapter.coefficients().shape == (p * (K - 1),)


def test_covariance_default(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


def test_covariance_hc(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    cov = adapter.covariance("HC0")
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_shape_and_sum_to_one(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    probs = adapter.predict(adapter.coefficients(), X)
    assert probs.shape == (10, adapter.n_outcomes)
    np.testing.assert_array_almost_equal(np.asarray(probs.sum(axis=1)), np.ones(10))


def test_predict_matches_statsmodels(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_probs = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_probs = mnlogit_fit_formula.predict(df[:10])
    np.testing.assert_allclose(our_probs, sm_probs, atol=1e-6)


def test_predict_jax_differentiable(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    beta = adapter.coefficients()

    def mean_pred(b):
        return jnp.mean(adapter.predict(b, X), axis=0)

    # Should not raise
    grad = jax.jacobian(mean_pred)(beta)
    assert grad.shape == (adapter.n_outcomes, beta.shape[0])


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    p = len(mnlogit_fit_formula.model.exog_names)
    assert X.shape[1] == p


def test_design_matrix_array(mnlogit_fit_array, df_mnlogit):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_array, training_data=df_mnlogit)
    X = adapter.design_matrix_from_df(df_mnlogit[:5])
    p = len(mnlogit_fit_array.model.exog_names)
    assert X.shape[1] == p


def test_variable_metadata(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


def test_column_index_of_variable(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)


def test_column_index_raises_for_categorical(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    with pytest.raises(ValueError, match="categorical"):
        adapter.column_index_of_variable("region")


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------


def test_margins_predict_aap(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    res = m.predict()
    assert res.estimate.shape == (adapter.n_outcomes,)
    np.testing.assert_allclose(res.estimate.sum(), 1.0, atol=1e-10)


def test_margins_dydx(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.shape == (adapter.n_outcomes,)
    assert np.isfinite(res.estimate).all()


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    new_adapter = adapter.refit(adapter.training_data)
    assert isinstance(new_adapter, StatsmodelsMNLogitAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array(mnlogit_fit_array, df_mnlogit):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_array, training_data=df_mnlogit)
    new_adapter = adapter.refit(df_mnlogit)
    assert isinstance(new_adapter, StatsmodelsMNLogitAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Outcome subsetting
# ---------------------------------------------------------------------------


def test_predict_outcome_subset(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    full = m.predict()
    sub = m.predict(outcome=1)
    np.testing.assert_allclose(sub.estimate, full.estimate[1:2], atol=1e-12)


def test_predict_outcome_multi_subset(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    full = m.predict()
    sub = m.predict(outcome=[0, 2])
    np.testing.assert_allclose(sub.estimate, full.estimate[[0, 2]], atol=1e-12)


def test_dydx_outcome_subset(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    full = m.dydx("x1")
    sub = m.dydx("x1", outcome=2)
    np.testing.assert_allclose(sub.estimate, full.estimate[2:3], atol=1e-12)


def test_result_outcome_helper(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    full = m.predict()
    sub = full.outcome(1)
    np.testing.assert_allclose(sub.estimate, full.estimate[1:2], atol=1e-12)


def test_result_outcome_helper_by_label(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    m = Margins.linear_scale(mnlogit_fit_formula, adapter=adapter)
    full = m.predict()
    sub = full.outcome("1")
    np.testing.assert_allclose(sub.estimate, full.estimate[1:2], atol=1e-12)


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------


def test_attach_rejects_bad_vcov(mnlogit_fit_formula):
    adapter = StatsmodelsMNLogitAdapter(mnlogit_fit_formula)
    from unittest.mock import MagicMock

    session = MagicMock()
    session.vcov_spec = "HAC"
    with pytest.raises(ValueError, match="HAC"):
        adapter.attach(session)
