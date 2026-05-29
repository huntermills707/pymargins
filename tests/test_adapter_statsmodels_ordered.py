"""Tests for StatsmodelsOrderedAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.miscmodels.ordinal_model as om

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_ordered import StatsmodelsOrderedAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_ordered():
    """Synthetic data for ordered logit."""
    rng = np.random.default_rng(43)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.6 * df["treatment"]
    # thresholds at -1, 0, 1
    y = np.zeros(n, dtype=int)
    y[eta > -1] = 1
    y[eta > 0] = 2
    y[eta > 1] = 3
    df["y"] = y
    return df


@pytest.fixture
def ordered_fit_array(df_ordered):
    """Array-fit OrderedModel."""
    return om.OrderedModel(
        df_ordered["y"], df_ordered[["x1", "x2", "treatment"]], distr="logit"
    ).fit(disp=False)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_array_requires_training_data(ordered_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(ordered_fit_array)


def test_auto_detect_with_training_data(ordered_fit_array, df_ordered):
    # Auto-detect returns the class; instantiation happens in auto_detect_adapter
    # which calls cls(model) without training_data. Since OrderedModel has no
    # formula API, we test the class directly.
    from pymargins._adapters import _detect_adapter_class

    cls = _detect_adapter_class(ordered_fit_array)
    assert cls is StatsmodelsOrderedAdapter


# ---------------------------------------------------------------------------
# Coefficients and covariance
# ---------------------------------------------------------------------------


def test_coefficients_shape(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    n_params = len(ordered_fit_array.params)
    assert adapter.coefficients().shape == (n_params,)


def test_covariance_default(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    cov = adapter.covariance()
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_shape_and_sum_to_one(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    X = adapter.design_matrix_from_df(df_ordered[:10])
    probs = adapter.predict(adapter.coefficients(), X)
    assert probs.shape == (10, adapter.n_outcomes)
    np.testing.assert_array_almost_equal(np.asarray(probs.sum(axis=1)), np.ones(10))


def test_predict_matches_statsmodels(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    X = adapter.design_matrix_from_df(df_ordered[:10])
    our_probs = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_probs = ordered_fit_array.model.predict(
        ordered_fit_array.params, exog=np.asarray(X)
    )
    np.testing.assert_allclose(our_probs, sm_probs, atol=1e-5)


def test_predict_jax_differentiable(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    X = adapter.design_matrix_from_df(df_ordered[:10])
    beta = adapter.coefficients()

    def mean_pred(b):
        return jnp.mean(adapter.predict(b, X), axis=0)

    # Should not raise — WrappedFDAdapter provides custom JVP
    grad = jax.jacobian(mean_pred)(beta)
    assert grad.shape == (adapter.n_outcomes, beta.shape[0])


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_array(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    X = adapter.design_matrix_from_df(df_ordered[:5])
    p = ordered_fit_array.model.exog.shape[1]
    assert X.shape[1] == p


def test_variable_metadata(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


def test_column_index_of_variable(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)


def test_column_index_raises_for_binary(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    with pytest.raises(ValueError, match="binary"):
        adapter.column_index_of_variable("treatment")


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------


def test_margins_predict_aap(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    m = Margins.linear_scale(ordered_fit_array, adapter=adapter)
    res = m.predict()
    assert res.estimate.shape == (adapter.n_outcomes,)
    np.testing.assert_allclose(res.estimate.sum(), 1.0, atol=1e-10)


def test_margins_dydx(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    m = Margins.linear_scale(ordered_fit_array, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.shape == (adapter.n_outcomes,)
    assert np.isfinite(res.estimate).all()


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_array(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    new_adapter = adapter.refit(df_ordered)
    assert isinstance(new_adapter, StatsmodelsOrderedAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Outcome subsetting
# ---------------------------------------------------------------------------


def test_predict_outcome_subset(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    m = Margins.linear_scale(ordered_fit_array, adapter=adapter)
    full = m.predict()
    sub = m.predict(outcome=2)
    np.testing.assert_allclose(sub.estimate, full.estimate[2:3], atol=1e-12)


def test_dydx_outcome_subset(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    m = Margins.linear_scale(ordered_fit_array, adapter=adapter)
    full = m.dydx("x1")
    sub = m.dydx("x1", outcome=[1, 2])
    np.testing.assert_allclose(sub.estimate, full.estimate[[1, 2]], atol=1e-12)


def test_result_outcome_helper(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    m = Margins.linear_scale(ordered_fit_array, adapter=adapter)
    full = m.predict()
    sub = full.outcome(2)
    np.testing.assert_allclose(sub.estimate, full.estimate[2:3], atol=1e-12)


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------


def test_attach_rejects_bad_vcov(ordered_fit_array, df_ordered):
    adapter = StatsmodelsOrderedAdapter(ordered_fit_array, training_data=df_ordered)
    from unittest.mock import MagicMock

    session = MagicMock()
    session.vcov_spec = "HAC"
    with pytest.raises(ValueError, match="HAC"):
        adapter.attach(session)
