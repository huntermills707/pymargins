"""Tests for StatsmodelsDiscreteCountAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_discrete_count import (
    StatsmodelsDiscreteCountAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_count():
    """Synthetic data for count models."""
    rng = np.random.default_rng(44)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.6 * df["treatment"]
    df["y"] = rng.poisson(np.exp(eta))
    return df


@pytest.fixture
def poisson_fit_array(df_count):
    return sm.Poisson(
        df_count["y"], sm.add_constant(df_count[["x1", "x2", "treatment"]])
    ).fit(disp=False)


@pytest.fixture
def poisson_fit_formula(df_count):
    return smf.poisson("y ~ x1 + x2 + treatment", data=df_count).fit(disp=False)


@pytest.fixture
def nb_fit_array(df_count):
    return sm.NegativeBinomial(
        df_count["y"], sm.add_constant(df_count[["x1", "x2", "treatment"]])
    ).fit(disp=False)


@pytest.fixture
def nbp_fit_array(df_count):
    return sm.NegativeBinomialP(
        df_count["y"], sm.add_constant(df_count[["x1", "x2", "treatment"]])
    ).fit(disp=False)


@pytest.fixture
def gp_fit_array(df_count):
    return sm.GeneralizedPoisson(
        df_count["y"], sm.add_constant(df_count[["x1", "x2", "treatment"]])
    ).fit(disp=False)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_poisson_formula(poisson_fit_formula):
    adapter = auto_detect_adapter(poisson_fit_formula)
    assert isinstance(adapter, StatsmodelsDiscreteCountAdapter)


def test_auto_detect_poisson_array_requires_training_data(poisson_fit_array):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(poisson_fit_array)


# ---------------------------------------------------------------------------
# Prediction accuracy
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels_poisson(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = poisson_fit_formula.predict(df[:10])
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_matches_statsmodels_nb(nb_fit_array, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(nb_fit_array, training_data=df_count)
    X = adapter.design_matrix_from_df(df_count[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    # For array-fit discrete models, use model.predict(params, exog=...) directly
    sm_pred = nb_fit_array.model.predict(nb_fit_array.params, exog=np.asarray(X))
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_matches_statsmodels_nbp(nbp_fit_array, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(nbp_fit_array, training_data=df_count)
    X = adapter.design_matrix_from_df(df_count[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = nbp_fit_array.model.predict(nbp_fit_array.params, exog=np.asarray(X))
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_matches_statsmodels_gp(gp_fit_array, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(gp_fit_array, training_data=df_count)
    X = adapter.design_matrix_from_df(df_count[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    sm_pred = gp_fit_array.model.predict(gp_fit_array.params, exog=np.asarray(X))
    np.testing.assert_allclose(our_pred, sm_pred, atol=1e-6)


def test_predict_jax_differentiable(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:10])
    beta = adapter.coefficients()

    def mean_pred(b):
        return jnp.mean(adapter.predict(b, X))

    grad = jax.grad(mean_pred)(beta)
    assert grad.shape == beta.shape


# ---------------------------------------------------------------------------
# Coefficients and covariance
# ---------------------------------------------------------------------------


def test_coefficients_shape_poisson(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    p = len(poisson_fit_formula.model.exog_names)
    assert adapter.coefficients().shape == (p,)


def test_coefficients_shape_nb(nb_fit_array, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(nb_fit_array, training_data=df_count)
    # Coefficient vector includes extra dispersion param(s)
    assert adapter.coefficients().shape == nb_fit_array.params.shape


def test_covariance_default(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    cov = adapter.covariance()
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix_formula(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    df = adapter.training_data
    X = adapter.design_matrix_from_df(df[:5])
    p = len(poisson_fit_formula.model.exog_names)
    assert X.shape[1] == p


def test_variable_metadata(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert meta["x1"].var_type == "continuous"


# ---------------------------------------------------------------------------
# End-to-end via GComputation
# ---------------------------------------------------------------------------


def test_gcomputation_predict_aap_poisson(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    est = GComputation(poisson_fit_formula, adapter=adapter)
    res = est.predict()
    assert res.estimate.size == 1
    assert float(res.estimate) > 0


def test_gcomputation_dydx_poisson(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    est = GComputation(poisson_fit_formula, adapter=adapter)
    res = est.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit_formula_poisson(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    new_adapter = adapter.refit(adapter.training_data)
    assert isinstance(new_adapter, StatsmodelsDiscreteCountAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_array_poisson(poisson_fit_array, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_array, training_data=df_count)
    new_adapter = adapter.refit(df_count)
    assert isinstance(new_adapter, StatsmodelsDiscreteCountAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Attach validation
# ---------------------------------------------------------------------------


def test_attach_rejects_bad_vcov(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    from unittest.mock import MagicMock

    session = MagicMock()
    session.vcov_spec = "HAC"
    with pytest.raises(ValueError, match="HAC"):
        adapter.attach(session)


# ---------------------------------------------------------------------------
# Covariance edge cases
# ---------------------------------------------------------------------------


def test_covariance_hc3_via_refit(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    cov = adapter.covariance(vcov_spec="hc3")
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


def test_covariance_cluster_via_refit(poisson_fit_formula, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    groups = df_count["treatment"].values
    cov = adapter.covariance(vcov_spec={"type": "cluster", "groups": groups})
    assert cov.shape == (
        adapter.coefficients().shape[0],
        adapter.coefficients().shape[0],
    )


def test_covariance_unsupported_string_raises(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov string"):
        adapter.covariance(vcov_spec="hac")


def test_covariance_unsupported_dict_raises(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov dict type"):
        adapter.covariance(vcov_spec={"type": "hac"})


def test_covariance_cluster_missing_groups_raises(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires"):
        adapter.covariance(vcov_spec={"type": "cluster"})


def test_covariance_unsupported_type_raises(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov_spec"):
        adapter.covariance(vcov_spec=123)


def test_covariance_precomputed_matrix(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    cov0 = adapter.covariance()
    cov1 = adapter.covariance(vcov_spec=np.asarray(cov0))
    np.testing.assert_allclose(np.asarray(cov0), np.asarray(cov1), rtol=1e-10)


# ---------------------------------------------------------------------------
# Score observations
# ---------------------------------------------------------------------------


def test_score_obs(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    score = adapter.score_obs()
    assert score.ndim == 2
    assert score.shape[1] == adapter.coefficients().shape[0]


# ---------------------------------------------------------------------------
# Column index
# ---------------------------------------------------------------------------


def test_column_index_of_variable(poisson_fit_formula):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)
    assert idx >= 0


# ---------------------------------------------------------------------------
# Refit with index
# ---------------------------------------------------------------------------


def test_refit_with_index(poisson_fit_formula, df_count):
    adapter = StatsmodelsDiscreteCountAdapter(poisson_fit_formula)
    idx = np.random.default_rng(7).choice(
        len(df_count), size=len(df_count), replace=True
    )
    new_adapter = adapter.refit(adapter.training_data.iloc[idx], index=idx)
    assert isinstance(new_adapter, StatsmodelsDiscreteCountAdapter)
