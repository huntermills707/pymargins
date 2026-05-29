"""Tests for StatsmodelsPHRegAdapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg

jax.config.update("jax_enable_x64", True)

from pymargins import Margins
from pymargins._adapter import auto_detect_adapter
from pymargins._adapters.statsmodels_phreg import StatsmodelsPHRegAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_survival():
    """Synthetic survival data."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    hazard = np.exp(0.5 + 0.3 * df["x1"] - 0.2 * df["x2"])
    df["T"] = rng.exponential(1.0 / hazard)
    df["E"] = (rng.random(n) < 0.8).astype(int)
    return df


@pytest.fixture
def phreg_fit(df_survival):
    return PHReg(
        df_survival["T"].values,
        df_survival[["x1", "x2"]].values,
        status=df_survival["E"].values,
    ).fit()


# ---------------------------------------------------------------------------
# Fixtures (additional)
# ---------------------------------------------------------------------------


@pytest.fixture
def phreg_fit_formula(df_survival):
    return PHReg.from_formula(
        "T ~ x1 + x2", data=df_survival, status=df_survival["E"].values
    ).fit()


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_requires_training_data(phreg_fit):
    with pytest.raises(ValueError, match="training_data"):
        auto_detect_adapter(phreg_fit)


def test_auto_detect_formula(phreg_fit_formula):
    adapter = auto_detect_adapter(phreg_fit_formula)
    assert isinstance(adapter, StatsmodelsPHRegAdapter)


# ---------------------------------------------------------------------------
# Construction and core data access
# ---------------------------------------------------------------------------


def test_adapter_coefficients(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        phreg_fit.params,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:10])
    our_pred = np.asarray(adapter.predict(adapter.coefficients(), X))
    # PHReg predict returns a bunch with predicted_values for lhr
    sm_pred = phreg_fit.model.predict(
        phreg_fit.params, exog=df_survival[["x1", "x2"]].iloc[:10].values
    )
    # The default pred_type is "lhr" (log hazard ratio)
    # Our predict returns exp(X @ beta) = hazard ratio
    # We need to compare with exp(lhr)
    sm_hr = np.exp(sm_pred.predicted_values)
    np.testing.assert_allclose(our_pred, sm_hr, atol=1e-6)


def test_predict_jax_differentiable(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:10])
    beta = adapter.coefficients()

    def mean_pred(b):
        return jnp.mean(adapter.predict(b, X))

    grad = jax.grad(mean_pred)(beta)
    assert grad.shape == beta.shape


# ---------------------------------------------------------------------------
# Coefficients and covariance
# ---------------------------------------------------------------------------


def test_coefficients_shape(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    assert adapter.coefficients().shape == (2,)


def test_covariance_default(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    cov = adapter.covariance()
    assert cov.shape == (2, 2)


def test_covariance_rejects_hc(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    with pytest.raises(ValueError, match="does not support"):
        adapter.covariance("HC0")


def test_covariance_ndarray_override(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    custom = np.eye(len(phreg_fit.params))
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


# ---------------------------------------------------------------------------
# Design matrix and metadata
# ---------------------------------------------------------------------------


def test_design_matrix(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    X = adapter.design_matrix_from_df(df_survival[:5])
    assert X.shape == (5, 2)


def test_design_matrix_missing_columns(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    bad_df = df_survival[["T", "E"]].copy()
    with pytest.raises(ValueError, match="Missing columns"):
        adapter.design_matrix_from_df(bad_df)


def test_variable_metadata(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert meta["x1"].var_type == "continuous"


def test_variable_metadata_is_cached(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    meta1 = adapter.variable_metadata()
    meta2 = adapter.variable_metadata()
    assert meta1 is meta2


# ---------------------------------------------------------------------------
# End-to-end via Margins session
# ---------------------------------------------------------------------------


def test_margins_predict(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    m = Margins.log_scale(phreg_fit, adapter=adapter)
    res = m.predict()
    assert res.estimate.size == 1


def test_margins_dydx(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    m = Margins.log_scale(phreg_fit, adapter=adapter)
    res = m.dydx("x1")
    assert res.estimate.size == 1
    assert np.isfinite(float(res.estimate))


# ---------------------------------------------------------------------------
# Bootstrap end-to-end
# ---------------------------------------------------------------------------


def test_bootstrap_end_to_end(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    m = Margins.log_scale(
        phreg_fit,
        adapter=adapter,
        at="typical",
        method="bootstrap",
        n_boot=50,
        rng_seed=42,
    )
    rd = m.dydx("x1")
    assert rd.method == "bootstrap"
    assert np.isfinite(float(rd.estimate))
    assert float(rd.conf_int_lower) < float(rd.conf_int_upper)
    assert rd.draws is not None


# ---------------------------------------------------------------------------
# Attach-time validation
# ---------------------------------------------------------------------------


def test_attach_rejects_unsupported_vcov_string(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    with pytest.raises(
        ValueError, match="StatsmodelsPHRegAdapter does not support vcov='HC0'"
    ):
        Margins(phreg_fit, adapter=adapter, vcov="HC0")


def test_attach_rejects_unsupported_vcov_dict(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    with pytest.raises(
        ValueError, match="StatsmodelsPHRegAdapter does not support vcov dict"
    ):
        Margins(
            phreg_fit,
            adapter=adapter,
            vcov={"type": "cluster", "groups": df_survival["E"]},
        )


def test_attach_accepts_supported_vcov(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    # default (None)
    m1 = Margins.log_scale(phreg_fit, adapter=adapter)
    assert m1.vcov_spec is None
    # ndarray
    cov = np.eye(len(phreg_fit.params))
    m2 = Margins(phreg_fit, adapter=adapter, vcov=cov)
    assert m2.vcov_spec is cov


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------


def test_refit(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    new_adapter = adapter.refit(df_survival)
    assert isinstance(new_adapter, StatsmodelsPHRegAdapter)
    np.testing.assert_allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
        rtol=1e-5,
    )


def test_refit_changes_coefficients(phreg_fit, df_survival):
    adapter = StatsmodelsPHRegAdapter(phreg_fit, training_data=df_survival)
    resampled = df_survival.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsPHRegAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )
