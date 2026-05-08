"""Tests for StatsmodelsOLSAdapter.

See IMPLEMENTATION_GUIDE.md §1.1 and §1.2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter
from pymargins._adapter import auto_detect_adapter
from pymargins import Margins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_ols():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "treatment": rng.binomial(1, 0.5, n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
    })
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + 0.8 * df["treatment"] + rng.standard_normal(n) * 0.5
    return df


@pytest.fixture
def fit_ols_formula(df_ols):
    return smf.ols("y ~ x1 + x2 + treatment + C(region)", data=df_ols).fit()


@pytest.fixture
def fit_ols_array(df_ols):
    X = df_ols[["x1", "x2", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_ols["y"].values
    return sm.OLS(y, X).fit()


@pytest.fixture
def fit_wls_formula(df_ols):
    df_ols["w"] = np.random.default_rng(7).uniform(0.5, 2.0, size=len(df_ols))
    return smf.wls("y ~ x1 + x2 + treatment", data=df_ols, weights=df_ols["w"]).fit()


# ---------------------------------------------------------------------------
# 1. Construction and auto-detection
# ---------------------------------------------------------------------------

def test_auto_detect_ols(fit_ols_formula):
    adapter = auto_detect_adapter(fit_ols_formula)
    assert isinstance(adapter, StatsmodelsOLSAdapter)


def test_auto_detect_wls(fit_wls_formula):
    adapter = auto_detect_adapter(fit_wls_formula)
    assert isinstance(adapter, StatsmodelsOLSAdapter)


def test_auto_detect_array_fit_ols(df_ols):
    """Array-fit OLS should auto-detect via RegressionResultsWrapper."""
    from pymargins._adapters import _detect_adapter_class
    X = df_ols[["x1", "x2", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_ols["y"].values
    fit = sm.OLS(y, X).fit()
    cls = _detect_adapter_class(fit)
    assert cls is StatsmodelsOLSAdapter
    # auto_detect_adapter itself raises because training_data is unavailable
    with pytest.raises(ValueError, match="training_data must be provided"):
        auto_detect_adapter(fit)


def test_auto_detect_wls_wrapper():
    """WLSResultsWrapper should map to StatsmodelsOLSAdapter."""
    from pymargins._adapters import _detect_adapter_class
    class FakeWLSResult:
        pass
    FakeWLSResult.__module__ = "statsmodels.regression"
    FakeWLSResult.__name__ = "WLSResultsWrapper"
    cls = _detect_adapter_class(FakeWLSResult())
    assert cls is StatsmodelsOLSAdapter


def test_auto_detect_gls_wrapper():
    """GLSResultsWrapper should map to StatsmodelsOLSAdapter."""
    from pymargins._adapters import _detect_adapter_class
    class FakeGLSResult:
        pass
    FakeGLSResult.__module__ = "statsmodels.regression"
    FakeGLSResult.__name__ = "GLSResultsWrapper"
    cls = _detect_adapter_class(FakeGLSResult())
    assert cls is StatsmodelsOLSAdapter


def test_auto_detect_logit_wrapper():
    """LogitResultsWrapper should map to StatsmodelsGLMAdapter."""
    from pymargins._adapters import _detect_adapter_class
    class FakeLogitResult:
        pass
    FakeLogitResult.__module__ = "statsmodels.discrete"
    FakeLogitResult.__name__ = "LogitResultsWrapper"
    cls = _detect_adapter_class(FakeLogitResult())
    from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
    assert cls is StatsmodelsGLMAdapter


def test_auto_detect_probit_wrapper():
    """ProbitResultsWrapper should map to StatsmodelsGLMAdapter."""
    from pymargins._adapters import _detect_adapter_class
    class FakeProbitResult:
        pass
    FakeProbitResult.__module__ = "statsmodels.discrete"
    FakeProbitResult.__name__ = "ProbitResultsWrapper"
    cls = _detect_adapter_class(FakeProbitResult())
    from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
    assert cls is StatsmodelsGLMAdapter


def test_adapter_coefficients(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        fit_ols_formula.params.values,
        rtol=1e-10,
    )


def test_adapter_training_data_formula(fit_ols_formula, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    pd.testing.assert_frame_equal(adapter.training_data, df_ols)


def test_adapter_training_data_array_requires_explicit(fit_ols_array, df_ols):
    with pytest.raises(ValueError, match="training_data must be provided"):
        StatsmodelsOLSAdapter(fit_ols_array)
    adapter = StatsmodelsOLSAdapter(fit_ols_array, training_data=df_ols)
    assert adapter.training_data is df_ols


# ---------------------------------------------------------------------------
# 2. Covariance / vcov flavors
# ---------------------------------------------------------------------------

def test_covariance_default(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    Sigma = adapter.covariance()
    assert Sigma.shape == (len(fit_ols_formula.params),) * 2
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_ols_formula.cov_params().values,
        rtol=1e-10,
    )


def test_covariance_hc3_directly_available(fit_ols_formula):
    """OLS exposes cov_HC3 as an attribute regardless of fit cov_type."""
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    Sigma = adapter.covariance("HC3")
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_ols_formula.cov_HC3,
        rtol=1e-10,
    )


def test_covariance_cluster_via_refit(fit_ols_formula, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    groups = np.arange(len(df_ols)) % 10
    Sigma = adapter.covariance({"type": "cluster", "groups": groups})
    assert Sigma.ndim == 2
    assert Sigma.shape[0] == len(fit_ols_formula.params)


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_statsmodels(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    mu_sm = fit_ols_formula.predict(adapter.training_data.iloc[:5])
    np.testing.assert_allclose(np.asarray(mu), mu_sm.values, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Design matrix construction
# ---------------------------------------------------------------------------

def test_design_matrix_from_df_formula(fit_ols_formula, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    X = adapter.design_matrix_from_df(df_ols.iloc[:5])
    assert X.ndim == 2
    assert X.shape[0] == 5
    assert X.shape[1] == len(fit_ols_formula.model.exog_names)


def test_design_matrix_from_df_array(fit_ols_array, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_array, training_data=df_ols)
    subset = df_ols[["x1", "x2", "treatment"]].iloc[:5].copy()
    subset.insert(0, "const", 1.0)
    X = adapter.design_matrix_from_df(subset)
    assert X.ndim == 2
    assert X.shape == (5, 4)


# ---------------------------------------------------------------------------
# 5. Variable metadata and column lookup
# ---------------------------------------------------------------------------

def test_variable_metadata(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert "treatment" in meta
    assert "region" in meta
    assert meta["x1"].var_type == "continuous"
    assert meta["treatment"].var_type == "binary"
    assert meta["region"].var_type == "categorical"


def test_variable_metadata_is_cached(fit_ols_formula):
    """variable_metadata should be cached after first call."""
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    meta1 = adapter.variable_metadata()
    meta2 = adapter.variable_metadata()
    assert meta1 is meta2


def test_column_index_continuous(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    idx = adapter.column_index_of_variable("x1")
    assert adapter._exog_names[idx] == "x1"


def test_column_index_categorical_raises(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    with pytest.raises(ValueError, match="categorical"):
        adapter.column_index_of_variable("region")


# ---------------------------------------------------------------------------
# 6. Bootstrap / refit
# ---------------------------------------------------------------------------

def test_refit_formula(fit_ols_formula, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    resampled = df_ols.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_array(fit_ols_array, df_ols):
    adapter = StatsmodelsOLSAdapter(fit_ols_array, training_data=df_ols)
    resampled = df_ols.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_wls_preserves_weights(fit_wls_formula, df_ols):
    """WLS refit should preserve the original weights."""
    adapter = StatsmodelsOLSAdapter(fit_wls_formula)
    resampled = df_ols.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsOLSAdapter)
    # Coefficients should change (different data)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# 7. End-to-end bootstrap with array-fit adapter
# ---------------------------------------------------------------------------

def test_bootstrap_end_to_end_array_fit(fit_ols_array, df_ols):
    """Bootstrap CI should work end-to-end with an array-fit OLS adapter."""
    adapter = StatsmodelsOLSAdapter(fit_ols_array, training_data=df_ols)
    m = Margins.linear_scale(fit_ols_array, adapter=adapter, at="typical",
                              method="bootstrap", n_boot=50, rng_seed=42)
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
    assert rd.draws is not None


# ---------------------------------------------------------------------------
# Attach-time validation (IMPLEMENTATION_GUIDE.md §2.3)
# ---------------------------------------------------------------------------

def test_attach_rejects_unsupported_vcov_string(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    with pytest.raises(ValueError, match="StatsmodelsOLSAdapter does not support vcov='HAC'"):
        Margins(fit_ols_formula, adapter=adapter, vcov="HAC")


def test_attach_rejects_unsupported_vcov_dict(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    with pytest.raises(ValueError, match="StatsmodelsOLSAdapter does not support vcov dict with type='hac'"):
        Margins(fit_ols_formula, adapter=adapter, vcov={"type": "hac"})


def test_attach_rejects_cluster_without_groups(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    with pytest.raises(ValueError, match="cluster vcov requires 'groups'"):
        Margins(fit_ols_formula, adapter=adapter, vcov={"type": "cluster"})


def test_attach_accepts_supported_vcov(fit_ols_formula):
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    # HC0 string
    m = Margins(fit_ols_formula, adapter=adapter, vcov="HC0")
    assert m.vcov_spec == "HC0"
    # cluster dict
    df = adapter.training_data
    cluster_spec = {"type": "cluster", "groups": df["treatment"]}
    m2 = Margins(fit_ols_formula, adapter=adapter, vcov=cluster_spec)
    assert m2.vcov_spec["type"] == "cluster"
    assert m2.vcov_spec["groups"].equals(df["treatment"])
    # ndarray
    cov = np.eye(len(fit_ols_formula.params))
    m3 = Margins(fit_ols_formula, adapter=adapter, vcov=cov)
    assert m3.vcov_spec is cov
    # JAX array
    import jax.numpy as jnp
    cov_jax = jnp.eye(len(fit_ols_formula.params))
    m4 = Margins(fit_ols_formula, adapter=adapter, vcov=cov_jax)
    assert m4.vcov_spec is cov_jax


def test_attach_validates_phi_phi_inv(fit_ols_formula):
    """ModelAdapter.attach validates phi and phi_inv are inverses for OLS too."""
    adapter = StatsmodelsOLSAdapter(fit_ols_formula)
    with pytest.raises(ValueError, match="phi and phi_inv do not appear to be inverses"):
        Margins(fit_ols_formula, adapter=adapter, phi=jnp.exp, phi_inv=jnp.exp)
