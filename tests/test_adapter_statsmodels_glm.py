"""Tests for StatsmodelsGLMAdapter.

See IMPLEMENTATION_GUIDE.md §0.3.
"""

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
from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_binary():
    """Synthetic data with a binary outcome."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
        }
    )
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.8 * df["treatment"]
    df["y"] = (rng.uniform(size=n) < (1 / (1 + np.exp(-eta)))).astype(float)
    return df


@pytest.fixture
def df_count():
    """Synthetic data with a count outcome."""
    rng = np.random.default_rng(43)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    df["y"] = rng.poisson(np.exp(eta))
    return df


@pytest.fixture
def fit_logit_formula(df_binary):
    fit = smf.glm(
        "y ~ x1 + x2 + treatment + C(region)",
        data=df_binary,
        family=sm.families.Binomial(),
    ).fit()
    return fit


@pytest.fixture
def fit_logit_array(df_binary):
    X = df_binary[["x1", "x2", "treatment"]].copy()
    X = sm.add_constant(X)
    y = df_binary["y"].values
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return fit


@pytest.fixture
def fit_poisson_formula(df_count):
    fit = smf.glm(
        "y ~ x1 + x2",
        data=df_count,
        family=sm.families.Poisson(),
    ).fit()
    return fit


# ---------------------------------------------------------------------------
# 1. Construction and auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_logit(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    assert isinstance(adapter, StatsmodelsGLMAdapter)


def test_adapter_coefficients(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    beta = adapter.coefficients()
    assert beta.ndim == 1
    np.testing.assert_allclose(
        np.asarray(beta),
        fit_logit_formula.params.values,
        rtol=1e-10,
    )


def test_adapter_training_data_formula(fit_logit_formula, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    pd.testing.assert_frame_equal(adapter.training_data, df_binary)


def test_adapter_training_data_array_requires_explicit(fit_logit_array, df_binary):
    # Array-fit models have no results.model.data.frame
    with pytest.raises(ValueError, match="training_data must be provided"):
        StatsmodelsGLMAdapter(fit_logit_array)
    # Explicit training_data should work
    adapter = StatsmodelsGLMAdapter(fit_logit_array, training_data=df_binary)
    assert adapter.training_data is df_binary


# ---------------------------------------------------------------------------
# 2. Covariance / vcov flavors
# ---------------------------------------------------------------------------


def test_covariance_default(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    Sigma = adapter.covariance()
    assert Sigma.shape == (len(fit_logit_formula.params),) * 2
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_logit_formula.cov_params().values,
        rtol=1e-10,
    )


def test_covariance_ndarray_override(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    custom = np.eye(len(fit_logit_formula.params))
    Sigma = adapter.covariance(custom)
    np.testing.assert_allclose(np.asarray(Sigma), custom, rtol=1e-10)


def test_covariance_hc0_hc1_hc2_smoke(fit_logit_formula):
    """HC0/HC1/HC2 should be obtainable via refit."""
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    for flavor in ("HC0", "HC1", "HC2"):
        Sigma = adapter.covariance(flavor)
        assert Sigma.ndim == 2
        assert Sigma.shape[0] == len(fit_logit_formula.params)


def test_covariance_hc3_refit_when_not_available(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    # When HC3 was not computed at fit time, the adapter refits and extracts it
    Sigma = adapter.covariance("HC3")
    assert Sigma.ndim == 2
    assert Sigma.shape[0] == len(fit_logit_formula.params)
    # Should differ from the default cov_params
    assert not np.allclose(np.asarray(Sigma), fit_logit_formula.cov_params().values)


def test_covariance_hc3_when_already_fit(df_binary):
    fit_hc3 = smf.glm(
        "y ~ x1 + x2 + treatment + C(region)",
        data=df_binary,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3")
    adapter = StatsmodelsGLMAdapter(fit_hc3)
    Sigma = adapter.covariance("HC3")
    np.testing.assert_allclose(
        np.asarray(Sigma),
        fit_hc3.cov_params().values,
        rtol=1e-10,
    )


def test_covariance_cluster_via_refit(fit_logit_formula, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    groups = np.arange(len(df_binary)) % 10  # 10 clusters
    Sigma = adapter.covariance({"type": "cluster", "groups": groups})
    assert Sigma.ndim == 2
    assert Sigma.shape[0] == len(fit_logit_formula.params)


def test_covariance_cluster_groups_length_mismatch(fit_logit_formula, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    groups = np.arange(len(df_binary) - 1) % 10  # wrong length
    with pytest.raises(ValueError, match="groups length"):
        adapter.covariance({"type": "cluster", "groups": groups})


def test_covariance_cluster_array_fit(fit_logit_array, df_binary):
    """Array-fit GLM can request cluster-robust vcov via refit."""
    adapter = StatsmodelsGLMAdapter(fit_logit_array, training_data=df_binary)
    groups = np.arange(len(df_binary)) % 10
    Sigma = adapter.covariance({"type": "cluster", "groups": groups})
    assert Sigma.ndim == 2
    assert Sigma.shape[0] == len(fit_logit_array.params)


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------


def test_predict_matches_statsmodels(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(adapter.training_data.iloc[:5])
    mu = adapter.predict(beta, X)
    mu_sm = fit_logit_formula.predict(adapter.training_data.iloc[:5])
    np.testing.assert_allclose(np.asarray(mu), mu_sm.values, rtol=1e-10)


def test_predict_new_data_unseen_levels(fit_logit_formula, df_binary):
    """Patsy with unseen categorical levels should raise a clear error."""
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    new_df = df_binary.iloc[:5].copy()
    new_df["region"] = "unknown_region"
    with pytest.raises(Exception):  # patsy raises PatsyError
        adapter.design_matrix_from_df(new_df)


# ---------------------------------------------------------------------------
# 4. Design matrix construction
# ---------------------------------------------------------------------------


def test_design_matrix_from_df_formula(fit_logit_formula, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    X = adapter.design_matrix_from_df(df_binary.iloc[:5])
    assert X.ndim == 2
    assert X.shape[0] == 5
    # Should include intercept + 2 continuous + treatment + 3 region dummies = 7
    assert X.shape[1] == len(fit_logit_formula.model.exog_names)


def test_design_matrix_from_df_array(fit_logit_array, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_array, training_data=df_binary)
    subset = df_binary[["x1", "x2", "treatment"]].iloc[:5].copy()
    subset.insert(0, "const", 1.0)
    X = adapter.design_matrix_from_df(subset)
    assert X.ndim == 2
    assert X.shape == (5, 4)


def test_design_matrix_from_df_array_auto_injects_intercept(fit_logit_array, df_binary):
    """Array-fit fallback should auto-inject intercept if exog_names expects it."""
    adapter = StatsmodelsGLMAdapter(fit_logit_array, training_data=df_binary)
    # Pass a df WITHOUT the const column
    subset = df_binary[["x1", "x2", "treatment"]].iloc[:5].copy()
    X = adapter.design_matrix_from_df(subset)
    assert X.shape == (5, 4)
    np.testing.assert_allclose(np.asarray(X)[:, 0], 1.0)


# ---------------------------------------------------------------------------
# 5. Variable metadata
# ---------------------------------------------------------------------------


def test_variable_metadata(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    meta = adapter.variable_metadata()
    assert "x1" in meta
    assert "x2" in meta
    assert "treatment" in meta
    assert "region" in meta
    assert meta["x1"].var_type == "continuous"
    assert meta["treatment"].var_type == "binary"
    assert meta["region"].var_type == "categorical"


def test_variable_metadata_is_cached(fit_logit_formula):
    """variable_metadata should be cached after first call."""
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    meta1 = adapter.variable_metadata()
    meta2 = adapter.variable_metadata()
    assert meta1 is meta2


# ---------------------------------------------------------------------------
# 6. Column index lookup
# ---------------------------------------------------------------------------


def test_column_index_continuous(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    idx = adapter.column_index_of_variable("x1")
    assert isinstance(idx, int)
    assert adapter._exog_names[idx] == "x1"


def test_column_index_categorical_raises(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    # Categorical variables should raise — dydx is undefined
    with pytest.raises(ValueError, match="categorical"):
        adapter.column_index_of_variable("region")


# ---------------------------------------------------------------------------
# 7. Bootstrap / refit
# ---------------------------------------------------------------------------


def test_refit_formula(fit_logit_formula, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    resampled = df_binary.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsGLMAdapter)
    # Coefficients should change (different data)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


def test_refit_array(fit_logit_array, df_binary):
    adapter = StatsmodelsGLMAdapter(fit_logit_array, training_data=df_binary)
    resampled = df_binary.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsGLMAdapter)
    # Coefficients should change (different data)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Attach-time validation (IMPLEMENTATION_GUIDE.md §2.3)
# ---------------------------------------------------------------------------


def test_attach_rejects_unsupported_vcov_string(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    with pytest.raises(
        ValueError, match="StatsmodelsGLMAdapter does not support vcov='HAC'"
    ):
        Margins(fit_logit_formula, adapter=adapter, vcov="HAC")


def test_attach_rejects_unsupported_vcov_dict(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    with pytest.raises(
        ValueError,
        match="StatsmodelsGLMAdapter does not support vcov dict with type='hac'",
    ):
        Margins(fit_logit_formula, adapter=adapter, vcov={"type": "hac"})


def test_attach_rejects_cluster_without_groups(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires 'groups'"):
        Margins(fit_logit_formula, adapter=adapter, vcov={"type": "cluster"})


def test_attach_accepts_supported_vcov(fit_logit_formula):
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    # HC0 string
    m = Margins(fit_logit_formula, adapter=adapter, vcov="HC0")
    assert m.vcov_spec == "HC0"
    # cluster dict
    df = adapter.training_data
    cluster_spec = {"type": "cluster", "groups": df["treatment"]}
    m2 = Margins(fit_logit_formula, adapter=adapter, vcov=cluster_spec)
    assert m2.vcov_spec["type"] == "cluster"
    assert m2.vcov_spec["groups"].equals(df["treatment"])
    # ndarray
    cov = np.eye(len(fit_logit_formula.params))
    m3 = Margins(fit_logit_formula, adapter=adapter, vcov=cov)
    assert m3.vcov_spec is cov


def test_attach_validates_phi_phi_inv(fit_logit_formula):
    """GLMAdapter.attach (via super) validates phi and phi_inv are inverses."""
    adapter = StatsmodelsGLMAdapter(fit_logit_formula)
    with pytest.raises(
        ValueError, match="phi and phi_inv do not appear to be inverses"
    ):
        Margins(fit_logit_formula, adapter=adapter, phi=jnp.exp, phi_inv=jnp.exp)


# ---------------------------------------------------------------------------
# Refit preserves model-specific args
# ---------------------------------------------------------------------------


def test_refit_preserves_offset_and_exposure(df_count):
    """Refit should preserve offset, exposure, freq_weights, var_weights."""
    rng = np.random.default_rng(44)
    df_count["offset"] = rng.standard_normal(len(df_count))
    df_count["exposure"] = rng.uniform(0.5, 2.0, size=len(df_count))
    fit = smf.glm(
        "y ~ x1 + x2",
        data=df_count,
        family=sm.families.Poisson(),
        offset=df_count["offset"],
        exposure=df_count["exposure"],
    ).fit()
    adapter = StatsmodelsGLMAdapter(fit, training_data=df_count)
    resampled = df_count.sample(frac=1.0, replace=True, random_state=42)
    new_adapter = adapter.refit(resampled)
    assert isinstance(new_adapter, StatsmodelsGLMAdapter)
    # Coefficients should change (different data)
    assert not np.allclose(
        np.asarray(adapter.coefficients()),
        np.asarray(new_adapter.coefficients()),
    )


# ---------------------------------------------------------------------------
# Covariance edge cases
# ---------------------------------------------------------------------------


def test_covariance_unsupported_string_raises(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov string"):
        adapter.covariance(vcov_spec="hac")


def test_covariance_unsupported_dict_raises(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov dict type"):
        adapter.covariance(vcov_spec={"type": "hac"})


def test_covariance_cluster_missing_groups_raises(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    with pytest.raises(ValueError, match="cluster vcov requires"):
        adapter.covariance(vcov_spec={"type": "cluster"})


def test_covariance_unsupported_type_raises(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    with pytest.raises(ValueError, match="Unsupported vcov_spec"):
        adapter.covariance(vcov_spec=123)


def test_covariance_precomputed_matrix(fit_logit_formula):
    adapter = auto_detect_adapter(fit_logit_formula)
    cov0 = adapter.covariance()
    cov1 = adapter.covariance(vcov_spec=np.asarray(cov0))
    np.testing.assert_allclose(np.asarray(cov0), np.asarray(cov1), rtol=1e-10)
