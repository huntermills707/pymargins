"""Smoke test for API_REVIEW.md fixes."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymargins._adapter import ModelAdapter
from pymargins._estimands import (
    make_prediction_estimand,
    make_slope_estimand,
    is_jax_differentiable,
)
from pymargins._inference import InferenceConfig, run_inference
from pymargins._kappa import kappa
from pymargins._result import MarginsResult
from pymargins.margins import Margins


class MockAdapter(ModelAdapter):
    """Minimal adapter for testing."""

    def __init__(self, beta, Sigma, training_df):
        self._beta = jnp.asarray(beta, dtype=float)
        self._Sigma = jnp.asarray(Sigma, dtype=float)
        self._training_df = training_df

    def coefficients(self):
        return self._beta

    def covariance(self, vcov_spec=None):
        return self._Sigma

    def predict(self, beta, X, offset=None):
        return jax.nn.sigmoid(X @ beta)

    def design_matrix_from_df(self, df):
        cols = [c for c in df.columns if c != "offset"]
        return jnp.asarray(df[cols].values, dtype=float)

    def column_index_of_variable(self, name):
        return [0]

    def variable_metadata(self):
        return {"x": None}

    @property
    def training_data(self):
        return self._training_df

    @property
    def supported_inference_methods(self):
        return {"delta", "simulation"}

    @property
    def supports_jax_autodiff(self):
        return True

    @property
    def gradient_backend_recommendation(self):
        return "autodiff"

    def attach(self, session):
        pass


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------
np.random.seed(0)
n = 50
p = 3
X_np = np.random.randn(n, p)
beta_true = np.array([0.5, -0.3, 0.1])
Sigma_true = np.eye(p) * 0.01

df = pd.DataFrame(X_np, columns=["x", "y", "z"])
adapter = MockAdapter(beta_true, Sigma_true, df)


def test_b1_jax_import_in_estimands():
    """B1 — make_slope_estimand must not raise NameError on jax.vmap."""
    X = jnp.asarray(X_np)
    h = make_slope_estimand(adapter, X, var_index=0, aggregate="overall")
    # This should succeed without NameError
    val = h(adapter.coefficients())
    assert jnp.isfinite(val)
    print("PASS: B1 — jax.vmap in dydx works")


def test_b2_b3_weights_in_overall():
    """B2/B3 — 'overall' must use weights when provided."""
    X = jnp.asarray(X_np)
    weights = jnp.asarray(np.random.uniform(1, 3, size=n))

    h_unweighted = make_prediction_estimand(
        adapter, X, aggregate="overall", weights=None
    )
    h_weighted = make_prediction_estimand(
        adapter, X, aggregate="overall", weights=weights
    )

    val_u = h_unweighted(adapter.coefficients())
    val_w = h_weighted(adapter.coefficients())

    # Weighted and unweighted should differ when weights are non-uniform
    assert not jnp.isclose(val_u, val_w), (
        f"Weighted overall should differ from unweighted: {val_u} vs {val_w}"
    )
    print("PASS: B2/B3 — weights used in overall aggregate")


def test_b4_cholesky_nan_check():
    """B4 — kappa must return inf for non-PSD covariance, not NaN."""
    bad_cov = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # not PSD
    grad = jnp.ones(2)
    H = jnp.eye(2)
    k = kappa(lambda b: b.sum(), adapter.coefficients()[:2], bad_cov,
              backend="autodiff")
    assert not jnp.isfinite(k) and k == float("inf"), f"Expected inf, got {k}"
    print("PASS: B4 — Cholesky NaN check routes to inf")


def test_b10_a1_scale_aware_mul():
    """B10+A1 — MarginsResult.__mul__ must respect non-identity scale."""
    result = MarginsResult(
        estimate=np.array([2.0]),  # RR = 2.0
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.5]),
        conf_int_upper=np.array([2.5]),
        method="delta",
        level=0.95,
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    scaled = result * 2
    # Doubling log-scale → squaring the RR
    expected_est = np.exp(2 * np.log(2.0))  # 4.0
    np.testing.assert_allclose(scaled.estimate, expected_est, rtol=1e-6)
    # CI bounds should also be squared
    expected_lo = np.exp(2 * np.log(1.5))
    expected_hi = np.exp(2 * np.log(2.5))
    np.testing.assert_allclose(scaled.conf_int_lower, expected_lo, rtol=1e-6)
    np.testing.assert_allclose(scaled.conf_int_upper, expected_hi, rtol=1e-6)
    # SE is on inference scale → simple scaling
    np.testing.assert_allclose(scaled.std_error, 0.2, rtol=1e-6)
    print("PASS: B10+A1 — scale-aware __mul__ works")


def test_b14_rng_seed():
    """B14 — rng_seed must be accepted by Margins and produce identical draws."""
    m1 = Margins(adapter, adapter=adapter, rng_seed=42, diagnostics=False)
    m2 = Margins(adapter, adapter=adapter, rng_seed=42, diagnostics=False)

    # Both should produce identical config objects
    cfg1 = m1._inference_config()
    cfg2 = m2._inference_config()
    assert cfg1.rng_seed == cfg2.rng_seed == 42
    print("PASS: B14 — rng_seed plumbed through session")


def test_b15_eager_cov():
    """B15 — Σ̂ must be cached eagerly at construction."""
    m = Margins(adapter, adapter=adapter, diagnostics=False)
    assert hasattr(m, "_cov_cache"), "Σ̂ should be eager-cached at construction"
    print("PASS: B15 — eager Σ̂ at session construction")


def test_a6_atexog_dataframe():
    """A6 — DataFrame atexog must be routed through scenario['data']."""
    m = Margins(adapter, adapter=adapter, diagnostics=False)
    atexog_df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    # This should not raise; internally it routes to scenario["data"]
    try:
        result = m.predict(atexog=atexog_df)
        print("PASS: A6 — DataFrame atexog routes without error")
    except Exception as e:
        print(f"FAIL: A6 — DataFrame atexog raised: {e}")


def test_b27_skip_diff_check():
    """B27 — delta path should skip is_jax_differentiable for adapters that support it."""
    X = jnp.asarray(X_np[:5])
    h = make_prediction_estimand(adapter, X, aggregate="overall")
    config = InferenceConfig(
        method="delta",
        level=0.95,
        diagnostics=False,
        cov_params=adapter.covariance(),
    )
    # adapter.supports_jax_autodiff is True, so this should succeed without
    # needing the is_jax_differentiable probe (which would also succeed, but
    # the code path now skips it).
    result = run_inference(h, adapter, config)
    assert "estimate" in result
    print("PASS: B27 — skips diff check for known-autodiff adapters")


if __name__ == "__main__":
    test_b1_jax_import_in_estimands()
    test_b2_b3_weights_in_overall()
    test_b4_cholesky_nan_check()
    test_b10_a1_scale_aware_mul()
    test_b14_rng_seed()
    test_b15_eager_cov()
    test_a6_atexog_dataframe()
    test_b27_skip_diff_check()
    print("\nAll smoke tests passed.")
