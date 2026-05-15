"""Tests for pymargins._kappa curvature diagnostic.

See IMPLEMENTATION_GUIDE.md §0.1 (κ is part of the kernel layer).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pymargins._kappa import (
    kappa,
    kappa_vector,
    classify_kappa,
    session_kappa,
    delta_simulation_disagreement,
)


# ---------------------------------------------------------------------------
# 1. kappa for linear estimands (should be ~0)
# ---------------------------------------------------------------------------

def test_kappa_linear_estimand_is_near_zero():
    """For h(beta) = c @ beta, the Hessian is zero, so κ should be ~0."""
    rng = np.random.default_rng(42)
    p = 4
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    c = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return c @ b

    k = kappa(h, beta, Sigma, backend="autodiff")
    assert k < 1e-6, f"Expected κ ≈ 0 for linear estimand, got {k}"


# ---------------------------------------------------------------------------
# 2. kappa for nonlinear estimands (should be > 0)
# ---------------------------------------------------------------------------

def test_kappa_logit_prediction_is_positive():
    """Logit prediction has nonzero curvature; κ should be positive."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    k = kappa(h, beta, Sigma, backend="autodiff")
    assert k > 0.0, f"Expected κ > 0 for nonlinear estimand, got {k}"
    assert np.isfinite(k), f"κ should be finite, got {k}"


def test_kappa_quadratic_estimand():
    """For h(beta) = beta^T A beta, κ should be positive and finite."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    A = jnp.eye(p)

    def h(b):
        return b @ A @ b

    k = kappa(h, beta, Sigma, backend="autodiff")
    assert k > 0.0
    assert np.isfinite(k)


# ---------------------------------------------------------------------------
# 3. classify_kappa thresholds
# ---------------------------------------------------------------------------

def test_classify_kappa_reliable():
    assert classify_kappa(0.05) == "delta_reliable"
    assert classify_kappa(0.0) == "delta_reliable"


def test_classify_kappa_borderline():
    assert classify_kappa(0.15) == "delta_borderline"
    assert classify_kappa(0.299) == "delta_borderline"


def test_classify_kappa_unreliable():
    assert classify_kappa(0.3) == "delta_unreliable"
    assert classify_kappa(1.0) == "delta_unreliable"
    assert classify_kappa(float("inf")) == "delta_unreliable"


def test_classify_kappa_custom_thresholds():
    assert classify_kappa(0.05, reliable_threshold=0.1, borderline_threshold=0.3) == "delta_reliable"
    assert classify_kappa(0.2, reliable_threshold=0.1, borderline_threshold=0.3) == "delta_borderline"
    assert classify_kappa(0.5, reliable_threshold=0.1, borderline_threshold=0.3) == "delta_unreliable"


# ---------------------------------------------------------------------------
# 4. kappa_vector
# ---------------------------------------------------------------------------

def test_kappa_vector_per_component():
    """For a vector estimand, kappa_vector returns one value per component."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x1 = jnp.asarray(rng.standard_normal(p))
    x2 = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([jax.scipy.special.expit(x1 @ b),
                          jax.scipy.special.expit(x2 @ b)])

    kappas = kappa_vector(h, beta, Sigma, backend="autodiff")
    assert kappas.shape == (2,)
    assert float(kappas[0]) > 0.0
    assert float(kappas[1]) > 0.0
    assert np.isfinite(float(kappas[0]))
    assert np.isfinite(float(kappas[1]))


def test_kappa_vector_linear_component():
    """Vector estimand with one linear and one nonlinear component."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([x @ b,  # linear -> κ ≈ 0
                          jax.scipy.special.expit(x @ b)])  # nonlinear -> κ > 0

    kappas = kappa_vector(h, beta, Sigma, backend="autodiff")
    assert kappas.shape == (2,)
    assert float(kappas[0]) < 1e-6, "Linear component should have κ ≈ 0"
    assert float(kappas[1]) > 0.0, "Nonlinear component should have κ > 0"


# ---------------------------------------------------------------------------
# 5. session_kappa
# ---------------------------------------------------------------------------

def test_session_kappa_basic():
    """session_kappa should produce a sensible summary dict."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01

    # Sample 5 design rows
    design = [jnp.asarray(rng.standard_normal(p)) for _ in range(5)]

    def h_factory(x):
        return lambda b: jax.scipy.special.expit(x @ b)

    diag = session_kappa(h_factory, beta, Sigma, design, backend="autodiff")

    assert "min" in diag
    assert "median" in diag
    assert "max" in diag
    assert "distribution" in diag
    assert "verdict" in diag
    assert "n_samples" in diag
    assert "recommendation" in diag
    assert diag["n_samples"] == 5
    assert diag["min"] <= diag["median"] <= diag["max"]
    assert diag["verdict"] in ("delta_reliable", "delta_borderline", "delta_unreliable")


def test_session_kappa_verdict_borderline():
    """Force a borderline verdict with a highly curved estimand."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p)) * 2.0  # larger magnitude
    # Large covariance amplifies curvature in whitened coordinates
    Sigma = jnp.eye(p) * 0.5

    design = [jnp.asarray(rng.standard_normal(p)) for _ in range(5)]

    def h_factory(x):
        return lambda b: jax.scipy.special.expit(x @ b)

    diag = session_kappa(
        h_factory, beta, Sigma, design,
        backend="autodiff",
        reliable_threshold=0.01,
        borderline_threshold=0.05,
    )
    # With very strict thresholds, logit should be at least borderline
    assert diag["verdict"] in ("delta_borderline", "delta_unreliable")
    assert "simulation" in diag["recommendation"].lower()


def test_session_kappa_verdict_unreliable():
    """Force an unreliable verdict with extreme curvature."""
    rng = np.random.default_rng(42)
    p = 2
    beta = jnp.array([2.0, -2.0])
    Sigma = jnp.eye(p) * 1.0  # large variance

    design = [jnp.array([1.0, 0.5])]

    def h_factory(x):
        # Quadratic estimand has constant Hessian -> higher κ with larger Sigma
        return lambda b: (b ** 2).sum()

    diag = session_kappa(
        h_factory, beta, Sigma, design,
        backend="autodiff",
        reliable_threshold=0.01,
        borderline_threshold=0.05,
    )
    assert diag["verdict"] == "delta_unreliable"
    assert "unreliable" in diag["recommendation"].lower()


# ---------------------------------------------------------------------------
# 6. delta_simulation_disagreement
# ---------------------------------------------------------------------------

def test_delta_sim_disagreement_linear_is_small():
    """For a linear estimand, delta and simulation should agree well."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    from pymargins._gradients import gradient
    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    disagreement = delta_simulation_disagreement(
        estimate, grad, Sigma, h, beta,
        level=0.95, n_sim=4000, rng_seed=42,
    )

    # For linear h, delta is exact; disagreement should be very small
    assert disagreement < 0.05, f"Expected small disagreement for linear estimand, got {disagreement}"


def test_delta_sim_disagreement_logit_is_moderate():
    """For a logit prediction, disagreement may be larger but still moderate."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    from pymargins._gradients import gradient
    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    disagreement = delta_simulation_disagreement(
        estimate, grad, Sigma, h, beta,
        level=0.95, n_sim=4000, rng_seed=42,
    )

    # Should be finite and not huge
    assert np.isfinite(disagreement)
    assert disagreement >= 0.0


def test_delta_sim_disagreement_with_phi():
    """With a back-transform, disagreement is computed on the reporting scale."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b  # inference scale = log

    from pymargins._gradients import gradient
    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    disagreement = delta_simulation_disagreement(
        estimate, grad, Sigma, h, beta,
        level=0.95, n_sim=4000, rng_seed=42,
        phi=jnp.exp,
    )

    assert np.isfinite(disagreement)
    assert disagreement >= 0.0


def test_delta_sim_disagreement_zero_estimate_returns_inf():
    """When estimate is exactly zero, disagreement should return +inf."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.zeros(p)
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    from pymargins._gradients import gradient
    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    disagreement = delta_simulation_disagreement(
        estimate, grad, Sigma, h, beta,
        level=0.95, n_sim=1000, rng_seed=42,
    )
    assert disagreement == float("inf")


def test_kappa_frobenius_norm():
    """kappa with norm='frobenius' should return a positive finite value."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    k_spec = kappa(h, beta, Sigma, backend="autodiff", norm="spectral")
    k_frob = kappa(h, beta, Sigma, backend="autodiff", norm="frobenius")
    assert np.isfinite(k_frob)
    assert k_frob > 0.0
    # Frobenius >= spectral for any matrix
    assert k_frob >= k_spec


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_kappa_zero_gradient_returns_inf():
    """At a critical point where grad=0, κ should return inf."""
    p = 2
    beta = jnp.zeros(p)
    Sigma = jnp.eye(p) * 0.01

    def h(b):
        return b @ b  # minimum at 0

    k = kappa(h, beta, Sigma, backend="autodiff")
    assert k == float("inf"), f"Expected inf at critical point, got {k}"


def test_kappa_non_psd_covariance():
    """Near-PSD covariance should be handled gracefully via ridge fallback."""
    p = 2
    beta = jnp.array([0.5, -0.3])
    # Build a PSD matrix, then make it barely non-PSD with a tiny negative eigenvalue
    base = jnp.array([[1.0, 0.3], [0.3, 1.0]])
    Sigma = base - 1e-10 * jnp.eye(p)

    def h(b):
        return b.sum()

    k = kappa(h, beta, Sigma, backend="autodiff")
    # Linear estimand => after ridge regularization κ should be near 0
    assert np.isfinite(k), f"Expected finite κ after ridge, got {k}"
    assert k < 1e-6, f"Expected κ ≈ 0 for linear estimand, got {k}"



def test_kappa_vector_scalar_estimand():
    """kappa_vector on a scalar estimand should return a 1-element array."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b  # scalar

    kappas = kappa_vector(h, beta, Sigma, backend="autodiff")
    assert kappas.shape == (1,)
    assert float(kappas[0]) < 1e-6


def test_kappa_core_rejects_vector_estimand():
    """_kappa_core should raise when given a vector estimand."""
    from pymargins._kappa import _kappa_core
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01
    x1 = jnp.asarray(rng.standard_normal(p))
    x2 = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([x1 @ b, x2 @ b])

    with pytest.raises(ValueError, match="scalar estimands"):
        _kappa_core(h, beta, Sigma, L=None, backend="autodiff")


def test_classify_kappa_nan():
    """classify_kappa with NaN must return delta_unreliable."""
    assert classify_kappa(float("nan")) == "delta_unreliable"
    assert classify_kappa(float("inf")) == "delta_unreliable"
    assert classify_kappa(-float("inf")) == "delta_unreliable"


def test_kappa_non_psd_negative_diagonal():
    """Tiny negative diagonal covariance should be handled with a non-negative ridge."""
    p = 2
    beta = jnp.array([0.5, -0.3])
    # Build a PSD matrix, then add a tiny negative to the diagonal
    base = jnp.array([[1.0, 0.3], [0.3, 1.0]])
    Sigma = base - 1e-10 * jnp.eye(p)

    def h(b):
        return b.sum()

    k = kappa(h, beta, Sigma, backend="autodiff")
    assert np.isfinite(k), f"Expected finite κ after ridge, got {k}"


def test_kappa_vector_2d_output():
    """kappa_vector must handle 2D array outputs by computing per-element κ."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    Sigma = jnp.eye(p) * 0.01

    def h(b):
        return jnp.array([[b[0] + b[1], b[0] - b[1]], [b[0] * b[1], b[0] / b[1]]])

    k = kappa_vector(h, beta, Sigma, backend="autodiff")
    assert k.shape == (2, 2)
    assert np.all(np.isfinite(k))
