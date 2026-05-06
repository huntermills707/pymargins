"""Tests for pymargins._gradients against analytical truth.

See IMPLEMENTATION_GUIDE.md §0.1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm

# Enable float64 for tests so that FD defaults (1e-6) are well-calibrated
# and analytical comparisons hold to tight tolerances.
jax.config.update("jax_enable_x64", True)

from pymargins._gradients import (
    gradient,
    hessian,
    directional_derivative,
    hessian_vector_product,
    make_predict_with_fd_jvp,
    make_glm_jvp_wrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def beta_hat(rng):
    return jnp.asarray(rng.standard_normal(5))


@pytest.fixture
def x_row(rng):
    return jnp.asarray(rng.standard_normal(5))


# ---------------------------------------------------------------------------
# 1. Autodiff vs analytical gradients
# ---------------------------------------------------------------------------

def test_ols_gradient_exact():
    """For OLS, ∂(x·β)/∂β = x exactly."""
    x = jnp.array([1.0, 2.0, -0.5, 0.0, 3.0])
    beta = jnp.array([0.1, -0.2, 0.3, 0.4, -0.5])

    def h(b):
        return x @ b

    g = gradient(h, beta, backend="autodiff")
    np.testing.assert_allclose(g, x, rtol=1e-10)


def test_logit_gradient_exact():
    """For logit, ∂σ(x·β)/∂β = σ(1-σ)·x exactly."""
    x = jnp.array([1.0, 2.0, -0.5, 0.0, 3.0])
    beta = jnp.array([0.1, -0.2, 0.3, 0.4, -0.5])

    def h(b):
        return jax.scipy.special.expit(x @ b)

    g = gradient(h, beta, backend="autodiff")
    p = float(jax.scipy.special.expit(x @ beta))
    expected = p * (1 - p) * x
    np.testing.assert_allclose(g, expected, rtol=1e-10)


def test_vector_estimand_jacobian():
    """For h: R^n -> R^k, gradient returns the (k, n) Jacobian."""
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    beta = jnp.array([0.5, -0.3])

    def h(b):
        return X @ b  # (3,)

    J = gradient(h, beta, backend="autodiff")
    np.testing.assert_allclose(J, X, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. make_glm_jvp_wrapper vs pure JAX reimplementation
# ---------------------------------------------------------------------------

def test_glm_jvp_wrapper_matches_native_jax():
    """The custom-JVP GLM wrapper must agree with a pure-JAX GLM predict."""
    rng = np.random.default_rng(42)
    n, p = 20, 3
    X_np = rng.standard_normal((n, p))
    beta = jnp.asarray(rng.standard_normal(p))
    offset = jnp.asarray(rng.standard_normal(n))

    # Pure JAX logit prediction
    def native_predict(b, X, off=None):
        eta = X @ b
        if off is not None:
            eta = eta + off
        return jax.scipy.special.expit(eta)

    # Wrapper using statsmodels family
    family = sm.families.Binomial()
    wrapped = make_glm_jvp_wrapper(family)

    X = jnp.asarray(X_np)

    # Forward values match
    y_native = native_predict(beta, X, offset)
    y_wrapped = wrapped(beta, X, offset)
    np.testing.assert_allclose(y_native, y_wrapped, rtol=1e-10)

    # Gradients w.r.t. beta match
    g_native = jax.grad(lambda b: native_predict(b, X, offset).sum())(beta)
    g_wrapped = jax.grad(lambda b: wrapped(b, X, offset).sum())(beta)
    np.testing.assert_allclose(g_native, g_wrapped, rtol=1e-8)

    # Gradients w.r.t. X match (needed for dydx)
    gX_native = jax.grad(lambda X_: native_predict(beta, X_, offset).sum())(X)
    gX_wrapped = jax.grad(lambda X_: wrapped(beta, X_, offset).sum())(X)
    np.testing.assert_allclose(gX_native, gX_wrapped, rtol=1e-8)


def test_glm_jvp_identity_link():
    """Gaussian/identity link should reduce to linear prediction."""
    rng = np.random.default_rng(7)
    n, p = 10, 4
    X = jnp.asarray(rng.standard_normal((n, p)))
    beta = jnp.asarray(rng.standard_normal(p))

    family = sm.families.Gaussian()
    wrapped = make_glm_jvp_wrapper(family)

    y_wrapped = wrapped(beta, X)
    y_linear = X @ beta
    np.testing.assert_allclose(y_wrapped, y_linear, rtol=1e-10)

    g_wrapped = jax.grad(lambda b: wrapped(b, X).sum())(beta)
    g_linear = jax.grad(lambda b: (X @ b).sum())(beta)
    np.testing.assert_allclose(g_wrapped, g_linear, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. FD vs autodiff agreement
# ---------------------------------------------------------------------------

def test_fd_gradient_agrees_with_autodiff():
    """Central-difference FD should agree with autodiff to ~10 digits."""
    rng = np.random.default_rng(99)
    X = jnp.asarray(rng.standard_normal((15, 4)))
    beta = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(X @ b).sum()

    g_auto = gradient(h, beta, backend="autodiff")
    g_fd = gradient(h, beta, backend="fd", fd_step=1e-6)
    np.testing.assert_allclose(g_auto, g_fd, rtol=1e-9, atol=1e-9)


def test_fd_hessian_agrees_with_autodiff():
    """FD Hessian should agree with autodiff Hessian to ~8 digits."""
    rng = np.random.default_rng(101)
    x = jnp.asarray(rng.standard_normal(4))
    beta = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    H_auto = hessian(h, beta, backend="autodiff")
    H_fd = hessian(h, beta, backend="fd", fd_step=1e-5)
    np.testing.assert_allclose(H_auto, H_fd, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Custom-JVP wrapper composition
# ---------------------------------------------------------------------------

def test_fd_jvp_wrapper_composes_with_grad_and_hessian():
    """A non-JAX predict wrapped with FD-JVP must support jax.grad/hessian."""
    rng = np.random.default_rng(202)
    n, p = 8, 3
    X_np = rng.standard_normal((n, p))
    beta = jnp.asarray(rng.standard_normal(p))

    # Native predict using NumPy
    def native_predict(beta_np, X):
        return np.asarray(X) @ np.asarray(beta_np)

    wrapped = make_predict_with_fd_jvp(native_predict, fd_step=1e-6)
    X = jnp.asarray(X_np)

    # jax.grad must work
    g = jax.grad(lambda b: wrapped(b, X).sum())(beta)
    assert g.shape == beta.shape
    np.testing.assert_allclose(g, X.sum(axis=0), rtol=1e-6)

    # jax.hessian must work
    H = jax.hessian(lambda b: wrapped(b, X).sum())(beta)
    assert H.shape == (p, p)
    # Linear function => Hessian is zero
    np.testing.assert_allclose(H, 0.0, atol=1e-6)


def test_fd_jvp_wrapper_nonlinear():
    """FD-JVP on a nonlinear native predict."""
    rng = np.random.default_rng(303)
    n, p = 8, 3
    X_np = rng.standard_normal((n, p))
    beta = jnp.asarray(rng.standard_normal(p))

    def native_predict(beta_np, X):
        eta = np.asarray(X) @ np.asarray(beta_np)
        return 1.0 / (1.0 + np.exp(-eta))  # sigmoid in numpy

    wrapped = make_predict_with_fd_jvp(native_predict, fd_step=1e-6)
    X = jnp.asarray(X_np)

    # Compare to pure JAX sigmoid
    def pure(b):
        return jax.scipy.special.expit(X @ b)

    g_wrapped = jax.grad(lambda b: wrapped(b, X).sum())(beta)
    g_pure = jax.grad(lambda b: pure(b).sum())(beta)
    np.testing.assert_allclose(g_wrapped, g_pure, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. directional_derivative and hessian_vector_product
# ---------------------------------------------------------------------------

def test_directional_derivative_matches_grad_dot_v():
    """directional_derivative should equal grad·v."""
    rng = np.random.default_rng(404)
    X = jnp.asarray(rng.standard_normal((10, 4)))
    beta = jnp.asarray(rng.standard_normal(4))
    v = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(X @ b).sum()

    g = gradient(h, beta, backend="autodiff")
    dd = directional_derivative(h, beta, v, backend="autodiff")
    np.testing.assert_allclose(dd, jnp.dot(g, v), rtol=1e-10)


def test_directional_derivative_fd():
    """FD directional_derivative should match autodiff."""
    rng = np.random.default_rng(505)
    X = jnp.asarray(rng.standard_normal((10, 4)))
    beta = jnp.asarray(rng.standard_normal(4))
    v = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(X @ b).sum()

    dd_auto = directional_derivative(h, beta, v, backend="autodiff")
    dd_fd = directional_derivative(h, beta, v, backend="fd", fd_step=1e-6)
    np.testing.assert_allclose(dd_auto, dd_fd, rtol=1e-8)


def test_hessian_vector_product_matches_full_hessian():
    """HVP should equal H @ v without materializing H."""
    rng = np.random.default_rng(606)
    x = jnp.asarray(rng.standard_normal(4))
    beta = jnp.asarray(rng.standard_normal(4))
    v = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    H = hessian(h, beta, backend="autodiff")
    hvp = hessian_vector_product(h, beta, v, backend="autodiff")
    np.testing.assert_allclose(hvp, H @ v, rtol=1e-10)


def test_hessian_vector_product_fd():
    """FD HVP should match autodiff HVP."""
    rng = np.random.default_rng(707)
    x = jnp.asarray(rng.standard_normal(4))
    beta = jnp.asarray(rng.standard_normal(4))
    v = jnp.asarray(rng.standard_normal(4))

    def h(b):
        return jax.scipy.special.expit(x @ b)

    hvp_auto = hessian_vector_product(h, beta, v, backend="autodiff")
    hvp_fd = hessian_vector_product(h, beta, v, backend="fd", fd_step=1e-5)
    np.testing.assert_allclose(hvp_auto, hvp_fd, rtol=1e-5, atol=1e-5)
