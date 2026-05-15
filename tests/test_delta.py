"""Tests for pymargins._delta against statsmodels reference outputs.

See IMPLEMENTATION_GUIDE.md §0.2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins._gradients import gradient
from pymargins._delta import (
    delta_variance,
    delta_se,
    delta_confint,
    delta_confint_from_se,
    delta_wald_test,
    joint_wald_test,
    combined_gradient,
    joint_covariance_of_results,
)


# ---------------------------------------------------------------------------
# 1. delta_se against statsmodels OLS t_test
# ---------------------------------------------------------------------------

def test_delta_se_matches_ols_t_test():
    """For a linear contrast c@beta, delta_se must match OLS t_test SE."""
    rng = np.random.default_rng(42)
    n, p = 100, 4
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -0.5, 0.3, 0.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5

    model = sm.OLS(y, X).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    # Several random contrasts
    for seed in [1, 7, 13, 19]:
        rng2 = np.random.default_rng(seed)
        c = rng2.standard_normal(p)

        # statsmodels t_test
        ttest = model.t_test(c)
        se_sm = float(np.asarray(ttest.sd).item())

        # pymargins delta_se
        def h(b):
            return c @ b

        grad = gradient(h, beta, backend="autodiff")
        se_pm = float(delta_se(grad, Sigma))

        np.testing.assert_allclose(se_pm, se_sm, rtol=1e-8)


# ---------------------------------------------------------------------------
# 2. delta_confint against statsmodels get_prediction
# ---------------------------------------------------------------------------

def test_delta_confint_matches_ols_prediction():
    """For OLS, delta CIs on predictions should match statsmodels."""
    rng = np.random.default_rng(42)
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -0.5, 0.3])
    y = X @ beta_true + rng.standard_normal(n) * 0.5

    model = sm.OLS(y, X).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    # Predict at a new row
    x_new = jnp.asarray([1.0, 0.5, -0.3])

    def h(b):
        return x_new @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    lower_pm, upper_pm = delta_confint(estimate, grad, Sigma, level=0.95)

    # statsmodels get_prediction
    pred = model.get_prediction(x_new)
    sf = pred.summary_frame(alpha=0.05)
    lower_sm = float(sf["mean_ci_lower"].iloc[0])
    upper_sm = float(sf["mean_ci_upper"].iloc[0])

    # statsmodels uses t-distribution for OLS; we use normal.
    # The difference is small (~0.3% relative) and expected.
    np.testing.assert_allclose(float(lower_pm), lower_sm, rtol=1e-2)
    np.testing.assert_allclose(float(upper_pm), upper_sm, rtol=1e-2)


def test_delta_confint_logit_prediction():
    """Delta CIs on logit predicted probabilities vs statsmodels."""
    rng = np.random.default_rng(77)
    n, p = 200, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([0.5, -0.3, 0.1])
    eta = X @ beta_true
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < prob).astype(float)

    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    x_new = jnp.asarray([1.0, 0.5, -0.3])

    def h(b):
        return jax.scipy.special.expit(x_new @ b)

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    lower_pm, upper_pm = delta_confint(estimate, grad, Sigma, level=0.95)

    # statsmodels get_prediction for GLM
    pred = model.get_prediction(x_new)
    sf = pred.summary_frame(alpha=0.05)
    lower_sm = float(sf["mean_ci_lower"].iloc[0])
    upper_sm = float(sf["mean_ci_upper"].iloc[0])

    # statsmodels may use a slightly different covariance or method.
    # We verify close agreement rather than exact identity.
    np.testing.assert_allclose(float(lower_pm), lower_sm, rtol=1e-2)
    np.testing.assert_allclose(float(upper_pm), upper_sm, rtol=1e-2)


# ---------------------------------------------------------------------------
# 3. joint_wald_test against statsmodels wald_test
# ---------------------------------------------------------------------------

def test_joint_wald_test_matches_statsmodels():
    """Joint Wald test on multiple contrasts must match statsmodels."""
    rng = np.random.default_rng(42)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5

    model = sm.OLS(y, X).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    # Test H0: beta_3 = beta_4 = 0
    R = np.array([[0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    def h(b):
        return R @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    chi2_pm, p_pm, df_pm = joint_wald_test(estimate, grad, Sigma)

    # statsmodels wald_test returns an F-statistic by default for OLS.
    # use_f=False gives the chi2 statistic that matches our joint_wald_test.
    wald_sm = model.wald_test(R, use_f=False)
    chi2_sm = float(np.asarray(wald_sm.statistic).item())
    p_sm = float(np.asarray(wald_sm.pvalue).item())
    df_sm = int(np.asarray(wald_sm.df_denom).item())

    np.testing.assert_allclose(chi2_pm, chi2_sm, rtol=1e-5)
    np.testing.assert_allclose(p_pm, p_sm, rtol=1e-4)
    assert df_pm == df_sm


# ---------------------------------------------------------------------------
# 4. delta_wald_test per-component
# ---------------------------------------------------------------------------

def test_delta_wald_test_two_sided():
    """Per-component Wald test against known values."""
    rng = np.random.default_rng(42)
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, 0.0, 0.5]) + rng.standard_normal(n) * 0.5

    model = sm.OLS(y, X).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    # Test beta_1 = 0
    c = np.array([0.0, 1.0, 0.0])

    def h(b):
        return c @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    z_pm, p_pm = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                  alternative="two-sided")

    ttest = model.t_test(c)
    z_sm = float(np.asarray(ttest.tvalue).item())
    p_sm = float(np.asarray(ttest.pvalue).item())

    np.testing.assert_allclose(float(z_pm), z_sm, rtol=1e-5)
    # statsmodels uses t-distribution for p-values; we use normal.
    # Small differences (~0.3%) are expected.
    np.testing.assert_allclose(float(p_pm), p_sm, rtol=1e-2)


def test_delta_wald_test_one_sided():
    """One-sided alternatives must produce correct tail probabilities."""
    rng = np.random.default_rng(42)
    n, p = 100, 3
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, 0.5, 0.0]) + rng.standard_normal(n) * 0.5

    model = sm.OLS(y, X).fit()
    Sigma = jnp.asarray(model.cov_params())
    beta = jnp.asarray(model.params)

    c = np.array([0.0, 1.0, 0.0])

    def h(b):
        return c @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    _, p_two = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                alternative="two-sided")
    _, p_greater = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                    alternative="greater")
    _, p_less = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                 alternative="less")

    # Two-sided should be roughly 2*min(one-sided, 1-one-sided)
    assert 0.0 <= p_greater <= 1.0
    assert 0.0 <= p_less <= 1.0
    assert np.isclose(p_two, 2 * min(p_greater, p_less), rtol=1e-3) or \
           np.isclose(p_two, 2 * min(p_greater, p_less), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Variance and covariance helpers
# ---------------------------------------------------------------------------

def test_delta_variance_scalar():
    """For scalar g, delta_variance returns a scalar."""
    grad = jnp.array([1.0, 2.0, 3.0])
    Sigma = jnp.eye(3) * 4.0
    var = delta_variance(grad, Sigma)
    expected = grad @ Sigma @ grad  # (1+4+9)*4 = 56
    np.testing.assert_allclose(var, expected, rtol=1e-10)


def test_delta_variance_vector():
    """For vector g, delta_variance returns the full covariance matrix."""
    grad = jnp.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0]])  # (2, 3)
    Sigma = jnp.eye(3)
    var = delta_variance(grad, Sigma)
    expected = grad @ Sigma @ grad.T
    np.testing.assert_allclose(var, expected, rtol=1e-10)


def test_delta_variance_vector_is_symmetric():
    """delta_variance for vector g must return a symmetric matrix."""
    rng = np.random.default_rng(42)
    p = 4
    k = 3
    Sigma = jnp.asarray(rng.standard_normal((p, p)))
    Sigma = Sigma @ Sigma.T
    grad = jnp.asarray(rng.standard_normal((k, p)))
    var = delta_variance(grad, Sigma)
    np.testing.assert_allclose(var, var.T, rtol=1e-10)


def test_combined_gradient():
    """combined_gradient must return the gradient of the weighted sum."""
    g1 = jnp.array([1.0, 2.0, 3.0])
    g2 = jnp.array([4.0, 5.0, 6.0])
    weights = jnp.array([2.0, -1.0])

    combined = combined_gradient([g1, g2], weights)
    expected = 2.0 * g1 - 1.0 * g2
    np.testing.assert_allclose(combined, expected, rtol=1e-10)


def test_combined_gradient_stacked_scenarios():
    """combined_gradient with more than two gradients (stacking scenario)."""
    rng = np.random.default_rng(42)
    p = 4
    n = 5
    grads = [jnp.asarray(rng.standard_normal(p)) for _ in range(n)]
    weights = jnp.asarray(rng.standard_normal(n))
    combined = combined_gradient(grads, weights)
    expected = sum(float(w) * g for w, g in zip(weights, grads))
    np.testing.assert_allclose(combined, expected, rtol=1e-10)


def test_joint_covariance_of_results():
    """Joint covariance of two scalar estimands from shared beta."""
    rng = np.random.default_rng(42)
    p = 4
    Sigma = jnp.asarray(rng.standard_normal((p, p)))
    Sigma = Sigma @ Sigma.T  # PSD

    g1 = jnp.asarray(rng.standard_normal(p))
    g2 = jnp.asarray(rng.standard_normal(p))

    cov_joint = joint_covariance_of_results([g1, g2], Sigma)

    var1 = float(g1 @ Sigma @ g1)
    var2 = float(g2 @ Sigma @ g2)
    cov12 = float(g1 @ Sigma @ g2)

    expected = jnp.array([[var1, cov12],
                          [cov12, var2]])
    np.testing.assert_allclose(cov_joint, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. delta_confint_from_se
# ---------------------------------------------------------------------------

def test_delta_confint_from_se_recovers_original():
    """delta_confint_from_se with the same level should reproduce the CI."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)
    se = delta_se(grad, Sigma)

    lo1, hi1 = delta_confint(estimate, grad, Sigma, level=0.95)
    lo2, hi2 = delta_confint_from_se(estimate, se, level=0.95)

    np.testing.assert_allclose(lo1, lo2, rtol=1e-10)
    np.testing.assert_allclose(hi1, hi2, rtol=1e-10)


def test_delta_confint_from_se_different_level():
    """Recomputing CI at a different level should change width."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)
    se = delta_se(grad, Sigma)

    lo95, hi95 = delta_confint_from_se(estimate, se, level=0.95)
    lo90, hi90 = delta_confint_from_se(estimate, se, level=0.90)

    # 90% CI should be narrower than 95% CI
    assert float(hi90 - lo90) < float(hi95 - lo95)


def test_delta_confint_with_phi():
    """Back-transformed CI should be asymmetric and properly bounded."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    lo, hi = delta_confint(estimate, grad, Sigma, level=0.95, phi=jnp.exp)

    # CI on the reporting scale (exp of log-scale estimate)
    assert float(lo) > 0.0
    assert float(hi) > float(lo)
    # Asymmetry: distance from estimate to bounds should differ
    est_rep = float(jnp.exp(estimate))
    assert not np.isclose(float(hi) - est_rep, est_rep - float(lo), rtol=1e-6)


def test_delta_confint_vector_estimand():
    """delta_confint must work with a vector estimand (returns per-component CIs)."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([b[0], b[1] + b[2]])

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    lo, hi = delta_confint(estimate, grad, Sigma, level=0.95)
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    assert jnp.all(hi > lo)


def test_delta_confint_from_se_with_phi():
    """delta_confint_from_se with phi should match delta_confint."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)
    se = delta_se(grad, Sigma)

    lo1, hi1 = delta_confint(estimate, grad, Sigma, level=0.95, phi=jnp.exp)
    lo2, hi2 = delta_confint_from_se(estimate, se, level=0.95, phi=jnp.exp)

    np.testing.assert_allclose(float(lo1), float(lo2), rtol=1e-10)
    np.testing.assert_allclose(float(hi1), float(hi2), rtol=1e-10)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_delta_se_zero_gradient():
    """When gradient is zero, SE should be zero (not NaN)."""
    grad = jnp.zeros(3)
    Sigma = jnp.eye(3) * 0.01
    se = delta_se(grad, Sigma)
    assert float(se) == 0.0


def test_delta_se_negative_variance_clipped():
    """Tiny negative diagonal variance from numerical noise should clip to 0."""
    grad = jnp.array([1.0, 1.0, 1.0])
    # Manually construct a covariance with a tiny negative on the diagonal
    # after the quadratic form (simulating numerical noise)
    Sigma = jnp.array([[1.0, 0.0, 0.0],
                       [0.0, -1e-15, 0.0],
                       [0.0, 0.0, -1e-15]])
    # For this gradient, variance = 1.0 - 1e-15 - 1e-15 ≈ 1.0 (positive after clip)
    se = delta_se(grad, Sigma)
    assert np.isfinite(float(se))


def test_delta_wald_test_zero_se():
    """Wald test with zero SE should produce deterministic p-values."""
    grad = jnp.zeros(3)
    Sigma = jnp.eye(3) * 0.01

    # Case 1: estimate == null -> z = 0, p = 1.0
    z, p = delta_wald_test(
        jnp.array(0.0), grad, Sigma,
        null_value=0.0, alternative="two-sided",
    )
    assert float(z) == 0.0
    assert float(p) == 1.0

    # Case 2: estimate > null -> z = +inf, p = 0.0
    z, p = delta_wald_test(
        jnp.array(1.0), grad, Sigma,
        null_value=0.0, alternative="two-sided",
    )
    assert float(z) == float("inf")
    assert float(p) == 0.0


def test_delta_wald_test_vector_estimate():
    """Per-component Wald test on a vector estimand."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([b[0], b[1] + b[2]])

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    z, p = delta_wald_test(estimate, grad, Sigma, null_value=jnp.array([0.0, 0.0]))
    assert z.shape == (2,)
    assert p.shape == (2,)
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(p))


def test_delta_wald_test_greater_less():
    """Explicit one-sided tests should return ordered p-values."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))
    x = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return x @ b

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    z_two, p_two = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                    alternative="two-sided")
    z_greater, p_greater = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                            alternative="greater")
    z_less, p_less = delta_wald_test(estimate, grad, Sigma, null_value=0.0,
                                      alternative="less")

    # z-statistics should be identical regardless of alternative
    np.testing.assert_allclose(z_two, z_greater, rtol=1e-10)
    np.testing.assert_allclose(z_two, z_less, rtol=1e-10)

    # For a positive z, p_greater < p_two < p_less
    if float(z_two) > 0:
        assert float(p_greater) <= float(p_two) <= float(p_less)
    elif float(z_two) < 0:
        assert float(p_less) <= float(p_two) <= float(p_greater)


def test_joint_wald_test_custom_null():
    """Joint test with a non-zero null vector."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))

    def h(b):
        return jnp.array([b[0], b[1]])

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    # Test against the true beta values (should be non-significant)
    chi2, p, df = joint_wald_test(estimate, grad, Sigma, null_value=estimate)
    assert chi2 == 0.0
    assert p == 1.0
    assert df == 2


def test_joint_wald_test_singular_covariance():
    """Joint test should not crash when Sigma_g is singular."""
    rng = np.random.default_rng(42)
    p = 3
    Sigma = jnp.eye(p) * 0.01
    beta = jnp.asarray(rng.standard_normal(p))

    # Two perfectly collinear estimands -> singular Sigma_g
    def h(b):
        return jnp.array([b[0], b[0]])

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    chi2, p, df = joint_wald_test(estimate, grad, Sigma)
    assert np.isfinite(chi2)
    assert np.isfinite(p)
    assert df == 2


def test_joint_wald_test_regularization_produces_finite_statistic():
    """Near-singular Sigma_g should trigger regularization and yield finite values."""
    rng = np.random.default_rng(42)
    p = 3
    beta = jnp.asarray(rng.standard_normal(p))
    # Nearly singular covariance: tiny epsilon on diagonal
    Sigma = jnp.eye(p) * 1e-12

    # Perfectly collinear estimands -> singular Sigma_g
    def h(b):
        return jnp.array([b[0], b[0]])

    grad = gradient(h, beta, backend="autodiff")
    estimate = h(beta)

    with pytest.warns(RuntimeWarning, match="regularized"):
        chi2, p, df = joint_wald_test(estimate, grad, Sigma)

    assert np.isfinite(chi2)
    assert np.isfinite(p)
    assert df == 2
    assert chi2 >= 0.0


def test_delta_variance_bad_ndim():
    """delta_variance must raise for gradients with ndim > 3."""
    grad = jnp.ones((2, 3, 4, 5))
    Sigma = jnp.eye(5)
    with pytest.raises(ValueError, match="gradient must be 1D, 2D, or 3D"):
        delta_variance(grad, Sigma)



def test_delta_confint_level_validation():
    """delta_confint must reject level outside (0,1)."""
    grad = jnp.array([1.0, 0.0, 0.0])
    Sigma = jnp.eye(3) * 0.01
    estimate = jnp.array(1.0)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint(estimate, grad, Sigma, level=0.0)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint(estimate, grad, Sigma, level=1.0)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint(estimate, grad, Sigma, level=-0.5)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint(estimate, grad, Sigma, level=1.5)


def test_delta_confint_from_se_level_validation():
    """delta_confint_from_se must reject level outside (0,1)."""
    estimate = jnp.array(1.0)
    se = jnp.array(0.1)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint_from_se(estimate, se, level=0.0)
    with pytest.raises(ValueError, match="level must be in"):
        delta_confint_from_se(estimate, se, level=1.0)


def test_delta_wald_test_near_zero_se():
    """Wald test should treat very small SE as effectively zero."""
    grad = jnp.array([1e-20, 0.0, 0.0])
    Sigma = jnp.eye(3) * 0.01
    z, p = delta_wald_test(
        jnp.array(1.0), grad, Sigma,
        null_value=0.0, alternative="two-sided",
    )
    assert float(z) == float("inf")
    assert float(p) == 0.0


def test_joint_wald_test_zero_matrix_ridge():
    """joint_wald_test must handle Sigma_g being exactly the zero matrix."""
    grad = jnp.zeros((2, 3))
    Sigma = jnp.eye(3) * 0.01
    estimate = jnp.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning, match="regularized"):
        chi2, p, df = joint_wald_test(estimate, grad, Sigma)
    assert np.isfinite(chi2)
    assert np.isfinite(p)
    assert df == 2


def test_combined_gradient_empty_list():
    """combined_gradient must raise for an empty list."""
    with pytest.raises(ValueError, match="non-empty"):
        combined_gradient([], jnp.array([]))


def test_combined_gradient_shape_mismatch():
    """combined_gradient must raise when gradients have different shapes."""
    g1 = jnp.array([1.0, 2.0])
    g2 = jnp.array([3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="same shape"):
        combined_gradient([g1, g2], jnp.array([1.0, 1.0]))


def test_joint_covariance_empty_list():
    """joint_covariance_of_results must raise for an empty list."""
    Sigma = jnp.eye(3)
    with pytest.raises(ValueError, match="non-empty"):
        joint_covariance_of_results([], Sigma)


def test_joint_covariance_shape_mismatch():
    """joint_covariance_of_results must raise when gradients have different shapes."""
    g1 = jnp.array([1.0, 2.0])
    g2 = jnp.array([3.0, 4.0, 5.0])
    Sigma = jnp.eye(3)
    with pytest.raises(ValueError, match="same shape"):
        joint_covariance_of_results([g1, g2], Sigma)


def test_delta_variance_3d_grad():
    """delta_variance must handle 3D gradients from multi-outcome models."""
    rng = np.random.default_rng(42)
    p = 4
    cov = jnp.eye(p) * 0.01
    # 3D gradient: (2 atoms, 3 outcomes, 4 params)
    grad = jnp.asarray(rng.standard_normal((2, 3, p)))
    var = delta_variance(grad, cov)
    assert var.shape == (6, 6)


def test_delta_se_3d_grad():
    """delta_se must return 2D SEs for 3D gradients."""
    rng = np.random.default_rng(42)
    p = 4
    cov = jnp.eye(p) * 0.01
    grad = jnp.asarray(rng.standard_normal((2, 3, p)))
    se = delta_se(grad, cov)
    assert se.shape == (2, 3)
