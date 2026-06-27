"""Tests for pure interval/test functions in ``pymargins._result._intervals``.

Design §4.3, §4.7, req §6. Added in 0.4.0 (R4).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pymargins._result._intervals import (
    bonferroni_level,
    delta_interval,
    draws_interval,
    draws_test,
    joint_wald,
    sidak_level,
    supt_interval_delta,
    supt_interval_draws,
    wald_interval,
    wald_test,
)

# ---------------------------------------------------------------------------
# Level allocation
# ---------------------------------------------------------------------------


def test_bonferroni_level():
    level = 0.95
    k = 4
    expected = 1.0 - (1.0 - level) / k
    assert bonferroni_level(level, k) == pytest.approx(expected)


def test_sidak_level():
    level = 0.95
    k = 4
    expected = level ** (1.0 / k)
    assert sidak_level(level, k) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Wald / delta intervals
# ---------------------------------------------------------------------------


def test_wald_interval_matches_z_quantile():
    estimate = np.array([1.0, 2.0])
    se = np.array([0.5, 1.0])
    level = 0.95
    z = stats.norm.ppf(0.5 + level / 2.0)
    lo_expected = estimate - z * se
    hi_expected = estimate + z * se
    lo, hi = wald_interval(estimate, se, level)
    np.testing.assert_allclose(lo, lo_expected)
    np.testing.assert_allclose(hi, hi_expected)


def test_wald_interval_with_phi():
    estimate = np.array(0.0)
    se = np.array(1.0)
    level = 0.95
    lo, hi = wald_interval(estimate, se, level, phi=jnp.exp)
    assert float(lo) > 0.0
    np.testing.assert_allclose(lo, np.exp(-stats.norm.ppf(0.975)))


def test_delta_interval_matches_manual():
    estimate = np.array(1.0)
    gradient = np.array([2.0, -1.0])
    cov_params = np.array([[1.0, 0.5], [0.5, 1.0]])
    level = 0.95
    var = float(gradient @ cov_params @ gradient)
    se = np.sqrt(var)
    z = stats.norm.ppf(0.975)
    lo, hi = delta_interval(estimate, gradient, cov_params, level)
    assert float(lo) == pytest.approx(1.0 - z * se)
    assert float(hi) == pytest.approx(1.0 + z * se)


# ---------------------------------------------------------------------------
# Draws intervals
# ---------------------------------------------------------------------------


def test_draws_interval_percentile():
    rng = np.random.default_rng(0)
    draws = rng.normal(loc=1.0, scale=1.0, size=20000)
    level = 0.95
    lo, hi = draws_interval(draws, level, ci_method="percentile")
    assert float(lo) == pytest.approx(-0.96, abs=0.05)
    assert float(hi) == pytest.approx(2.96, abs=0.05)


def test_draws_interval_basic():
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=0.0, scale=1.0, size=20000)
    level = 0.95
    lo, hi = draws_interval(draws, level, ci_method="basic")
    assert float(lo) == pytest.approx(-1.96, abs=0.05)
    assert float(hi) == pytest.approx(1.96, abs=0.05)


def test_draws_interval_bca_missing_z0_falls_back_to_percentile():
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=0.0, scale=1.0, size=20000)
    level = 0.95
    lo, hi = draws_interval(draws, level, ci_method="bca", bootstrap_extras={})
    assert float(lo) == pytest.approx(-1.96, abs=0.05)
    assert float(hi) == pytest.approx(1.96, abs=0.05)


def test_draws_interval_studentized_missing_extras_falls_back_to_percentile():
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=0.0, scale=1.0, size=20000)
    level = 0.95
    lo, hi = draws_interval(draws, level, ci_method="studentized", bootstrap_extras={})
    assert float(lo) == pytest.approx(-1.96, abs=0.05)
    assert float(hi) == pytest.approx(1.96, abs=0.05)


# ---------------------------------------------------------------------------
# Sup-t intervals
# ---------------------------------------------------------------------------


def test_supt_interval_delta_scalar_equals_wald():
    """For a scalar estimand sup-t collapses to the pointwise z interval."""
    estimate = np.array([1.0])
    gradient = np.array([1.0, 0.0])
    cov_params = np.eye(2)
    level = 0.95
    lo_supt, hi_supt = supt_interval_delta(estimate, gradient, cov_params, level)
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(lo_supt, 1.0 - z)
    np.testing.assert_allclose(hi_supt, 1.0 + z)


def test_supt_interval_draws_wider_than_pointwise():
    rng = np.random.default_rng(2)
    # Two independent components; sup-t band wider than component-wise.
    draws = rng.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=10000)
    estimate = np.array([0.0, 0.0])
    se = np.array([1.0, 1.0])
    level = 0.95
    lo, hi = supt_interval_draws(draws, estimate, se, level)
    width = hi - lo
    assert width[0] > 3.5  # wider than ~3.92 for two components
    assert width[1] > 3.5


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def test_wald_test_matches_scipy():
    estimate = np.array(2.0)
    gradient = np.array([1.0, 0.0])
    cov_params = np.eye(2)
    statistic, pvalue = wald_test(
        estimate, gradient, cov_params, null_value=0.0, alternative="two-sided"
    )
    expected_z = 2.0
    expected_p = 2.0 * stats.norm.sf(abs(expected_z))
    assert float(statistic) == pytest.approx(expected_z)
    assert float(pvalue) == pytest.approx(expected_p, abs=1e-10)


def test_draws_test_empirical_pvalue():
    rng = np.random.default_rng(3)
    estimate = np.array(3.0)
    draws = rng.normal(loc=3.0, scale=1.0, size=10000)
    statistic, pvalue = draws_test(
        estimate, draws, null_value=0.0, alternative="two-sided"
    )
    assert float(statistic) == pytest.approx(3.0)
    assert float(pvalue) < 0.01


def test_joint_wald_chi2():
    estimate = np.array([1.0, 0.5])
    gradient = np.eye(2)
    cov_params = np.eye(2)
    chi2, p, df = joint_wald(estimate, gradient, cov_params, null_value=np.zeros(2))
    expected_chi2 = float(estimate @ estimate)
    expected_p = 1.0 - stats.chi2.cdf(expected_chi2, 2)
    assert chi2 == pytest.approx(expected_chi2)
    assert p == pytest.approx(expected_p, abs=1e-10)
    assert df == 2
