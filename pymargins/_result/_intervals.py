"""Pure interval and test functions for result objects.

Lifted from ``pymargins._result._margins.MarginsResult`` (lines 671–900 for
intervals, 903–1163 for tests) and ``pymargins._inference._dispatch.run_test``.
The legacy class keeps its own copies until R7; this module provides the
reviewed, doctrine-shaped functions used by ``GraphResult``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from scipy import stats

from .._delta import (
    delta_confint_from_se,
    delta_se,
    delta_wald_test,
    joint_covariance_of_results,
    joint_wald_test,
)

# ---------------------------------------------------------------------------
# Familywise level allocation
# ---------------------------------------------------------------------------


def bonferroni_level(level: float, k: int) -> float:
    """Per-component confidence level under a Bonferroni allocation."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return 1.0 - (1.0 - level) / k


def sidak_level(level: float, k: int) -> float:
    """Per-component confidence level under a Šidák allocation."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return level ** (1.0 / k)


# ---------------------------------------------------------------------------
# Wald / delta intervals
# ---------------------------------------------------------------------------


def wald_interval(
    estimate_inf: np.ndarray,
    se_inf: np.ndarray,
    level: float,
    phi: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise Wald interval from a precomputed SE."""
    lower_inf, upper_inf = delta_confint_from_se(
        jnp.asarray(estimate_inf),
        jnp.asarray(se_inf),
        level=level,
        phi=None,
    )
    if phi is not None:
        return np.asarray(phi(lower_inf)), np.asarray(phi(upper_inf))
    return np.asarray(lower_inf), np.asarray(upper_inf)


def delta_interval(
    estimate_inf: np.ndarray,
    gradient: np.ndarray,
    cov_params: np.ndarray,
    level: float,
    phi: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-method confidence interval from gradient and Σ̂."""
    se = delta_se(jnp.asarray(gradient), jnp.asarray(cov_params))
    return wald_interval(estimate_inf, se, level, phi=phi)


# ---------------------------------------------------------------------------
# Simulation / bootstrap intervals
# ---------------------------------------------------------------------------


def draws_interval(
    draws_inf: np.ndarray,
    level: float,
    phi: Callable | None = None,
    ci_method: str = "percentile",
    bootstrap_extras: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interval from simulation/bootstrap draws on the inference scale.

    Supports ``percentile``, ``basic``, ``bca`` (when ``bootstrap_extras``
    carries ``z0`` and ``a``), and ``studentized`` (when ``bootstrap_extras``
    carries ``t_star`` and ``se_hat``).
    """
    draws = np.asarray(draws_inf)
    est_inf = np.mean(draws, axis=0)
    alpha = (1.0 - level) / 2.0

    if ci_method == "basic":
        lower_inf = 2.0 * est_inf - np.quantile(draws, 1.0 - alpha, axis=0)
        upper_inf = 2.0 * est_inf - np.quantile(draws, alpha, axis=0)
    elif ci_method == "bca" and bootstrap_extras is not None:
        z0 = bootstrap_extras.get("z0")
        a = bootstrap_extras.get("a")
        if z0 is not None:
            from .._inference._bootstrap import _bca_confint

            lower, upper = _bca_confint(
                draws_inf, np.atleast_1d(est_inf), level, z0, a, phi
            )
            return np.asarray(lower), np.asarray(upper)
    elif ci_method == "studentized" and bootstrap_extras is not None:
        t_stats = bootstrap_extras.get("t_star")
        se_hat = bootstrap_extras.get("se_hat")
        if t_stats is not None and se_hat is not None:
            t_lower = np.quantile(t_stats, alpha, axis=0)
            t_upper = np.quantile(t_stats, 1.0 - alpha, axis=0)
            lower_inf = est_inf - t_upper * se_hat
            upper_inf = est_inf - t_lower * se_hat
        else:
            ci_method = "percentile"
    else:
        if ci_method not in ("percentile", "bca", "studentized", "basic"):
            raise ValueError(f"Unsupported ci_method: {ci_method!r}")
        lower_inf = np.quantile(draws, alpha, axis=0)
        upper_inf = np.quantile(draws, 1.0 - alpha, axis=0)

    if phi is not None:
        return np.asarray(phi(lower_inf)), np.asarray(phi(upper_inf))
    return np.asarray(lower_inf), np.asarray(upper_inf)


# ---------------------------------------------------------------------------
# Simultaneous (sup-t) intervals
# ---------------------------------------------------------------------------


def supt_interval_draws(
    draws_inf: np.ndarray,
    estimate_inf: np.ndarray,
    se_inf: np.ndarray,
    level: float,
    phi: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sup-t band from simulation/bootstrap draws (Montiel Olea–Plagborg-Møller)."""
    draws = np.asarray(draws_inf)
    est_arr = np.atleast_1d(np.asarray(estimate_inf))
    se = np.atleast_1d(np.asarray(se_inf))

    if se.ndim == 0:
        se = np.array([se])
    if draws.ndim == 1:
        draws = draws[:, None]
    if est_arr.ndim == 0:
        est_arr = np.array([est_arr])

    est_bc = est_arr[None, :] if est_arr.ndim == 1 else est_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        std_dev = np.abs(draws - est_bc) / se[None, :]
    max_dev = np.nanmax(std_dev, axis=1)
    crit = float(np.quantile(max_dev, level))
    lower_inf = est_arr - crit * se
    upper_inf = est_arr + crit * se

    if phi is not None:
        return np.asarray(phi(lower_inf)), np.asarray(phi(upper_inf))
    return np.asarray(lower_inf), np.asarray(upper_inf)


def supt_interval_delta(
    estimate_inf: np.ndarray,
    gradient: np.ndarray,
    cov_params: np.ndarray,
    level: float,
    phi: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sup-t band from the delta-method correlation structure."""
    grad = jnp.asarray(gradient)
    if grad.ndim == 1:
        grad = grad[None, :]

    cov_joint = joint_covariance_of_results(
        [grad[i] for i in range(grad.shape[0])],
        jnp.asarray(cov_params),
    )
    se_vec = jnp.sqrt(jnp.diag(cov_joint))
    n_comp = int(se_vec.shape[0])

    if n_comp == 1:
        crit = float(stats.norm.ppf(0.5 + level / 2.0))
    else:
        cov_np = np.asarray(cov_joint)
        cov_np = (cov_np + cov_np.T) / 2.0
        eigvals = np.linalg.eigvalsh(cov_np)
        min_eig = float(np.min(eigvals))
        ridge = max(0.0, -min_eig + 1e-6)
        if ridge > 0:
            cov_np = cov_np + np.eye(n_comp) * ridge
        R = cov_np / np.outer(se_vec, se_vec)
        R = (R + R.T) / 2.0
        R = np.clip(R, -1.0, 1.0)
        np.fill_diagonal(R, 1.0)
        rng = np.random.default_rng(42)
        n_mc = 10000
        z_draws = rng.multivariate_normal(mean=np.zeros(n_comp), cov=R, size=n_mc)
        max_abs = np.max(np.abs(z_draws), axis=1)
        crit = float(np.quantile(max_abs, level))

    est_arr = np.atleast_1d(np.asarray(estimate_inf))
    lower_inf = est_arr - crit * se_vec
    upper_inf = est_arr + crit * se_vec

    if phi is not None:
        return np.asarray(phi(lower_inf)), np.asarray(phi(upper_inf))
    return np.asarray(lower_inf), np.asarray(upper_inf)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def wald_test(
    estimate_inf: np.ndarray,
    gradient: np.ndarray,
    cov_params: np.ndarray,
    *,
    null_value: float | np.ndarray = 0.0,
    alternative: str = "two-sided",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-component Wald test on the inference scale."""
    return delta_wald_test(
        jnp.asarray(estimate_inf),
        jnp.asarray(gradient),
        jnp.asarray(cov_params),
        null_value=null_value,
        alternative=alternative,
    )


def draws_test(
    estimate: np.ndarray,
    draws: np.ndarray,
    *,
    null_value: float | np.ndarray = 0.0,
    alternative: str = "two-sided",
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical p-value from simulation/bootstrap draws.

    Returns the observed estimate minus null as the effect-size statistic.
    """
    estimate = np.asarray(estimate)
    null_value = np.asarray(null_value)
    draws = np.asarray(draws)

    if alternative == "two-sided":
        p_left = np.mean(draws <= null_value, axis=0)
        p_right = np.mean(draws >= null_value, axis=0)
        p = np.clip(2.0 * np.minimum(p_left, p_right), min=None, max=1.0)
    elif alternative == "greater":
        p = np.mean(draws <= null_value, axis=0)
    elif alternative == "less":
        p = np.mean(draws >= null_value, axis=0)
    else:
        raise ValueError(f"Unknown alternative: {alternative!r}")

    statistic = estimate - null_value
    return np.asarray(statistic), np.asarray(p)


def joint_wald(
    estimates_inf: np.ndarray,
    gradient: np.ndarray,
    cov_params: np.ndarray,
    *,
    null_value: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Joint Wald test H₀: g(β̂) = null_value."""
    chi2, p, df = joint_wald_test(
        jnp.asarray(estimates_inf),
        jnp.asarray(gradient),
        jnp.asarray(cov_params),
        null_value=null_value,
    )
    return float(chi2), float(p), int(df)
