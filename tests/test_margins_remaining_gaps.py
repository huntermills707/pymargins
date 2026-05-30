"""Targeted tests for remaining coverage gaps in _result/_margins.py."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pymargins._result._margins import (
    MarginsResult,
    _check_draws_match,
    _name_to_phi,
    _phi_to_name,
    compose_results,
)

# ---------------------------------------------------------------------------
# conf_int() simultaneous with draws
# ---------------------------------------------------------------------------


def test_conf_int_simultaneous_draws_only():
    """Cover conf_int simultaneous path when only draws are available."""
    draws = np.random.randn(100, 3)
    est = np.mean(draws, axis=0)
    se = np.std(draws, axis=0, ddof=1)
    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=est - 1.96 * se,
        conf_int_upper=est + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws,
    )
    lo, hi = r.conf_int(simultaneous=True)
    assert lo.shape == est.shape
    assert hi.shape == est.shape


def test_conf_int_simultaneous_draws_only_scalar():
    """Cover conf_int simultaneous with scalar draws."""
    draws = np.random.randn(100)
    est = np.array([np.mean(draws)])
    se = np.array([np.std(draws, ddof=1)])
    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=est - 1.96 * se,
        conf_int_upper=est + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws,
    )
    lo, hi = r.conf_int(simultaneous=True)
    assert lo.shape == (1,)


def test_conf_int_simultaneous_draws_inf_with_phi():
    """Cover lines 788-792: simultaneous draws_inf with phi."""
    import jax.numpy as jnp

    draws_inf = np.random.randn(100, 2)
    est_inf = np.mean(draws_inf, axis=0)
    se = np.std(draws_inf, axis=0, ddof=1)
    r = MarginsResult(
        estimate=np.exp(est_inf),
        std_error=se,
        conf_int_lower=np.exp(est_inf - 1.96 * se),
        conf_int_upper=np.exp(est_inf + 1.96 * se),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=np.exp(draws_inf),
        draws_inf=draws_inf,
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    lo, hi = r.conf_int(simultaneous=True)
    assert lo.shape == est_inf.shape


# ---------------------------------------------------------------------------
# conf_int() basic and studentized CI methods
# ---------------------------------------------------------------------------


def test_conf_int_basic_method():
    """Cover basic bootstrap CI method."""
    draws_inf = np.random.randn(100, 2)
    est_inf = np.mean(draws_inf, axis=0)
    se = np.std(draws_inf, axis=0, ddof=1)
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
        ci_method="basic",
    )
    lo, hi = r.conf_int()
    assert lo.shape == est_inf.shape


def test_conf_int_studentized_method():
    """Cover studentized bootstrap CI method."""
    draws_inf = np.random.randn(100, 2)
    est_inf = np.mean(draws_inf, axis=0)
    se = np.std(draws_inf, axis=0, ddof=1)
    t_star = np.random.randn(100, 2)
    se_hat = np.ones(2)
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
        ci_method="studentized",
        bootstrap_extras={"t_star": t_star, "se_hat": se_hat},
    )
    lo, hi = r.conf_int()
    assert lo.shape == est_inf.shape


def test_conf_int_bca_method():
    """Cover bca bootstrap CI method."""
    draws_inf = np.random.randn(100, 2)
    est_inf = np.mean(draws_inf, axis=0)
    se = np.std(draws_inf, axis=0, ddof=1)
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
        ci_method="bca",
        bootstrap_extras={"z0": 0.1, "a": 0.05},
    )
    lo, hi = r.conf_int()
    assert lo.shape == est_inf.shape


# ---------------------------------------------------------------------------
# joint_test() empirical/delta branches
# ---------------------------------------------------------------------------


def test_joint_test_empirical_kind():
    """Cover joint_test empirical path (kind='empirical')."""
    draws_inf = np.random.randn(100, 3)
    est_inf = np.mean(draws_inf, axis=0)
    se = np.std(draws_inf, axis=0, ddof=1)
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
    )
    result = r.joint_test(kind="empirical")
    assert hasattr(result, "statistic")


def test_joint_test_empirical_singular_cov():
    """Cover joint_test empirical path with singular covariance (LinAlgError)."""
    # Create draws where covariance is singular (all draws identical)
    draws_inf = np.ones((50, 2)) * 0.1
    est_inf = np.array([0.1, 0.1])
    se = np.array([0.01, 0.01])
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
    )
    result = r.joint_test(kind="empirical")
    assert hasattr(result, "statistic")


def test_joint_test_draws_1d():
    """Cover joint_test with 1D draws (lines 1067-1071)."""
    draws_inf = np.random.randn(100)
    est_inf = np.array([np.mean(draws_inf)])
    se = np.array([np.std(draws_inf, ddof=1)])
    r = MarginsResult(
        estimate=est_inf,
        std_error=se,
        conf_int_lower=est_inf - 1.96 * se,
        conf_int_upper=est_inf + 1.96 * se,
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=draws_inf,
        draws_inf=draws_inf,
    )
    result = r.joint_test(kind="empirical")
    assert hasattr(result, "statistic")


def test_joint_test_value_shape_mismatch():
    """Cover joint_test value shape mismatch error."""
    cov = np.eye(2)
    grad = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([1.0, 1.0]),
        conf_int_lower=np.array([0.0, 1.0]),
        conf_int_upper=np.array([2.0, 3.0]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=grad,
        cov_params=cov,
    )
    with pytest.raises(ValueError, match="shape"):
        r.joint_test(value=np.array([1.0]))


def test_joint_test_non_finite_value():
    """Cover joint_test non-finite value error."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([1.0]),
        conf_int_lower=np.array([0.0]),
        conf_int_upper=np.array([2.0]),
        method="delta",
        level=0.95,
        n_obs=50,
    )
    with pytest.raises(ValueError, match="finite"):
        r.joint_test(value=np.array([np.inf]))


def test_joint_test_invalid_null_scale():
    """Cover joint_test invalid null_scale error."""
    import jax.numpy as jnp

    r = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([1.0, 1.0]),
        conf_int_lower=np.array([0.0, 1.0]),
        conf_int_upper=np.array([2.0, 3.0]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
        cov_params=np.eye(2),
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    with pytest.raises(ValueError, match="null_scale"):
        r.joint_test(value=np.array([0.0, 0.0]), null_scale="invalid")


# ---------------------------------------------------------------------------
# _combine_results error paths
# ---------------------------------------------------------------------------


def test_combine_results_mixed_gradient_draws():
    """Cover _combine_results mixed gradient+draws error."""
    mock_session = MagicMock()
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        draws=np.random.randn(50),
        session=mock_session,
    )
    with pytest.raises(ValueError, match="gradient"):
        a + b


def test_combine_results_cov_params_none():
    """Cover _combine_results cov_params None error (line 1899-1903)."""
    mock_session = MagicMock()
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=None,
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([0.0, 1.0]),
        cov_params=None,
        session=mock_session,
    )
    with pytest.raises(ValueError, match="cov_params"):
        a + b


# ---------------------------------------------------------------------------
# compose_results vector-Jacobian and error paths
# ---------------------------------------------------------------------------


def test_compose_results_vector_jacobian():
    """Cover compose_results vector-Jacobian path (lines 2165-2200)."""
    import jax.numpy as jnp

    mock_session = MagicMock()
    cov = np.eye(2)
    grad = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.1]),
        conf_int_lower=np.array([0.8, 1.8]),
        conf_int_upper=np.array([1.2, 2.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=grad,
        cov_params=cov,
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([3.0, 4.0]),
        std_error=np.array([0.1, 0.1]),
        conf_int_lower=np.array([2.8, 3.8]),
        conf_int_upper=np.array([3.2, 4.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=grad,
        cov_params=cov,
        session=mock_session,
    )

    def fn(theta):
        return jnp.sum(theta, axis=0)

    result = compose_results([a, b], fn)
    assert hasattr(result, "estimate")


def test_compose_results_delta_requires_gradients():
    """Cover compose_results delta path gradient missing error."""
    mock_session = MagicMock()
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=None,
        cov_params=np.eye(2),
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
        session=mock_session,
    )

    def fn(theta):
        return theta[0]

    with pytest.raises(ValueError, match="gradients"):
        compose_results([a, b], fn)


def test_compose_results_delta_requires_cov_params():
    """Cover compose_results delta path cov_params missing error."""
    mock_session = MagicMock()
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=None,
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([0.0, 1.0]),
        cov_params=None,
        session=mock_session,
    )

    def fn(theta):
        return theta[0]

    with pytest.raises(ValueError, match="cov_params"):
        compose_results([a, b], fn)


def test_compose_results_simulation_phi_path():
    """Cover compose_results simulation path with phi (lines 2007-2010)."""
    import jax.numpy as jnp

    mock_session = MagicMock()
    draws_inf = np.random.randn(50, 2)
    a = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.1]),
        conf_int_lower=np.array([0.8, 1.8]),
        conf_int_upper=np.array([1.2, 2.2]),
        method="simulation",
        level=0.95,
        n_obs=50,
        draws=jnp.exp(draws_inf),
        draws_inf=draws_inf,
        phi=jnp.exp,
        phi_inv=jnp.log,
        session=mock_session,
    )
    b = MarginsResult(
        estimate=np.array([3.0, 4.0]),
        std_error=np.array([0.1, 0.1]),
        conf_int_lower=np.array([2.8, 3.8]),
        conf_int_upper=np.array([3.2, 4.2]),
        method="simulation",
        level=0.95,
        n_obs=50,
        draws=jnp.exp(draws_inf + 0.5),
        draws_inf=draws_inf + 0.5,
        phi=jnp.exp,
        phi_inv=jnp.log,
        session=mock_session,
    )

    def fn(theta):
        return theta[0] + theta[1]

    result = compose_results([a, b], fn)
    assert hasattr(result, "estimate")


# ---------------------------------------------------------------------------
# _phi_to_name and _name_to_phi exception handlers
# ---------------------------------------------------------------------------


def test_phi_to_name_unknown():
    """Cover _phi_to_name returning None for unknown function."""
    assert _phi_to_name(lambda x: x) is None


def test_phi_to_name_none():
    """Cover _phi_to_name with None input."""
    assert _phi_to_name(None) is None


def test_name_to_phi_unknown_name():
    """Cover _name_to_phi warning for unknown name."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _name_to_phi("unknown.module.function")
        assert result is None
        assert len(w) == 1
        assert "Unknown phi name" in str(w[0].message)


def test_name_to_phi_import_failure():
    """Cover _name_to_phi exception handler (lines 1786-1792)."""
    with patch.dict(
        "pymargins._result._margins._KNOWN_PHI_MAP",
        {"bad.import": ("nonexistent_module_12345", "foo")},
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _name_to_phi("bad.import")
            assert result is None
            assert len(w) == 1
            assert "Could not reconstruct" in str(w[0].message)


# ---------------------------------------------------------------------------
# __mul__ with negative scalar
# ---------------------------------------------------------------------------


def test_mul_negative_scalar():
    """Cover __mul__ with negative scalar (lines 1185-1187)."""
    r = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.5]),
        conf_int_lower=np.array([1.0]),
        conf_int_upper=np.array([3.0]),
        method="delta",
        level=0.95,
        n_obs=50,
    )
    result = r * (-2.0)
    assert float(result.estimate[0]) == pytest.approx(-4.0)
    # CI should be swapped for negative scalar
    assert float(result.conf_int_lower[0]) == pytest.approx(-6.0)
    assert float(result.conf_int_upper[0]) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# _check_draws_match error paths
# ---------------------------------------------------------------------------


def test_check_draws_match_bootstrap_no_bank_id():
    """Cover _check_draws_match bootstrap without bank_id."""
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        resample_bank_id=None,
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        resample_bank_id="bank1",
    )
    with pytest.raises(ValueError, match="resample_bank_id"):
        _check_draws_match(a, b)


def test_check_draws_match_different_bank_id():
    """Cover _check_draws_match different bank_id error."""
    a = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        resample_bank_id="bank1",
    )
    b = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
        resample_bank_id="bank2",
    )
    with pytest.raises(ValueError, match="resample bank"):
        _check_draws_match(a, b)


# ---------------------------------------------------------------------------
# outcome() legacy slicing
# ---------------------------------------------------------------------------


def test_outcome_legacy_slicing():
    """Cover outcome() legacy label-heuristic slicing."""
    n_outcomes = 3
    n_atoms = 2
    n_components = n_atoms * n_outcomes
    est = np.arange(n_components, dtype=float)
    # Labels must be in format "label (suffix)" for heuristic grouping
    labels = [f"x[{i}] ({j})" for i in range(n_atoms) for j in range(n_outcomes)]
    r = MarginsResult(
        estimate=est,
        std_error=est * 0.1,
        conf_int_lower=est - 0.5,
        conf_int_upper=est + 0.5,
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": labels},
    )
    sliced = r.outcome(1)
    assert len(sliced.estimate) == n_atoms


def test_outcome_invalid_label():
    """Cover outcome() invalid label error."""
    n_outcomes = 2
    n_atoms = 2
    n_components = n_atoms * n_outcomes
    est = np.arange(n_components, dtype=float)
    labels = [f"x[{i}] ({j})" for i in range(n_atoms) for j in range(n_outcomes)]
    r = MarginsResult(
        estimate=est,
        std_error=est * 0.1,
        conf_int_lower=est - 0.5,
        conf_int_upper=est + 0.5,
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": labels},
    )
    with pytest.raises(ValueError, match="not found"):
        r.outcome("nonexistent")


def test_outcome_index_out_of_range():
    """Cover outcome() index out of range error."""
    n_outcomes = 2
    n_atoms = 2
    n_components = n_atoms * n_outcomes
    est = np.arange(n_components, dtype=float)
    labels = [f"x[{i}] ({j})" for i in range(n_atoms) for j in range(n_outcomes)]
    r = MarginsResult(
        estimate=est,
        std_error=est * 0.1,
        conf_int_lower=est - 0.5,
        conf_int_upper=est + 0.5,
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": labels},
    )
    with pytest.raises(ValueError, match="out of range"):
        r.outcome(5)


# ---------------------------------------------------------------------------
# test() with invalid inputs
# ---------------------------------------------------------------------------


def test_test_non_finite_value():
    """Cover test() non-finite value error."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
    )
    with pytest.raises(ValueError, match="finite"):
        r.test(value=np.nan)


def test_test_invalid_null_scale():
    """Cover test() invalid null_scale error."""
    import jax.numpy as jnp

    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    with pytest.raises(ValueError, match="null_scale"):
        r.test(null_scale="bad")


# ---------------------------------------------------------------------------
# summary / to_latex / to_html with various options
# ---------------------------------------------------------------------------


def test_summary_with_stars():
    """Cover summary with star notation."""
    r = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.5]),
        conf_int_lower=np.array([0.8, 1.0]),
        conf_int_upper=np.array([1.2, 3.0]),
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": ["a", "b"]},
    )
    s = r.summary(stars=True)
    assert isinstance(s, str)


def test_to_latex():
    """Cover to_latex output."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": ["a"]},
    )
    latex = r.to_latex()
    assert "\\begin{tabular}" in latex


def test_to_html():
    """Cover to_html output."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        estimand_metadata={"labels": ["a"]},
    )
    html = r.to_html()
    assert "<table" in html


# ---------------------------------------------------------------------------
# pairwise_contrasts with bad inputs
# ---------------------------------------------------------------------------


def test_pairwise_contrasts_no_gradient():
    """Cover pairwise_contrasts error when no gradient."""
    r = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.1]),
        conf_int_lower=np.array([0.8, 1.8]),
        conf_int_upper=np.array([1.2, 2.2]),
        method="bootstrap",
        level=0.95,
        n_obs=50,
    )
    with pytest.raises(ValueError, match="delta-method"):
        r.pairwise_contrasts()


def test_pairwise_contrasts_too_few():
    """Cover pairwise_contrasts error with < 2 components."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
    )
    with pytest.raises(ValueError, match="at least 2"):
        r.pairwise_contrasts()


# ---------------------------------------------------------------------------
# Materialize and influence edge cases
# ---------------------------------------------------------------------------


def test_materialize_no_gradient_no_draws():
    """Cover materialize when no gradient or draws."""
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
    )
    # materialize always returns a new object with machinery stripped
    m = r.materialize()
    assert m.gradient is None
    assert m.draws is None
    assert m.session is None


def test_influence_no_score_obs():
    """Cover influence when session has no score_obs."""
    mock_session = MagicMock()
    mock_session.adapter.score_obs.side_effect = NotImplementedError
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        n_obs=50,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
        session=mock_session,
    )
    with pytest.raises(NotImplementedError):
        r.influence()
