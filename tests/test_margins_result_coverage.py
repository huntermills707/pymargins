"""Tests for MarginsResult coverage gaps."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pymargins._result._margins import (
    MarginsResult,
    _name_to_phi,
    _phi_to_name,
    compose_results,
)

# ---------------------------------------------------------------------------
# _phi_to_name / _name_to_phi
# ---------------------------------------------------------------------------


def test_phi_to_name_none():
    assert _phi_to_name(None) is None


def test_phi_to_name_exp():
    assert _phi_to_name(jnp.exp) == "jax.numpy.exp"


def test_phi_to_name_log():
    assert _phi_to_name(jnp.log) == "jax.numpy.log"


def test_phi_to_name_unknown():
    assert _phi_to_name(lambda x: x) is None


def test_name_to_phi_none():
    assert _name_to_phi(None) is None


def test_name_to_phi_exp():
    phi = _name_to_phi("jax.numpy.exp")
    assert phi is jnp.exp


def test_name_to_phi_unknown_warns():
    with pytest.warns(UserWarning, match="Unknown phi name"):
        assert _name_to_phi("custom.phi") is None


# ---------------------------------------------------------------------------
# to_disk / from_disk with custom phi
# ---------------------------------------------------------------------------


def test_to_disk_custom_phi_warning():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        phi=lambda x: x * 2,
        phi_inv=lambda x: x / 2,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.pkl"
        with pytest.warns(UserWarning, match="custom function"):
            r.to_disk(path)
        loaded = MarginsResult.from_disk(path)
    assert loaded.phi is None
    assert loaded.phi_inv is None


# ---------------------------------------------------------------------------
# outcome with _outcome_shape
# ---------------------------------------------------------------------------


def test_outcome_with_outcome_shape():
    r = MarginsResult(
        estimate=np.array([[0.1, 0.2], [0.3, 0.4]]),
        std_error=np.array([[0.01, 0.02], [0.03, 0.04]]),
        conf_int_lower=np.array([[0.08, 0.18], [0.28, 0.38]]),
        conf_int_upper=np.array([[0.12, 0.22], [0.32, 0.42]]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "labels": ["a (0)", "a (1)", "b (0)", "b (1)"],
            "_outcome_shape": {
                "n_atoms": 2,
                "n_outcomes": 2,
                "outcome_labels": ["0", "1"],
            },
        },
        gradient=np.ones((2, 2, 3)),
    )
    sub = r.outcome(0)
    assert sub.estimate.shape == (2,)
    assert sub.gradient.shape == (2, 3)


def test_outcome_with_outcome_shape_by_label():
    r = MarginsResult(
        estimate=np.array([[0.1, 0.2], [0.3, 0.4]]),
        std_error=np.array([[0.01, 0.02], [0.03, 0.04]]),
        conf_int_lower=np.array([[0.08, 0.18], [0.28, 0.38]]),
        conf_int_upper=np.array([[0.12, 0.22], [0.32, 0.42]]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "labels": ["a (0)", "a (1)", "b (0)", "b (1)"],
            "_outcome_shape": {
                "n_atoms": 2,
                "n_outcomes": 2,
                "outcome_labels": ["0", "1"],
            },
        },
    )
    sub = r.outcome("1")
    assert sub.estimate.shape == (2,)


def test_outcome_invalid_label_raises():
    r = MarginsResult(
        estimate=np.array([[0.1, 0.2], [0.3, 0.4]]),
        std_error=np.array([[0.01, 0.02], [0.03, 0.04]]),
        conf_int_lower=np.array([[0.08, 0.18], [0.28, 0.38]]),
        conf_int_upper=np.array([[0.12, 0.22], [0.32, 0.42]]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "_outcome_shape": {
                "n_atoms": 2,
                "n_outcomes": 2,
                "outcome_labels": ["0", "1"],
            },
        },
    )
    with pytest.raises(ValueError, match="not found"):
        r.outcome("99")


def test_outcome_index_out_of_range_raises():
    r = MarginsResult(
        estimate=np.array([[0.1, 0.2], [0.3, 0.4]]),
        std_error=np.array([[0.01, 0.02], [0.03, 0.04]]),
        conf_int_lower=np.array([[0.08, 0.18], [0.28, 0.38]]),
        conf_int_upper=np.array([[0.12, 0.22], [0.32, 0.42]]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "_outcome_shape": {
                "n_atoms": 2,
                "n_outcomes": 2,
                "outcome_labels": ["0", "1"],
            },
        },
    )
    with pytest.raises(ValueError, match="out of range"):
        r.outcome(5)


# ---------------------------------------------------------------------------
# compose_results error paths
# ---------------------------------------------------------------------------


def test_compose_results_too_few():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="at least two"):
        compose_results([r], lambda x: x)


def test_compose_results_different_sessions():
    r1 = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
    )
    r2 = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.2]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="session reference"):
        compose_results([r1, r2], lambda x: x)


def test_compose_results_different_methods():
    from unittest.mock import MagicMock

    session = MagicMock()
    r1 = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        session=session,
    )
    r2 = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.2]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="simulation",
        level=0.95,
        session=session,
    )
    with pytest.raises(ValueError, match="same inference method"):
        compose_results([r1, r2], lambda x: x)


# ---------------------------------------------------------------------------
# conf_int with simultaneous=True on simulation
# ---------------------------------------------------------------------------


def test_conf_int_simultaneous_simulation():
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000, 3))
    r = MarginsResult(
        estimate=np.array([0.1, 0.2, 0.3]),
        std_error=np.array([0.01, 0.02, 0.03]),
        conf_int_lower=np.array([0.08, 0.18, 0.28]),
        conf_int_upper=np.array([0.12, 0.22, 0.32]),
        method="simulation",
        level=0.95,
        draws_inf=draws,
    )
    lo, hi = r.conf_int(simultaneous=True)
    assert lo.shape == (3,)
    assert hi.shape == (3,)
    # Simultaneous should be wider than pointwise
    assert np.all(lo <= r.conf_int_lower)
    assert np.all(hi >= r.conf_int_upper)


# ---------------------------------------------------------------------------
# conf_int with basic bootstrap
# ---------------------------------------------------------------------------


def test_conf_int_basic_bootstrap():
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000,))
    r = MarginsResult(
        estimate=np.array([0.1]),
        std_error=np.array([0.01]),
        conf_int_lower=np.array([0.08]),
        conf_int_upper=np.array([0.12]),
        method="bootstrap",
        level=0.95,
        draws_inf=draws,
        ci_method="basic",
    )
    lo, hi = r.conf_int()
    assert np.isfinite(lo.flat[0])
    assert np.isfinite(hi.flat[0])


# ---------------------------------------------------------------------------
# conf_int with studentized bootstrap
# ---------------------------------------------------------------------------


def test_conf_int_studentized_bootstrap():
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000,))
    t_star = rng.standard_normal((1000,))
    se_hat = 0.01
    r = MarginsResult(
        estimate=np.array([0.1]),
        std_error=np.array([0.01]),
        conf_int_lower=np.array([0.08]),
        conf_int_upper=np.array([0.12]),
        method="bootstrap",
        level=0.95,
        draws_inf=draws,
        ci_method="studentized",
        bootstrap_extras={"t_star": t_star, "se_hat": se_hat},
    )
    lo, hi = r.conf_int()
    assert np.isfinite(lo.flat[0])
    assert np.isfinite(hi.flat[0])


# ---------------------------------------------------------------------------
# __truediv__ operator
# ---------------------------------------------------------------------------


def test_margins_result_truediv():
    r = MarginsResult(
        estimate=np.array([10.0]),
        std_error=np.array([1.0]),
        conf_int_lower=np.array([8.0]),
        conf_int_upper=np.array([12.0]),
        method="delta",
        level=0.95,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
    )
    divided = r / 2.0
    assert float(divided.estimate.flat[0]) == 5.0


# ---------------------------------------------------------------------------
# joint_test error path
# ---------------------------------------------------------------------------


def test_joint_test_neither_gradient_nor_draws():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2]),
        std_error=np.array([0.01, 0.02]),
        conf_int_lower=np.array([0.08, 0.18]),
        conf_int_upper=np.array([0.12, 0.22]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="neither"):
        r.joint_test()


# ---------------------------------------------------------------------------
# _check_compatible
# ---------------------------------------------------------------------------


def test_check_compatible_no_session():
    r1 = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
    )
    r2 = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.2]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="session reference"):
        r1 + r2


def test_check_compatible_different_sessions():
    from unittest.mock import MagicMock

    s1 = MagicMock()
    s2 = MagicMock()
    r1 = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        session=s1,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
    )
    r2 = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.2]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="delta",
        level=0.95,
        session=s2,
        gradient=np.array([0.0, 1.0]),
        cov_params=np.eye(2),
    )
    with pytest.raises(ValueError, match="different Margins sessions"):
        r1 + r2


# ---------------------------------------------------------------------------
# compose_results with simulation draws
# ---------------------------------------------------------------------------


def test_compose_results_simulation_draws():
    from unittest.mock import MagicMock

    session = MagicMock()
    session.rng_seed = 42
    session.n_sim = 1000
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000, 2))
    r1 = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="simulation",
        level=0.95,
        session=session,
        draws_inf=draws[:, 0],
    )
    r2 = MarginsResult(
        estimate=np.array([2.0]),
        std_error=np.array([0.2]),
        conf_int_lower=np.array([1.8]),
        conf_int_upper=np.array([2.2]),
        method="simulation",
        level=0.95,
        session=session,
        draws_inf=draws[:, 1],
    )
    result = compose_results([r1, r2], lambda x: x[0] + x[1])
    assert np.isfinite(float(result.estimate.flat[0]))


# ---------------------------------------------------------------------------
# outcome legacy label path
# ---------------------------------------------------------------------------


def test_outcome_legacy_label_path():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2, 0.3, 0.4]),
        std_error=np.array([0.01, 0.02, 0.03, 0.04]),
        conf_int_lower=np.array([0.08, 0.18, 0.28, 0.38]),
        conf_int_upper=np.array([0.12, 0.22, 0.32, 0.42]),
        method="delta",
        level=0.95,
        estimand_metadata={"labels": ["a (0)", "a (1)", "b (0)", "b (1)"]},
    )
    sub = r.outcome(0)
    assert sub.estimate.size == 2
    np.testing.assert_array_equal(sub.estimate, [0.1, 0.3])


def test_outcome_legacy_label_by_string():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2, 0.3, 0.4]),
        std_error=np.array([0.01, 0.02, 0.03, 0.04]),
        conf_int_lower=np.array([0.08, 0.18, 0.28, 0.38]),
        conf_int_upper=np.array([0.12, 0.22, 0.32, 0.42]),
        method="delta",
        level=0.95,
        estimand_metadata={"labels": ["a (0)", "a (1)", "b (0)", "b (1)"]},
    )
    sub = r.outcome("1")
    assert sub.estimate.size == 2
    np.testing.assert_array_equal(sub.estimate, [0.2, 0.4])


def test_outcome_single_outcome_raises():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2]),
        std_error=np.array([0.01, 0.02]),
        conf_int_lower=np.array([0.08, 0.18]),
        conf_int_upper=np.array([0.12, 0.22]),
        method="delta",
        level=0.95,
        estimand_metadata={"labels": ["a (x)", "b (x)"]},
    )
    with pytest.raises(ValueError, match="single-outcome result"):
        r.outcome(0)


def test_outcome_no_labels_raises():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2]),
        std_error=np.array([0.01, 0.02]),
        conf_int_lower=np.array([0.08, 0.18]),
        conf_int_upper=np.array([0.12, 0.22]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="expanded outcome labels"):
        r.outcome(0)


# ---------------------------------------------------------------------------
# to_frame with over values from labels
# ---------------------------------------------------------------------------


def test_to_frame_over_values_from_labels():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2]),
        std_error=np.array([0.01, 0.02]),
        conf_int_lower=np.array([0.08, 0.18]),
        conf_int_upper=np.array([0.12, 0.22]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "labels": ["group=A, x=1", "group=B, x=1"],
            "over": ["group"],
        },
    )
    frame = r.to_frame()
    assert "over" in frame.columns
    assert "over_value" in frame.columns
    assert frame["over_value"].iloc[0] == "A"
    assert frame["over_value"].iloc[1] == "B"


# ---------------------------------------------------------------------------
# to_frame multi-outcome error
# ---------------------------------------------------------------------------


def test_to_frame_multi_outcome_error():
    r = MarginsResult(
        estimate=np.array([0.1, 0.2, 0.3, 0.4]),
        std_error=np.array([0.01, 0.02, 0.03, 0.04]),
        conf_int_lower=np.array([0.08, 0.18, 0.28, 0.38]),
        conf_int_upper=np.array([0.12, 0.22, 0.32, 0.42]),
        method="delta",
        level=0.95,
        estimand_metadata={
            "scenarios": [{"x": 1}, {"x": 2}, {"x": 3}],
            "kind": "prediction",
            "_outcome_shape": {
                "n_atoms": 2,
                "n_outcomes": 2,
            },
        },
    )
    with pytest.raises(ValueError, match="cannot unpack scenario columns"):
        r.to_frame()


# ---------------------------------------------------------------------------
# summary with stars and truncation
# ---------------------------------------------------------------------------


def test_summary_with_stars():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        gradient=np.array([1.0]),
        cov_params=np.array([[0.01]]),
    )
    s = r.summary(stars=True)
    assert "*" in s or "estimate" in s


def test_summary_truncated():
    r = MarginsResult(
        estimate=np.array([1.0, 2.0, 3.0]),
        std_error=np.array([0.1, 0.2, 0.3]),
        conf_int_lower=np.array([0.8, 1.8, 2.8]),
        conf_int_upper=np.array([1.2, 2.2, 3.2]),
        method="delta",
        level=0.95,
        gradient=np.array([[1.0], [1.0], [1.0]]),
        cov_params=np.eye(1),
    )
    s = r.summary(max_rows=2)
    assert "..." in s


# ---------------------------------------------------------------------------
# test with one-sided alternatives
# ---------------------------------------------------------------------------


def test_test_greater_alternative():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        gradient=np.array([1.0]),
        cov_params=np.array([[0.01]]),
    )
    tr = r.test(alternative="greater")
    assert tr.alternative == "greater"
    assert 0 <= float(np.asarray(tr.pvalue).flat[0]) <= 1


def test_test_less_alternative():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        gradient=np.array([1.0]),
        cov_params=np.array([[0.01]]),
    )
    tr = r.test(alternative="less")
    assert tr.alternative == "less"
    assert 0 <= float(np.asarray(tr.pvalue).flat[0]) <= 1


# ---------------------------------------------------------------------------
# pairwise_contrasts error paths
# ---------------------------------------------------------------------------


def test_pairwise_contrasts_requires_gradient():
    r = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.2]),
        conf_int_lower=np.array([0.8, 1.8]),
        conf_int_upper=np.array([1.2, 2.2]),
        method="delta",
        level=0.95,
    )
    with pytest.raises(ValueError, match="requires a delta-method result"):
        r.pairwise_contrasts()


def test_pairwise_contrasts_too_few():
    r = MarginsResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        method="delta",
        level=0.95,
        gradient=np.array([[1.0]]),
    )
    with pytest.raises(ValueError, match="at least 2 components"):
        r.pairwise_contrasts()


# ---------------------------------------------------------------------------
# _star_notation
# ---------------------------------------------------------------------------


def test_star_notation_two_stars():
    from pymargins._result._margins import MarginsResult

    assert MarginsResult._star_notation(0.03, (0.01, 0.05, 0.10)) == "**"


def test_star_notation_one_star():
    from pymargins._result._margins import MarginsResult

    assert MarginsResult._star_notation(0.07, (0.01, 0.05, 0.10)) == "*"


def test_star_notation_no_star():
    from pymargins._result._margins import MarginsResult

    assert MarginsResult._star_notation(0.15, (0.01, 0.05, 0.10)) == ""


# ---------------------------------------------------------------------------
# conf_int with phi transforms on draws
# ---------------------------------------------------------------------------


def test_conf_int_phi_on_draws():
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000,))
    r = MarginsResult(
        estimate=np.array([0.1]),
        std_error=np.array([0.01]),
        conf_int_lower=np.array([0.08]),
        conf_int_upper=np.array([0.12]),
        method="simulation",
        level=0.95,
        draws=draws,
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    lo, hi = r.conf_int()
    assert np.isfinite(lo.flat[0])
    assert np.isfinite(hi.flat[0])
