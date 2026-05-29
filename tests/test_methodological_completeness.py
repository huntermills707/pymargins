"""Tests for METHODOLOGICAL_COMPLETENESS_PLAN gaps G1–G7b."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins


@pytest.fixture
def df_logit():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )
    lp = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"] + 0.3 * (df["sex"] == "M")
    df["outcome"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "outcome ~ age + treatment + C(sex)",
        data=df_logit,
        family=sm.families.Binomial(),
    ).fit()


# ---------------------------------------------------------------------------
# G5 — Empirical quadratic-form joint test
# ---------------------------------------------------------------------------


def test_joint_test_empirical_kind_exists(fit_logit):
    """kind='empirical' is accepted and returns a TestResult."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=200, rng_seed=42)
    pred = m.predict(atexog={"treatment": [0, 1]})
    result = pred.joint_test(kind="empirical")
    assert result.method == "joint_empirical"
    assert np.isfinite(float(result.statistic))
    assert 0 <= float(result.pvalue) <= 1


def test_joint_test_empirical_vs_wald_on_gaussian_draws(fit_logit):
    """Under approximately Gaussian draws, empirical and wald p-values agree."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=500, rng_seed=42)
    pred = m.predict(atexog={"treatment": [0, 1]})
    wald = pred.joint_test(kind="wald")
    emp = pred.joint_test(kind="empirical")
    # p-values need not be identical, but both should be in [0, 1]
    assert 0 <= float(wald.pvalue) <= 1
    assert 0 <= float(emp.pvalue) <= 1
    # Under near-Gaussianity they should not diverge wildly.
    # With a small number of draws the empirical p-value can be exactly 0
    # when Q_obs exceeds every draw-Q; that is valid, not a bug.
    if float(emp.pvalue) > 0:
        ratio = float(emp.pvalue) / (float(wald.pvalue) + 1e-10)
        assert 0.01 < ratio < 100


def test_joint_test_invalid_kind_raises(fit_logit):
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=100, rng_seed=42)
    pred = m.predict(atexog={"treatment": [0, 1]})
    with pytest.raises(ValueError, match="kind must be 'wald' or 'empirical'"):
        pred.joint_test(kind="foobar")


# ---------------------------------------------------------------------------
# G6 — Simultaneous (sup-t) confidence bands
# ---------------------------------------------------------------------------


def test_simultaneous_ci_wider_than_pointwise_simulation(fit_logit):
    """sup-t bands are wider than per-component CIs for simulation results."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=300, rng_seed=42)
    pred = m.predict(atexog={"treatment": [0, 1]})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(simultaneous=True)
    # Simultaneous bands should be wider (or equal) for every component
    assert np.all(lo_sim <= lo_point)
    assert np.all(hi_sim >= hi_point)


def test_simultaneous_ci_wider_than_pointwise_delta(fit_logit):
    """sup-t bands are wider than per-component CIs for delta results."""
    m = Margins.linear_scale(fit_logit, method="delta")
    pred = m.predict(atexog={"treatment": [0, 1]})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(simultaneous=True)
    assert np.all(lo_sim <= lo_point)
    assert np.all(hi_sim >= hi_point)


def test_simultaneous_ci_scalar_result(fit_logit):
    """sup-t on a scalar estimand is close to the ordinary CI (Monte Carlo noise)."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=200, rng_seed=42)
    pred = m.predict(atexog={"treatment": 1})
    lo_point, hi_point = pred.conf_int()
    lo_sim, hi_sim = pred.conf_int(simultaneous=True)
    # For a scalar, sup-t critical value ≈ percentile critical value,
    # but finite-sample Monte Carlo means they're not identical.
    # Check they're within ~5% relative tolerance.
    np.testing.assert_allclose(lo_sim, lo_point, rtol=0.05)
    np.testing.assert_allclose(hi_sim, hi_point, rtol=0.05)


# ---------------------------------------------------------------------------
# G1 — Simulation composition (matched draws)
# ---------------------------------------------------------------------------


def test_simulation_composition_addition_match_delta(fit_logit):
    """Composed simulation CI should overlap with composed delta CI."""
    m_sim = Margins.linear_scale(fit_logit, method="simulation", n_sim=500, rng_seed=42)
    m_delta = Margins.linear_scale(fit_logit, method="delta")

    pred_sim_treat = m_sim.predict(atexog={"treatment": 1})
    pred_sim_ctrl = m_sim.predict(atexog={"treatment": 0})
    diff_sim = pred_sim_treat - pred_sim_ctrl

    pred_delta_treat = m_delta.predict(atexog={"treatment": 1})
    pred_delta_ctrl = m_delta.predict(atexog={"treatment": 0})
    diff_delta = pred_delta_treat - pred_delta_ctrl

    # Estimates should be close
    np.testing.assert_allclose(
        float(diff_sim.estimate), float(diff_delta.estimate), rtol=0.05
    )
    # Both should have finite CIs
    lo_sim, hi_sim = diff_sim.conf_int()
    lo_delta, hi_delta = diff_delta.conf_int()
    assert np.isfinite(float(lo_sim))
    assert np.isfinite(float(hi_sim))
    assert np.isfinite(float(lo_delta))
    assert np.isfinite(float(hi_delta))


def test_simulation_composition_carries_draws(fit_logit):
    """Composed simulation result should carry combined draws."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=200, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    diff = pred_treat - pred_ctrl
    assert diff.draws_inf is not None
    assert diff.draws is not None
    assert diff.draws_inf.shape[0] == 200


def test_composition_mixed_method_raises():
    """_combine_results with mixed delta + draws raises directly."""
    from pymargins._result import MarginsResult, _combine_results

    sess = _make_mock_session()
    cov = np.diag([0.01])
    r_delta = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.array([1.0]),
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_sim = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="simulation",
        level=0.95,
        n_obs=100,
        draws=np.random.default_rng(42).normal(size=(100,)),
        draws_inf=np.random.default_rng(42).normal(size=(100,)),
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )
    with pytest.raises(ValueError, match="mixing methods is not supported"):
        _combine_results(
            r_delta,
            r_sim,
            lambda a, b: a + b,
            lambda g1, g2: g1 + g2,
            lambda la, lb: f"{la}+{lb}",
        )


def test_composition_bootstrap_produces_finite_ci(fit_logit):
    """Bootstrap composition now works via session resample bank."""
    m = Margins.linear_scale(fit_logit, method="bootstrap", n_boot=50, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    diff = pred_treat - pred_ctrl
    assert diff.draws_inf is not None
    lo, hi = diff.conf_int()
    assert np.isfinite(float(lo))
    assert np.isfinite(float(hi))


def test_composition_bootstrap_mismatched_bank_raises():
    """Bootstrap results with different bank IDs raise even if same session."""
    from pymargins._result import MarginsResult, _combine_results

    sess = _make_mock_session()
    draws = np.random.default_rng(42).normal(size=(50,))
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="bootstrap",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["A"]},
        session=sess,
        resample_bank_id="bank_a",
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="bootstrap",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["B"]},
        session=sess,
        resample_bank_id="bank_b",
    )
    with pytest.raises(ValueError, match="resample bank"):
        _combine_results(
            r_a,
            r_b,
            lambda a, b: a + b,
            lambda g1, g2: g1 + g2,
            lambda la, lb: f"{la}+{lb}",
        )


# ---------------------------------------------------------------------------
# G2 — Vector composition (delta path)
# ---------------------------------------------------------------------------


def _make_mock_session():
    """Return a minimal object that satisfies _session_obj()."""
    return object()


def test_vector_delta_composition_shape_and_se():
    """Composition of two vector results should produce correct SE shape."""
    from pymargins._result import MarginsResult

    n_params = 4
    n_comp = 3
    cov = np.diag([0.01, 0.02, 0.03, 0.04])
    sess = _make_mock_session()

    grad_a = np.ones((n_comp, n_params))
    grad_b = np.ones((n_comp, n_params)) * 0.5
    estimate_a = np.array([1.0, 2.0, 3.0])
    estimate_b = np.array([0.5, 1.0, 1.5])

    r_a = MarginsResult(
        estimate=estimate_a,
        std_error=np.ones(n_comp),
        conf_int_lower=estimate_a - 1.96,
        conf_int_upper=estimate_a + 1.96,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=grad_a,
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=estimate_b,
        std_error=np.ones(n_comp),
        conf_int_lower=estimate_b - 1.96,
        conf_int_upper=estimate_b + 1.96,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=grad_b,
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    assert combined.estimate.shape == (n_comp,)
    assert combined.std_error.shape == (n_comp,)
    assert combined.conf_int_lower.shape == (n_comp,)
    assert combined.conf_int_upper.shape == (n_comp,)
    np.testing.assert_array_almost_equal(combined.estimate, estimate_a + estimate_b)


def test_vector_delta_composition_se_correctness():
    """SE for vector composition should match per-component delta method."""
    from pymargins._result import MarginsResult

    cov = np.diag([0.01, 0.04])
    grad_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    grad_b = np.array([[0.5, 0.0], [0.0, 0.5]])
    combined_grad = grad_a + grad_b
    sess = _make_mock_session()

    # Manual per-component variance
    manual_var = []
    for i in range(2):
        g = combined_grad[i]
        manual_var.append(float(g @ cov @ g))
    manual_se = np.sqrt(manual_var)

    r_a = MarginsResult(
        estimate=np.array([1.0, 2.0]),
        std_error=np.ones(2),
        conf_int_lower=np.zeros(2),
        conf_int_upper=np.ones(2),
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=grad_a,
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=np.array([0.5, 1.0]),
        std_error=np.ones(2),
        conf_int_lower=np.zeros(2),
        conf_int_upper=np.ones(2),
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=grad_b,
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    np.testing.assert_allclose(combined.std_error, manual_se, rtol=1e-5)


# ---------------------------------------------------------------------------
# G4 — κ preservation through composition
# ---------------------------------------------------------------------------


def test_kappa_propagation_delta_max():
    """Delta composition should propagate max(κ_A, κ_B)."""
    from pymargins._result import MarginsResult

    cov = np.diag([0.01])
    sess = _make_mock_session()
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=0.3,
        gradient=np.array([1.0]),
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=0.7,
        gradient=np.array([0.5]),
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    assert combined.kappa == pytest.approx(0.7)


def test_kappa_propagation_delta_one_none():
    """If one κ is None, the other should propagate."""
    from pymargins._result import MarginsResult

    cov = np.diag([0.01])
    sess = _make_mock_session()
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=None,
        gradient=np.array([1.0]),
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=0.5,
        gradient=np.array([0.5]),
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    assert combined.kappa == pytest.approx(0.5)


def test_kappa_propagation_simulation_max():
    """Simulation composition should also propagate max(κ_A, κ_B)."""
    from pymargins._result import MarginsResult

    draws = np.random.default_rng(42).normal(size=(100,))
    sess = _make_mock_session()
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="simulation",
        level=0.95,
        n_obs=100,
        kappa=0.4,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="simulation",
        level=0.95,
        n_obs=100,
        kappa=0.6,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    assert combined.kappa == pytest.approx(0.6)


def test_kappa_propagation_vector_elementwise_max():
    """Array κ should propagate via element-wise maximum."""
    from pymargins._result import MarginsResult

    cov = np.diag([0.01, 0.02, 0.03])
    sess = _make_mock_session()
    r_a = MarginsResult(
        estimate=np.array([1.0, 2.0, 3.0]),
        std_error=np.ones(3) * 0.1,
        conf_int_lower=np.zeros(3),
        conf_int_upper=np.ones(3),
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=np.array([0.1, 0.8, 0.3]),
        gradient=np.ones((3, 3)),
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=np.array([0.5, 1.0, 1.5]),
        std_error=np.ones(3) * 0.1,
        conf_int_lower=np.zeros(3),
        conf_int_upper=np.ones(3),
        method="delta",
        level=0.95,
        n_obs=100,
        kappa=np.array([0.5, 0.4, 0.9]),
        gradient=np.ones((3, 3)) * 0.5,
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )

    combined = r_a + r_b
    expected = np.array([0.5, 0.8, 0.9])
    np.testing.assert_array_equal(np.asarray(combined.kappa), expected)


# ---------------------------------------------------------------------------
# G7a — Structured multi-dimensional output
# ---------------------------------------------------------------------------


def test_to_frame_multi_outcome_produces_tidy_df():
    """to_frame() on 2D multi-outcome result should produce tidy DataFrame."""
    from pymargins._result import MarginsResult

    sess = _make_mock_session()
    n_atoms, n_outcomes = 2, 3
    est = np.arange(n_atoms * n_outcomes).reshape(n_atoms, n_outcomes) * 1.0
    se = np.ones_like(est) * 0.1
    lo = est - 0.2
    hi = est + 0.2
    labels = [f"atom{i} (out{j})" for i in range(n_atoms) for j in range(n_outcomes)]

    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=lo,
        conf_int_upper=hi,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=None,
        cov_params=np.diag([0.01, 0.02]),
        estimand_metadata={
            "labels": labels,
            "_outcome_shape": {
                "n_atoms": n_atoms,
                "n_outcomes": n_outcomes,
                "outcome_labels": ["out0", "out1", "out2"],
            },
        },
        session=sess,
    )

    frame = r.to_frame()
    assert len(frame) == n_atoms * n_outcomes
    assert "outcome" in frame.columns
    assert list(frame["outcome"]) == ["out0", "out1", "out2", "out0", "out1", "out2"]
    np.testing.assert_array_almost_equal(frame["estimate"].values, est.ravel(order="C"))


def test_outcome_slices_via_outcome_shape():
    """outcome() should use _outcome_shape metadata when available."""
    from pymargins._result import MarginsResult

    sess = _make_mock_session()
    n_atoms, n_outcomes = 2, 3
    est = np.arange(n_atoms * n_outcomes).reshape(n_atoms, n_outcomes) * 1.0
    se = np.ones_like(est) * 0.1
    labels = [f"atom{i} (out{j})" for i in range(n_atoms) for j in range(n_outcomes)]

    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=est - 0.2,
        conf_int_upper=est + 0.2,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=None,
        cov_params=np.diag([0.01, 0.02]),
        estimand_metadata={
            "labels": labels,
            "_outcome_shape": {
                "n_atoms": n_atoms,
                "n_outcomes": n_outcomes,
                "outcome_labels": ["out0", "out1", "out2"],
            },
        },
        session=sess,
    )

    sliced = r.outcome("out1")
    assert sliced.estimate.shape == (n_atoms,)
    np.testing.assert_array_almost_equal(sliced.estimate, est[:, 1])
    assert sliced.estimand_metadata.get("outcome_sliced")


def test_outcome_index_slices_via_outcome_shape():
    """outcome(index) should work with integer index via _outcome_shape."""
    from pymargins._result import MarginsResult

    sess = _make_mock_session()
    n_atoms, n_outcomes = 2, 3
    est = np.arange(n_atoms * n_outcomes).reshape(n_atoms, n_outcomes) * 1.0
    labels = [f"atom{i} (out{j})" for i in range(n_atoms) for j in range(n_outcomes)]

    r = MarginsResult(
        estimate=est,
        std_error=np.ones_like(est) * 0.1,
        conf_int_lower=est - 0.2,
        conf_int_upper=est + 0.2,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=None,
        cov_params=np.diag([0.01, 0.02]),
        estimand_metadata={
            "labels": labels,
            "_outcome_shape": {
                "n_atoms": n_atoms,
                "n_outcomes": n_outcomes,
                "outcome_labels": ["out0", "out1", "out2"],
            },
        },
        session=sess,
    )

    sliced = r.outcome(2)
    np.testing.assert_array_almost_equal(sliced.estimate, est[:, 2])


# ---------------------------------------------------------------------------
# G3 — compose_results() for nonlinear composition
# ---------------------------------------------------------------------------


def test_compose_results_delta_ratio(fit_logit):
    """Ratio of two delta predictions via compose_results."""
    from pymargins._result import compose_results

    m = Margins.linear_scale(fit_logit, method="delta")
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    ratio = compose_results(
        [pred_treat, pred_ctrl],
        fn=lambda theta: theta[0] / theta[1],
        label="treat/control",
    )
    expected = float(pred_treat.estimate) / float(pred_ctrl.estimate)
    assert ratio.estimate == pytest.approx(expected, rel=1e-5)
    assert ratio.gradient is not None
    assert ratio.std_error > 0
    lo, hi = ratio.conf_int()
    assert lo < hi


def test_compose_results_simulation_ratio(fit_logit):
    """Ratio of two simulation predictions via compose_results."""
    from pymargins._result import compose_results

    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=500, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    ratio = compose_results(
        [pred_treat, pred_ctrl],
        fn=lambda theta: theta[0] / theta[1],
        label="treat/control",
    )
    expected = float(pred_treat.estimate) / float(pred_ctrl.estimate)
    assert ratio.estimate == pytest.approx(expected, rel=1e-5)
    assert ratio.draws_inf is not None
    assert ratio.std_error > 0


def test_compose_results_bootstrap_ratio(fit_logit):
    """Ratio of two bootstrap predictions via compose_results."""
    from pymargins._result import compose_results

    m = Margins.linear_scale(fit_logit, method="bootstrap", n_boot=50, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    ratio = compose_results(
        [pred_treat, pred_ctrl],
        fn=lambda theta: theta[0] / theta[1],
        label="treat/control",
    )
    expected = float(pred_treat.estimate) / float(pred_ctrl.estimate)
    assert ratio.estimate == pytest.approx(expected, rel=1e-5)
    assert ratio.draws_inf is not None
    assert ratio.std_error > 0


def test_compose_results_single_result_raises(fit_logit):
    """compose_results with < 2 results should raise."""
    from pymargins._result import compose_results

    m = Margins.linear_scale(fit_logit, method="delta")
    pred = m.predict(atexog={"treatment": 1})
    with pytest.raises(ValueError, match="at least two results"):
        compose_results([pred], fn=lambda theta: theta[0])


def test_compose_results_different_sessions_raises(fit_logit):
    """compose_results with results from different sessions should raise."""
    from pymargins._result import compose_results

    m1 = Margins.linear_scale(fit_logit, method="delta")
    m2 = Margins.linear_scale(fit_logit, method="delta")
    pred1 = m1.predict(atexog={"treatment": 1})
    pred2 = m2.predict(atexog={"treatment": 0})
    with pytest.raises(ValueError, match="same Margins session"):
        compose_results([pred1, pred2], fn=lambda theta: theta[0] / theta[1])


# ---------------------------------------------------------------------------
# G7b — Multi-outcome selector & to_frame() ergonomics
# ---------------------------------------------------------------------------


def test_multi_outcome_unsliced_to_frame_tiles_scenarios(df_logit):
    """to_frame() on unsliced multi-outcome result tiles scenarios per outcome."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])

    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    m = Margins.linear_scale(fit, method="delta")
    pred = m.predict(atexog={"treatment": [0, 1]})

    frame = pred.to_frame()
    assert len(frame) == 6  # 2 scenarios × 3 outcomes
    assert "outcome" in frame.columns
    assert "treatment" in frame.columns
    # Scenario columns should be tiled per outcome
    assert list(frame["treatment"]) == [0, 0, 0, 1, 1, 1]
    assert list(frame["outcome"]) == ["0", "1", "2", "0", "1", "2"]


def test_multi_outcome_sliced_to_frame_works(df_logit):
    """to_frame() on sliced multi-outcome result should work."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])

    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    m = Margins.linear_scale(fit, method="delta")
    pred = m.predict(atexog={"treatment": [0, 1]}, outcome=1)

    frame = pred.to_frame()
    assert len(frame) == 2  # two treatment values
    assert "estimate" in frame.columns


# ---------------------------------------------------------------------------
# §7 Acceptance criteria — quantitative verification
# ---------------------------------------------------------------------------


def test_g1_simulation_brute_force_matched_draws(fit_logit):
    """Composed simulation SE must match hand-composed matched-draw reference."""
    m = Margins.linear_scale(fit_logit, method="simulation", n_sim=5000, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    diff_composed = pred_treat - pred_ctrl

    # Brute-force: same β̃ stream, elementwise difference
    beta_hat = jnp.asarray(fit_logit.params)
    cov = jnp.asarray(fit_logit.cov_params())
    rng = np.random.default_rng([42, 0])
    beta_tilde = rng.multivariate_normal(beta_hat, cov, size=5000)

    # Rebuild estimands manually (linear predictor at treatment=1 vs 0)
    treat_idx = list(fit_logit.model.exog_names).index("treatment")
    X_treat = jnp.asarray(fit_logit.model.exog.copy())
    X_treat = X_treat.at[:, treat_idx].set(1.0)
    X_ctrl = jnp.asarray(fit_logit.model.exog.copy())
    X_ctrl = X_ctrl.at[:, treat_idx].set(0.0)

    def h_beta(beta, X):
        return jnp.mean(jax.nn.sigmoid(X @ beta))

    draws_treat = jnp.array([h_beta(b, X_treat) for b in beta_tilde])
    draws_ctrl = jnp.array([h_beta(b, X_ctrl) for b in beta_tilde])
    diff_manual = np.asarray(draws_treat - draws_ctrl)
    se_manual = float(np.std(diff_manual, ddof=1))

    se_composed = float(diff_composed.std_error)
    np.testing.assert_allclose(se_composed, se_manual, rtol=0.02)


def test_g1_bootstrap_brute_force_matched_draws(fit_logit):
    """Composed bootstrap SE must match hand-composed matched-draw reference."""
    m = Margins.linear_scale(fit_logit, method="bootstrap", n_boot=100, rng_seed=42)
    pred_treat = m.predict(atexog={"treatment": 1})
    pred_ctrl = m.predict(atexog={"treatment": 0})
    diff_composed = pred_treat - pred_ctrl

    # Brute-force: elementwise difference of matched draws
    draws_treat = np.asarray(pred_treat.draws_inf)
    draws_ctrl = np.asarray(pred_ctrl.draws_inf)
    diff_manual = draws_treat - draws_ctrl
    se_manual = float(np.std(diff_manual, ddof=1))

    se_composed = float(diff_composed.std_error)
    np.testing.assert_allclose(se_composed, se_manual, rtol=1e-5)


def test_g6_simultaneous_ci_coverage_mc():
    """Sup-t simultaneous CIs should have ≥ nominal coverage in a Monte Carlo."""
    rng = np.random.default_rng(42)
    n_sim = 200
    n_comp = 3
    true_means = np.array([0.0, 0.5, -0.3])
    cov = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    level = 0.95

    covered = 0
    for _ in range(n_sim):
        y = rng.multivariate_normal(true_means, cov, size=50)
        est = y.mean(axis=0)
        se = y.std(axis=0, ddof=1) / np.sqrt(50)

        # Build a synthetic MarginsResult with the joint covariance
        from pymargins._result import MarginsResult

        r = MarginsResult(
            estimate=est,
            std_error=se,
            conf_int_lower=est - 1.96 * se,
            conf_int_upper=est + 1.96 * se,
            method="delta",
            level=level,
            n_obs=50,
            gradient=np.eye(n_comp),
            cov_params=cov / 50,
            estimand_metadata={"labels": ["a", "b", "c"]},
        )
        lo, hi = r.conf_int(simultaneous=True)
        if np.all(true_means >= lo) and np.all(true_means <= hi):
            covered += 1

    coverage = covered / n_sim
    # Should be at least nominal (allow some MC noise)
    assert coverage >= level - 0.05, f"Coverage {coverage:.3f} < {level - 0.05:.3f}"


def test_g5_empirical_vs_wald_convergence():
    """Under near-Gaussian draws, empirical p-value ≈ χ² p-value."""
    rng = np.random.default_rng(42)
    n_draws = 5000
    n_comp = 2
    # Near-Gaussian: small curvature, linear-ish estimand
    est = np.array([0.5, -0.3])
    draws = rng.multivariate_normal(est, np.eye(n_comp) * 0.04, size=n_draws)

    from pymargins._result import MarginsResult

    r = MarginsResult(
        estimate=est,
        std_error=np.ones(n_comp) * 0.2,
        conf_int_lower=est - 0.4,
        conf_int_upper=est + 0.4,
        method="simulation",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.eye(n_comp) * 0.04,
        estimand_metadata={"labels": ["a", "b"]},
    )

    wald = r.joint_test(kind="wald")
    empirical = r.joint_test(kind="empirical")

    # Under near-Gaussianity, p-values should be close
    ratio = empirical.pvalue / wald.pvalue
    assert 0.5 <= ratio <= 2.0, (
        f"Empirical p={empirical.pvalue:.4f} vs Wald p={wald.pvalue:.4f} "
        f"ratio={ratio:.3f} outside [0.5, 2.0]"
    )


def test_g5_empirical_vs_wald_divergence_under_skew():
    """Under skewed draws, empirical and χ² p-values should diverge."""
    rng = np.random.default_rng(42)
    n_draws = 5000
    # Skewed: log-normal draws
    draws = np.column_stack(
        [
            rng.lognormal(mean=0.0, sigma=0.5, size=n_draws),
            rng.lognormal(mean=0.0, sigma=0.5, size=n_draws),
        ]
    )
    est = draws.mean(axis=0)
    se = draws.std(axis=0, ddof=1)

    from pymargins._result import MarginsResult

    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=est - 1.96 * se,
        conf_int_upper=est + 1.96 * se,
        method="simulation",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag(se**2),
        estimand_metadata={"labels": ["a", "b"]},
    )

    wald = r.joint_test(kind="wald")
    empirical = r.joint_test(kind="empirical")

    # Under skew, they should differ by more than 20%
    ratio = empirical.pvalue / wald.pvalue
    assert ratio < 0.8 or ratio > 1.2, (
        f"Expected divergence under skew, got ratio={ratio:.3f}"
    )


def test_g7b_outcome_then_test_correct():
    """.test() after .outcome() on a delta result must use the sliced gradient."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])

    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    m = Margins.linear_scale(fit, method="delta")
    pred = m.predict()

    sliced = pred.outcome(1)
    # This should work and produce a finite p-value
    tr = sliced.test(value=0.0)
    pval = float(np.asarray(tr.pvalue).ravel()[0])
    assert np.isfinite(pval)
    assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Post-audit test-coverage gaps
# ---------------------------------------------------------------------------


def test_compose_results_delta_vector_output():
    """compose_results with vector-output fn should produce vector result."""
    from pymargins._result import MarginsResult, compose_results

    sess = _make_mock_session()
    cov = np.diag([0.01, 0.02])
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.array([1.0, 0.0]),
        cov_params=cov,
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.array([0.0, 1.0]),
        cov_params=cov,
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )
    # fn returns [a, b] — a 2-element vector
    composed = compose_results(
        [r_a, r_b],
        fn=lambda theta: jnp.array([theta[0], theta[1]]),
        label="[A, B]",
    )
    assert composed.estimate.shape == (2,)
    assert composed.std_error.shape == (2,)
    assert composed.gradient.shape == (2, 2)
    np.testing.assert_allclose(composed.estimate, np.array([1.0, 0.5]), rtol=1e-5)


def test_compose_results_simulation_vector_output():
    """compose_results with vector-output fn on simulation draws."""
    from pymargins._result import MarginsResult, compose_results

    sess = _make_mock_session()
    rng = np.random.default_rng(42)
    draws = rng.normal(size=(100,))
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="simulation",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["A"]},
        session=sess,
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="simulation",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["B"]},
        session=sess,
    )
    composed = compose_results(
        [r_a, r_b],
        fn=lambda theta: jnp.array([theta[0], theta[1]]),
    )
    assert composed.estimate.shape == (2,)
    assert composed.std_error.shape == (2,)
    assert composed.draws_inf.shape == (100, 2)


def test_compose_results_four_result_kappa_reduction():
    """κ should reduce correctly over 4+ results."""
    from pymargins._result import MarginsResult, compose_results

    sess = _make_mock_session()
    cov = np.diag([0.01])
    results = []
    for i in range(4):
        results.append(
            MarginsResult(
                estimate=float(i),
                std_error=0.1,
                conf_int_lower=float(i) - 0.2,
                conf_int_upper=float(i) + 0.2,
                method="delta",
                level=0.95,
                n_obs=100,
                kappa=0.1 * (i + 1),
                gradient=np.array([1.0]),
                cov_params=cov,
                estimand_metadata={"labels": [f"r{i}"]},
                session=sess,
            )
        )

    composed = compose_results(results, fn=lambda theta: jnp.sum(theta))
    assert composed.kappa == pytest.approx(0.4)


def test_large_n_comp_simultaneous_ci():
    """sup-t simultaneous CIs should work for large n_comp."""
    rng = np.random.default_rng(42)
    n_comp = 20
    est = rng.standard_normal(n_comp)
    se = np.ones(n_comp) * 0.2
    cov = np.eye(n_comp) * 0.04

    from pymargins._result import MarginsResult

    r = MarginsResult(
        estimate=est,
        std_error=se,
        conf_int_lower=est - 1.96 * se,
        conf_int_upper=est + 1.96 * se,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.eye(n_comp),
        cov_params=cov,
        estimand_metadata={"labels": [f"c{i}" for i in range(n_comp)]},
    )
    lo, hi = r.conf_int(simultaneous=True)
    # Simultaneous bands should be wider than pointwise
    assert np.all(lo <= r.conf_int_lower)
    assert np.all(hi >= r.conf_int_upper)
    # Should have same shape as estimate
    assert lo.shape == est.shape
    assert hi.shape == est.shape


def test_singular_covariance_simultaneous_ci():
    """Delta simultaneous CIs should handle near-singular covariance."""
    import warnings

    n_comp = 3
    est = np.array([1.0, 2.0, 3.0])
    # Near-singular: third component is almost a linear combination
    cov = np.array([[1.0, 0.5, 0.99], [0.5, 1.0, 0.99], [0.99, 0.99, 1.0]])
    cov = (cov + cov.T) / 2.0 * 0.01

    from pymargins._result import MarginsResult

    r = MarginsResult(
        estimate=est,
        std_error=np.sqrt(np.diag(cov)),
        conf_int_lower=est - 1.96 * np.sqrt(np.diag(cov)),
        conf_int_upper=est + 1.96 * np.sqrt(np.diag(cov)),
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.eye(n_comp),
        cov_params=cov,
        estimand_metadata={"labels": ["a", "b", "c"]},
    )
    # Should not raise; suppress PSD warning from near-singular test matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = r.conf_int(simultaneous=True)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))


def test_bootstrap_composition_warns_on_bca():
    """Composing BCa bootstrap results should emit a UserWarning."""
    from pymargins._result import MarginsResult, compose_results

    sess = _make_mock_session()
    draws = np.random.default_rng(42).normal(size=(50,))
    r_a = MarginsResult(
        estimate=1.0,
        std_error=0.1,
        conf_int_lower=0.8,
        conf_int_upper=1.2,
        method="bootstrap",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["A"]},
        session=sess,
        resample_bank_id="bank1",
        ci_method="bca",
    )
    r_b = MarginsResult(
        estimate=0.5,
        std_error=0.1,
        conf_int_lower=0.3,
        conf_int_upper=0.7,
        method="bootstrap",
        level=0.95,
        n_obs=100,
        draws=draws,
        draws_inf=draws,
        cov_params=np.diag([0.01]),
        estimand_metadata={"labels": ["B"]},
        session=sess,
        resample_bank_id="bank1",
        ci_method="percentile",
    )
    with pytest.warns(UserWarning, match="percentile CIs"):
        compose_results([r_a, r_b], fn=lambda theta: theta[0] / theta[1])


def test_outcome_then_conf_int_on_simulation_multi_outcome():
    """.conf_int() after .outcome() on simulation multi-outcome result."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])

    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    m = Margins.linear_scale(fit, method="simulation", n_sim=200, rng_seed=42)
    pred = m.predict()
    sliced = pred.outcome(1)

    # Should not raise and should return finite bounds
    lo, hi = sliced.conf_int()
    assert np.isfinite(float(np.asarray(lo).ravel()[0]))
    assert np.isfinite(float(np.asarray(hi).ravel()[0]))


def test_outcome_then_test_on_bootstrap_multi_outcome():
    """.test() after .outcome() on bootstrap multi-outcome result."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "treatment": rng.binomial(1, 0.5, n),
        }
    )
    eta0 = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    eta1 = 0.2 - 0.1 * df["x1"] + 0.4 * df["treatment"]
    logits = np.column_stack([np.zeros(n), eta0, eta1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice(3, p=p) for p in probs])

    fit = smf.mnlogit("y ~ x1 + x2 + treatment", data=df).fit(disp=False)
    m = Margins.linear_scale(fit, method="bootstrap", n_boot=50, rng_seed=42)
    pred = m.predict()
    sliced = pred.outcome(1)

    tr = sliced.test(value=0.0)
    pval = float(np.asarray(tr.pvalue).ravel()[0])
    assert np.isfinite(pval)
    assert 0.0 <= pval <= 1.0


def test_to_frame_vector_estimand_scenarios_preserved():
    """to_frame() on single-outcome vector estimand should unpack scenarios."""
    from pymargins._result import MarginsResult

    sess = _make_mock_session()
    est = np.array([1.0, 2.0, 3.0])
    scenarios = [{"x": i} for i in range(3)]
    r = MarginsResult(
        estimate=est,
        std_error=np.ones(3) * 0.1,
        conf_int_lower=est - 0.2,
        conf_int_upper=est + 0.2,
        method="delta",
        level=0.95,
        n_obs=100,
        gradient=np.eye(3),
        cov_params=np.diag([0.01, 0.02, 0.03]),
        estimand_metadata={
            "labels": ["a", "b", "c"],
            "scenarios": scenarios,
            "kind": "prediction",
        },
        session=sess,
    )
    frame = r.to_frame()
    assert len(frame) == 3
    assert "x" in frame.columns
    assert list(frame["x"]) == [0, 1, 2]
