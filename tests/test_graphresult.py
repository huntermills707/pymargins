"""Tests for self-contained GraphResult.

Design §7.1, req §6. Added in 0.4.0 (R4).
"""

from __future__ import annotations

import os
import pickle
import tempfile
import weakref

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins._graph._plan import Plan
from pymargins._result._graphresult import GraphResult
from pymargins._result._intervals import supt_interval_delta, supt_interval_draws

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _plan(**kw):
    defaults = dict(
        method_resolved="delta",
        method_declared="delta",
        scale="response",
        level=0.95,
        ci=None,
        B=0,
        n_sim=0,
        seed=42,
    )
    defaults.update(kw)
    return Plan(**defaults)


def _delta_result_data(**kw):
    defaults = dict(
        estimate=np.array(1.0),
        std_error=np.array(0.1),
        conf_int_lower=np.array(0.8),
        conf_int_upper=np.array(1.2),
        method="delta",
        level=0.95,
        kappa=np.array(0.05),
        delta_sim_disagreement=None,
        fallback_triggered=False,
        fallback_reason=None,
        gradient=np.array([1.0, 0.0]),
        cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
        draws=None,
        draws_inf=None,
        estimand_metadata={"labels": ["x1"]},
        ci_method=None,
        bootstrap_extras=None,
        n_boot_effective=None,
        n_boot_failed=None,
    )
    defaults.update(kw)
    return defaults


def _sim_result_data(**kw):
    rng = np.random.default_rng(7)
    draws_inf = rng.normal(loc=1.0, scale=0.1, size=2000)
    se = float(np.std(draws_inf, ddof=1))
    lo = np.quantile(draws_inf, 0.025)
    hi = np.quantile(draws_inf, 0.975)
    defaults = dict(
        estimate=np.array(1.0),
        std_error=np.array(se),
        conf_int_lower=np.array(lo),
        conf_int_upper=np.array(hi),
        method="simulation",
        level=0.95,
        kappa=None,
        delta_sim_disagreement=None,
        fallback_triggered=False,
        fallback_reason=None,
        gradient=None,
        cov_params=None,
        draws=None,
        draws_inf=draws_inf,
        estimand_metadata={"labels": ["x1"]},
        ci_method="percentile",
        bootstrap_extras=None,
        n_boot_effective=None,
        n_boot_failed=None,
    )
    defaults.update(kw)
    return defaults


def _graph_from_data(data, **kw):
    plan = kw.pop("plan", _plan(method_resolved=data["method"]))
    return GraphResult.from_engine(
        data,
        plan=plan,
        labels=data.get("estimand_metadata", {}).get("labels"),
        population_note=kw.pop("population_note", None),
        n_obs=kw.pop("n_obs", 100),
        psi_h=kw.pop("psi_h", None),
        phi=kw.pop("phi", None),
        phi_inv=kw.pop("phi_inv", None),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_engine_delta():
    gr = _graph_from_data(_delta_result_data())
    assert gr.method == "delta"
    assert gr.level == 0.95
    assert gr.scale == "response"
    assert np.allclose(gr.estimate, 1.0)


def test_from_engine_simulation():
    gr = _graph_from_data(_sim_result_data())
    assert gr.method == "simulation"
    assert gr.draws_inf is not None
    assert gr.gradient is None


# ---------------------------------------------------------------------------
# conf_int doctrine surface
# ---------------------------------------------------------------------------


def test_conf_int_level_typeerror():
    gr = _graph_from_data(_delta_result_data())
    with pytest.raises(TypeError, match="declared at the estimator constructor"):
        gr.conf_int(level=0.90)


def test_conf_int_rejects_unknown_correction():
    gr = _graph_from_data(_delta_result_data())
    with pytest.raises(ValueError, match="correction=.*holms"):
        gr.conf_int(correction="holms")


@pytest.mark.parametrize("method", ["delta", "simulation"])
def test_corrections_only_widen(method):
    if method == "delta":
        data = _delta_result_data(
            estimate=np.array([1.0, 2.0]),
            std_error=np.array([0.1, 0.2]),
            conf_int_lower=np.array([0.8, 1.6]),
            conf_int_upper=np.array([1.2, 2.4]),
            gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
            cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
            estimand_metadata={"labels": ["a", "b"]},
        )
    else:
        rng = np.random.default_rng(8)
        draws = rng.multivariate_normal(
            mean=[1.0, 2.0], cov=np.diag([0.01, 0.04]), size=5000
        )
        data = _sim_result_data(
            estimate=np.array([1.0, 2.0]),
            std_error=np.std(draws, axis=0, ddof=1),
            conf_int_lower=np.quantile(draws, 0.025, axis=0),
            conf_int_upper=np.quantile(draws, 0.975, axis=0),
            draws_inf=draws,
            estimand_metadata={"labels": ["a", "b"]},
        )

    gr = _graph_from_data(data)
    lo_none, hi_none = gr.conf_int()
    for correction in ("bonferroni", "sidak", "sup-t"):
        lo, hi = gr.conf_int(correction=correction)
        assert np.all(lo <= lo_none)
        assert np.all(hi >= hi_none)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["delta", "simulation"])
def test_supt_delta_vs_draws_consistency(method):
    """Sup-t from MVN and from a large MC draw sample agree (≈5%)."""
    rng = np.random.default_rng(9)
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    draws = rng.multivariate_normal(mean, cov, size=50000)

    if method == "delta":
        data = _delta_result_data(
            estimate=mean,
            std_error=np.sqrt(np.diag(cov)),
            conf_int_lower=mean - 1.96 * np.sqrt(np.diag(cov)),
            conf_int_upper=mean + 1.96 * np.sqrt(np.diag(cov)),
            gradient=np.eye(2),
            cov_params=cov,
            estimand_metadata={"labels": ["a", "b"]},
        )
    else:
        data = _sim_result_data(
            estimate=mean,
            std_error=np.std(draws, axis=0, ddof=1),
            conf_int_lower=np.quantile(draws, 0.025, axis=0),
            conf_int_upper=np.quantile(draws, 0.975, axis=0),
            draws_inf=draws,
            estimand_metadata={"labels": ["a", "b"]},
        )

    gr = _graph_from_data(data)
    lo_delta, hi_delta = supt_interval_delta(
        gr._inference_estimate(),
        np.eye(2),
        cov,
        gr.level,
        phi=gr.phi,
    )
    lo_draws, hi_draws = supt_interval_draws(
        gr.draws_inf if gr.draws_inf is not None else draws,
        gr._inference_estimate(),
        gr.std_error,
        gr.level,
        phi=gr.phi,
    )
    np.testing.assert_allclose(lo_delta, lo_draws, rtol=0.05)
    np.testing.assert_allclose(hi_delta, hi_draws, rtol=0.05)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def test_test_delta():
    gr = _graph_from_data(_delta_result_data(estimate=np.array(2.0)))
    tr = gr.test(value=0.0, null_scale="inference")
    assert float(tr.pvalue) < 0.01


def test_test_simulation():
    rng = np.random.default_rng(10)
    draws = rng.normal(loc=2.0, scale=0.5, size=2000)
    data = _sim_result_data(
        estimate=np.array(2.0),
        std_error=np.std(draws, ddof=1),
        conf_int_lower=np.quantile(draws, 0.025),
        conf_int_upper=np.quantile(draws, 0.975),
        draws_inf=draws,
    )
    gr = _graph_from_data(data)
    tr = gr.test(value=0.0, null_scale="inference")
    assert float(tr.pvalue) < 0.01


def test_joint_test_delta():
    data = _delta_result_data(
        estimate=np.array([1.0, 0.5]),
        std_error=np.array([0.5, 0.5]),
        conf_int_lower=np.array([0.0, -0.5]),
        conf_int_upper=np.array([2.0, 1.5]),
        gradient=np.eye(2),
        cov_params=np.eye(2),
        estimand_metadata={"labels": ["a", "b"]},
    )
    gr = _graph_from_data(data)
    tr = gr.joint_test()
    assert tr.df == 2
    assert float(tr.pvalue) < 1.0


# ---------------------------------------------------------------------------
# Summary / formatting
# ---------------------------------------------------------------------------


def test_summary_contains_plan_hash():
    plan = _plan()
    gr = _graph_from_data(_delta_result_data(), plan=plan)
    summary = gr.summary()
    assert plan.hash in summary


def test_summary_contains_population_note():
    gr = _graph_from_data(
        _delta_result_data(),
        plan=_plan(),
        population_note="representative",
    )
    summary = gr.summary()
    assert "representative" in summary


def test_to_frame_columns():
    gr = _graph_from_data(_delta_result_data())
    frame = gr.to_frame()
    assert {"estimate", "std_error", "ci_lower", "ci_upper", "p_value"} <= set(
        frame.columns
    )


def test_to_latex_contains_table():
    gr = _graph_from_data(_delta_result_data())
    latex = gr.to_latex()
    assert "\\begin{tabular}" in latex


def test_to_html_contains_table():
    gr = _graph_from_data(_delta_result_data())
    html = gr.to_html()
    assert "<table" in html


# ---------------------------------------------------------------------------
# Outcome / contrast / scaled
# ---------------------------------------------------------------------------


def test_scaled():
    gr = _graph_from_data(_delta_result_data())
    scaled = gr.scaled(100.0, units="percentage points")
    assert np.allclose(scaled.estimate, 100.0)
    assert scaled.estimand_metadata.get("units") == "percentage points"


def test_contrast():
    data = _delta_result_data(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.2]),
        conf_int_lower=np.array([0.8, 1.6]),
        conf_int_upper=np.array([1.2, 2.4]),
        gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
        cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
        estimand_metadata={"labels": ["a", "b"]},
    )
    gr = _graph_from_data(data)
    diff = gr.contrast(np.array([[1.0, -1.0]]), labels=["a - b"])
    assert diff.labels == ["a - b"]
    assert np.allclose(diff.estimate, -1.0)


def test_pairwise_contrasts():
    data = _delta_result_data(
        estimate=np.array([1.0, 2.0, 3.0]),
        std_error=np.array([0.1, 0.2, 0.3]),
        conf_int_lower=np.array([0.8, 1.6, 2.4]),
        conf_int_upper=np.array([1.2, 2.4, 3.6]),
        gradient=np.eye(3),
        cov_params=np.eye(3),
        estimand_metadata={"labels": ["a", "b", "c"]},
    )
    gr = _graph_from_data(data)
    pw = gr.pairwise_contrasts()
    assert len(pw.estimate) == 3


# ---------------------------------------------------------------------------
# Influence
# ---------------------------------------------------------------------------


def test_psi_h_variance_identity():
    """ψ^h.T @ ψ^h ≈ ∇h.T Σ̂ ∇h for a nonrobust OLS prediction."""
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    from pymargins._adapter import auto_detect_adapter
    from pymargins._gradients import gradient

    adapter = auto_detect_adapter(fit)
    beta = jnp.asarray(adapter.coefficients())
    cov = jnp.asarray(adapter.covariance())

    # Prediction at the mean covariate profile.
    x_mean = df[["x1", "x2"]].mean().values
    x_design = jnp.concatenate([jnp.ones(1), jnp.asarray(x_mean)])

    def h(beta):
        return jnp.dot(x_design, beta)

    grad = gradient(h, beta)
    psi_beta = np.asarray(adapter.influence())  # (n, p)
    psi_h = psi_beta @ np.asarray(grad).T       # (n, 1)

    lhs = psi_h.T @ psi_h
    rhs = np.asarray(grad) @ np.asarray(cov) @ np.asarray(grad).T
    # Finite-sample residual variance prevents exact equality on real data;
    # the first-order identity holds to within a few percent.
    np.testing.assert_allclose(lhs, rhs, rtol=0.05)


# ---------------------------------------------------------------------------
# Disk round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_disk():
    plan = _plan()
    data = _delta_result_data(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.2]),
        conf_int_lower=np.array([0.8, 1.6]),
        conf_int_upper=np.array([1.2, 2.4]),
        gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
        cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
        estimand_metadata={"labels": ["a", "b"]},
    )
    gr = _graph_from_data(data, plan=plan, population_note="pop")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "result.pkl")
        gr.to_disk(path)
        gr2 = GraphResult.from_disk(path)

    assert gr2.method == gr.method
    assert gr2.level == gr.level
    assert gr2.population_note == gr.population_note
    np.testing.assert_array_equal(gr2.estimate, gr.estimate)
    np.testing.assert_array_equal(gr2.std_error, gr.std_error)
    np.testing.assert_array_equal(gr2.conf_int_lower, gr.conf_int_lower)
    np.testing.assert_array_equal(gr2.conf_int_upper, gr.conf_int_upper)
    np.testing.assert_array_equal(gr2.gradient, gr.gradient)
    np.testing.assert_array_equal(gr2.cov_params, gr.cov_params)


def test_no_session_reference():
    """GraphResult is self-contained: pickle succeeds and object graph has no weakrefs."""
    gr = _graph_from_data(_delta_result_data())
    blob = pickle.dumps(gr)
    gr2 = pickle.loads(blob)
    assert type(gr2) is GraphResult

    refs = [ref for ref in weakref.getweakrefs(gr)]
    assert refs == []


def test_to_disk_rejects_custom_phi():
    gr = _graph_from_data(_delta_result_data(), phi=lambda x: x, phi_inv=lambda x: x)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "result.pkl")
        with pytest.raises(ValueError, match="custom function"):
            gr.to_disk(path)


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


def test_joint_test_empirical():
    """kind='empirical' must not raise UnboundLocalError on nonsingular draws."""
    rng = np.random.default_rng(50)
    draws = rng.multivariate_normal(
        mean=[0.5, 0.0], cov=np.eye(2), size=2000
    )
    data = _sim_result_data(
        estimate=np.array([0.5, 0.0]),
        std_error=np.std(draws, axis=0, ddof=1),
        conf_int_lower=np.quantile(draws, 0.025, axis=0),
        conf_int_upper=np.quantile(draws, 0.975, axis=0),
        draws_inf=draws,
        estimand_metadata={"labels": ["a", "b"]},
    )
    gr = _graph_from_data(data)
    tr = gr.joint_test(kind="empirical")
    assert tr.method == "joint_empirical"
    assert 0.0 <= float(tr.pvalue) <= 1.0


def test_summary_kappa_once():
    data = _delta_result_data(kappa=np.array(0.25))
    gr = _graph_from_data(data)
    summary = gr.summary()
    # κ should appear exactly once (in the plan footer line).
    assert summary.count("κ") == 1


def test_scaled_rescales_psi_h():
    data = _delta_result_data(
        estimate=np.array([1.0]),
        gradient=np.array([1.0, 0.0]),
        cov_params=np.eye(2),
    )
    psi = np.ones((10, 1))
    gr = _graph_from_data(data, psi_h=psi)
    scaled = gr.scaled(100.0)
    np.testing.assert_array_equal(scaled.psi_h, psi * 100.0)


def test_from_engine_with_executor_roundtrip():
    """A real execute_query → from_engine round-trip populates cov_params for delta."""
    rng = np.random.default_rng(51)
    n = 120
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    from pymargins import GComputation
    from pymargins._engine._banks import BankSet
    from pymargins._engine._execute import execute_query
    from pymargins._engine._queries import (
        QueryContext,
        QuerySpec,
        WiringFacts,
        build_inference_config,
        compile_query,
    )

    est = GComputation(fit, at="overall", method="delta")
    ctx = QueryContext(
        adapter=est._compiled.adapter,
        base_data=est._compiled.base_data,
        at="overall",
        weights=None,
        phi=None,
        phi_inv=None,
        fd_step=1e-6,
        gradient_backend="autodiff",
    )
    compiled = compile_query(QuerySpec(kind="predict"), ctx)
    plan = _plan(method_resolved="delta")
    facts = WiringFacts()
    banks = BankSet(plan.plan_hash, 0, plan.seed)
    config = build_inference_config(plan, ctx.adapter, facts, banks)

    result = execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=config.cov_params,
    )

    gr = GraphResult.from_engine(
        result,
        plan=plan,
        labels=result.get("estimand_metadata", {}).get("labels"),
        n_obs=len(df),
        phi=None,
        phi_inv=None,
    )

    assert gr.cov_params is not None
    assert np.all(np.isfinite(gr.cov_params))
    # These post-estimation ops require cov_params; they must not raise.
    lo, hi = gr.conf_int(correction="bonferroni")
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    tr = gr.test(value=0.0, null_scale="inference")
    assert np.all(np.isfinite(tr.pvalue))
