"""Doctrine executor tests.

Design §5, req §5. Added in 0.4.0 (R3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins._engine._banks import BankSet
from pymargins._engine._execute import execute_query
from pymargins._engine._queries import (
    QueryContext,
    QuerySpec,
    WiringFacts,
    build_inference_config,
    compile_query,
)
from pymargins._graph._plan import Plan
from pymargins._inference._config import InferenceConfig
from pymargins._inference._dispatch import run_inference
from pymargins._soundness._predicates import CompileError, SoundnessWarning
from pymargins.margins import Margins
from pymargins.survey import SurveyDesign


@pytest.fixture
def fit_logit():
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n).astype(float),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()


@pytest.fixture
def separated_logit():
    """A strongly-separated-ish logit for a high-curvature prediction."""
    rng = np.random.default_rng(4)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 2.0 + 5.0 * x1 - 3.0 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()


def ctx_from_fit(fit, base_data=None, **overrides) -> QueryContext:
    """Build a QueryContext from a statsmodels fit result."""
    m = Margins(fit, at="overall", method="delta")
    return QueryContext(
        adapter=m.adapter,
        base_data=base_data if base_data is not None else m._base_data,
        at=overrides.get("at", "overall"),
        weights=overrides.get("weights", None),
        phi=overrides.get("phi", None),
        phi_inv=overrides.get("phi_inv", None),
        fd_step=overrides.get("fd_step", 1e-6),
        gradient_backend=overrides.get("gradient_backend", "autodiff"),
    )


def plan_for(method, **overrides) -> Plan:
    """Build a Plan for the executor."""
    return Plan(
        method_resolved=method,
        method_declared=method,
        scale=overrides.get("scale", "response"),
        level=overrides.get("level", 0.95),
        ci=overrides.get("ci", None),
        B=overrides.get("B", 0),
        # Default n_sim mirrors the legacy default; delta diagnostics need a
        # positive draw count.
        n_sim=overrides.get("n_sim", 4000),
        seed=overrides.get("seed", 42),
    )


# ---------------------------------------------------------------------------
# Doctrine: no fallbacks on the new path
# ---------------------------------------------------------------------------


def test_no_fallback_attributes_delta(fit_logit):
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)
    plan = plan_for("delta")
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
    assert result["method"] == "delta"
    assert result["fallback_triggered"] is False
    assert result["fallback_reason"] is None


def test_no_fallback_attributes_high_kappa(separated_logit):
    """High κ is recorded, but the new engine never flips to simulation."""
    ctx = ctx_from_fit(separated_logit)
    # Predict at an extreme point to push curvature high.
    compiled = compile_query(
        QuerySpec(kind="predict", scenario={"atexog": {"x1": 3.0, "x2": -2.0}}),
        ctx,
    )
    plan = plan_for("delta")
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
    assert result["method"] == "delta"
    assert result["fallback_triggered"] is False
    assert result["fallback_reason"] is None
    assert result["kappa"] is not None
    assert np.all(np.isfinite(result["kappa"]))
    # Curvature should be material for this near-boundary estimand.
    assert float(np.max(np.asarray(result["kappa"]))) > 0.0


def test_no_fallback_attributes_simulation(fit_logit):
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)
    plan = plan_for("simulation", n_sim=200)
    facts = WiringFacts()
    banks = BankSet(plan.plan_hash, 0, plan.seed)
    result = execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )
    assert result["method"] == "simulation"
    assert result["fallback_triggered"] is False
    assert result["fallback_reason"] is None
    assert result["draws"] is not None


# ---------------------------------------------------------------------------
# Non-differentiable estimand under delta
# ---------------------------------------------------------------------------


def test_nondifferentiable_delta_refuses(fit_logit):
    """A non-differentiable compose under method='delta' is a CompileError."""
    ctx = ctx_from_fit(fit_logit)

    def compose(preds):
        # Python conditional on a tracer value → non-JAX-differentiable.
        if preds[0] > 0.5:
            return preds[0]
        return 0.0

    compiled = compile_query(
        QuerySpec(
            kind="evaluate",
            scenarios=({"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}}),
            compose=compose,
        ),
        ctx,
    )
    plan = plan_for("delta")
    facts = WiringFacts()
    banks = BankSet(plan.plan_hash, 0, plan.seed)
    config = build_inference_config(plan, ctx.adapter, facts, banks)

    with pytest.raises(CompileError, match='method="simulation"'):
        execute_query(
            compiled,
            adapter=ctx.adapter,
            plan=plan,
            wiring_facts=facts,
            banks=banks,
            frozen_cov=config.cov_params,
        )


def test_nondifferentiable_delta_no_warning(fit_logit):
    """The refusal must not emit a UserWarning or produce a simulation result."""
    ctx = ctx_from_fit(fit_logit)

    def compose(preds):
        if preds[0] > 0.5:
            return preds[0]
        return 0.0

    compiled = compile_query(
        QuerySpec(
            kind="evaluate",
            scenarios=({"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}}),
            compose=compose,
        ),
        ctx,
    )
    plan = plan_for("delta")
    facts = WiringFacts()
    banks = BankSet(plan.plan_hash, 0, plan.seed)
    config = build_inference_config(plan, ctx.adapter, facts, banks)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(CompileError):
            execute_query(
                compiled,
                adapter=ctx.adapter,
                plan=plan,
                wiring_facts=facts,
                banks=banks,
                frozen_cov=config.cov_params,
            )
        assert not any(isinstance(x.message, UserWarning) for x in w)


# ---------------------------------------------------------------------------
# κ recorded but not steering
# ---------------------------------------------------------------------------


def test_kappa_recorded_not_steering(separated_logit):
    """New engine keeps method='delta' and records κ; legacy flips."""
    ctx = ctx_from_fit(separated_logit)
    compiled = compile_query(
        QuerySpec(kind="predict", scenario={"atexog": {"x1": 2.5, "x2": -1.5}}),
        ctx,
    )

    # New path: method='delta', κ recorded.
    plan_new = plan_for("delta")
    banks_new = BankSet(plan_new.plan_hash, 0, plan_new.seed)
    facts = WiringFacts()
    config_new = build_inference_config(plan_new, ctx.adapter, facts, banks_new)
    result_new = execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan_new,
        wiring_facts=facts,
        banks=banks_new,
        frozen_cov=config_new.cov_params,
    )
    assert result_new["method"] == "delta"
    assert result_new["kappa"] is not None

    # Legacy path with default threshold flips.
    m = Margins(ctx.adapter.results, at="overall", method="delta")
    h_old, _, _ = m._build_prediction_estimand({"atexog": {"x1": 2.5, "x2": -1.5}}, None)

    legacy_config = InferenceConfig(
        method="delta",
        level=0.95,
        n_sim=4000,
        kappa_threshold=0.3,
        cov_params=ctx.adapter.covariance(),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        legacy_result = run_inference(h_old, ctx.adapter, legacy_config)
    assert legacy_result["method"] == "simulation"
    assert legacy_result["fallback_triggered"] is True
    # κ should be identical (both computed at the same beta/h).
    np.testing.assert_allclose(
        np.asarray(result_new["kappa"]),
        np.asarray(legacy_result["kappa"]),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Resampling declaration routing
# ---------------------------------------------------------------------------


def test_survey_design_drives_resampler(fit_logit, monkeypatch):
    """Survey design PSU/strata flow to the resampler."""
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)

    n = len(ctx.base_data)
    psu = np.repeat(np.arange(10), n // 10)
    strata = np.repeat(np.arange(2), n // 2)
    design = SurveyDesign(weights=np.ones(n), psu=psu, strata=strata)

    plan = plan_for("bootstrap", B=5)
    facts = WiringFacts(design=design)
    banks = BankSet(plan.plan_hash, 0, plan.seed)

    captured = {}
    original = __import__(
        "pymargins._inference._bootstrap", fromlist=["_generate_resample_indices"]
    )._generate_resample_indices

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pymargins._inference._bootstrap._generate_resample_indices", spy
    )

    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )

    assert captured["cluster_ids"] is design.psu
    assert captured["strata"] is design.strata


def test_cluster_declaration_drives_resampler(fit_logit, monkeypatch):
    """Cluster IDs declared at input flow to the resampler."""
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)

    n = len(ctx.base_data)
    cluster = np.repeat(np.arange(12), n // 12)

    plan = plan_for("bootstrap", B=5)
    facts = WiringFacts(cluster=cluster)
    banks = BankSet(plan.plan_hash, 0, plan.seed)

    captured = {}
    original = __import__(
        "pymargins._inference._bootstrap", fromlist=["_generate_resample_indices"]
    )._generate_resample_indices

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pymargins._inference._bootstrap._generate_resample_indices", spy
    )

    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )

    assert captured["cluster_ids"] is cluster
    assert captured["strata"] is None


def test_block_declaration_drives_resampler(fit_logit, monkeypatch):
    """Block bootstrap parameters flow to the resampler."""
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)

    plan = plan_for("bootstrap", B=5)
    facts = WiringFacts(block=5, block_type="circular")
    banks = BankSet(plan.plan_hash, 0, plan.seed)

    captured = {}
    original = __import__(
        "pymargins._inference._bootstrap", fromlist=["_generate_resample_indices"]
    )._generate_resample_indices

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pymargins._inference._bootstrap._generate_resample_indices", spy
    )

    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )

    assert captured["block_size"] == 5
    assert captured["block_type"] == "circular"


# ---------------------------------------------------------------------------
# Banks are replayed across queries
# ---------------------------------------------------------------------------


def test_banks_replayed_across_queries(fit_logit, monkeypatch):
    """Shared banks generate indices and draws only once across queries."""
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)

    plan_boot = plan_for("bootstrap", B=5)
    plan_sim = plan_for("simulation", n_sim=50)
    facts = WiringFacts()
    banks = BankSet(plan_boot.plan_hash, 0, plan_boot.seed)

    from pymargins._engine import _seeds

    idx_calls = []
    orig_idx = _seeds.legacy_resample_indices

    def spy_idx(*args, **kwargs):
        idx_calls.append((args, kwargs))
        return orig_idx(*args, **kwargs)

    draw_calls = []
    orig_draws = _seeds.legacy_sim_draws

    def spy_draws(*args, **kwargs):
        draw_calls.append((args, kwargs))
        return orig_draws(*args, **kwargs)

    monkeypatch.setattr(_seeds, "legacy_resample_indices", spy_idx)
    monkeypatch.setattr(_seeds, "legacy_sim_draws", spy_draws)

    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan_boot,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )
    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan_sim,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )
    execute_query(
        compiled,
        adapter=ctx.adapter,
        plan=plan_boot,
        wiring_facts=facts,
        banks=banks,
        frozen_cov=ctx.adapter.covariance(),
    )

    assert len(idx_calls) == 1
    assert len(draw_calls) == 1


# ---------------------------------------------------------------------------
# Replicate failure thresholds
# ---------------------------------------------------------------------------


def test_replicate_failure_warning(fit_logit, monkeypatch):
    """High replicate failure rate emits a SoundnessWarning."""
    ctx = ctx_from_fit(fit_logit)
    compiled = compile_query(QuerySpec(kind="predict"), ctx)

    plan = plan_for("bootstrap", B=200)
    facts = WiringFacts()
    banks = BankSet(plan.plan_hash, 0, plan.seed)

    def fake_bootstrap(h, adapter, config, estimand_metadata, *, h_factory=None):
        return {
            "estimate": np.array([0.5]),
            "std_error": np.array([0.1]),
            "conf_int_lower": np.array([0.3]),
            "conf_int_upper": np.array([0.7]),
            "method": "bootstrap",
            "level": config.level,
            "kappa": None,
            "delta_sim_disagreement": None,
            "fallback_triggered": False,
            "fallback_reason": None,
            "gradient": None,
            "draws": None,
            "draws_inf": None,
            "estimand_metadata": dict(estimand_metadata or {}),
            "ci_method": "percentile",
            "bootstrap_extras": None,
            "n_boot_effective": 189,
            "n_boot_failed": 11,
        }

    monkeypatch.setattr("pymargins._engine._execute._run_bootstrap", fake_bootstrap)

    with pytest.warns(SoundnessWarning, match="failure rate"):
        result = execute_query(
            compiled,
            adapter=ctx.adapter,
            plan=plan,
            wiring_facts=facts,
            banks=banks,
            frozen_cov=ctx.adapter.covariance(),
        )

    assert result["n_boot_failed"] == 11
    diagnostics = result["estimand_metadata"].get("diagnostics", [])
    assert any("failure rate" in str(d) for d in diagnostics)


def test_unknown_method_is_assertion():
    """An unreachable method raises AssertionError."""
    plan = plan_for("unsupported_method")
    plan = Plan(**{**plan.__dict__, "method_resolved": "unsupported_method"})
    banks = BankSet(plan.plan_hash, 0, plan.seed)
    with pytest.raises(AssertionError, match="Unreachable method"):
        execute_query(
            None,
            adapter=None,
            plan=plan,
            wiring_facts=WiringFacts(),
            banks=banks,
            frozen_cov=np.eye(1),
        )
