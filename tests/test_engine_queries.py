"""Query-layer tests.

Design \u00a74.2/\u00a74.8, req \u00a72. Added in 0.4.0 (R2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins._engine._queries import (
    QueryContext,
    QuerySpec,
    WiringFacts,
    build_inference_config,
    compile_query,
    resolve_scale,
)
from pymargins._soundness._predicates import CompileError

# Legacy builders are imported by tests only (I6 allows test reach).
from pymargins.margins import Margins
from pymargins.margins._estimands import (
    _build_contrast_estimand,
    _build_evaluate_estimand,
    _build_prediction_estimand,
    _build_slope_estimand,
)


def ctx_from_margins(m: Margins) -> QueryContext:
    """Build a QueryContext that mirrors a legacy Margins session."""
    return QueryContext(
        adapter=m.adapter,
        base_data=m._base_data,
        at=m.at,
        weights=m.weights,
        phi=m.phi,
        phi_inv=m.phi_inv,
        fd_step=m.fd_step,
        gradient_backend=m.gradient_backend,
    )


@pytest.fixture
def fit_logit():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n).astype(float),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.choice(["a", "b"], size=n),
        }
    )
    return smf.glm("y ~ x1 + x2 + C(g)", data=df, family=sm.families.Binomial()).fit()


# ---------------------------------------------------------------------------
# resolve_scale
# ---------------------------------------------------------------------------


def test_resolve_scale_identity():
    assert resolve_scale(None) == (None, None)
    assert resolve_scale("response") == (None, None)
    assert resolve_scale("identity") == (None, None)


def test_resolve_scale_log():
    phi, phi_inv = resolve_scale("log")
    assert phi is np.exp or phi.__name__ == "exp"
    assert phi_inv is np.log or phi_inv.__name__ == "log"


def test_resolve_scale_logit():
    phi, phi_inv = resolve_scale("logit")
    import jax

    assert phi is jax.scipy.special.expit
    assert phi_inv is jax.scipy.special.logit


def test_resolve_scale_probit():
    phi, phi_inv = resolve_scale("probit")
    import jax

    assert phi is jax.scipy.special.ndtr
    assert phi_inv is jax.scipy.special.ndtri


def test_resolve_scale_callable_pair():
    def f(x):
        return x + 1

    def g(x):
        return x - 1

    assert resolve_scale((f, g)) == (f, g)


def test_resolve_scale_unknown():
    with pytest.raises(CompileError, match="Unknown scale"):
        resolve_scale("lift")


# ---------------------------------------------------------------------------
# build_inference_config doctrine shape
# ---------------------------------------------------------------------------


def test_config_doctrine_shape(fit_logit):
    from pymargins._graph._plan import Plan

    m = Margins(fit_logit, at="overall", method="delta")
    plan = Plan(
        method_resolved="delta",
        method_declared="delta",
        scale="response",
        level=0.95,
        ci="wald",
        B=0,
        n_sim=0,
        seed=42,
    )
    facts = WiringFacts()
    config = build_inference_config(plan, m.adapter, facts, None)
    assert config.method == "delta"
    assert config.kappa_threshold == float("inf")
    assert config.diagnostics is True
    assert config.all_idx is None
    assert config.all_states is None
    assert config.all_states_failures is None
    assert config.sim_draws is None


def test_config_resolves_vcov_cluster(fit_logit):
    from pymargins._graph._plan import Plan

    plan = Plan(
        method_resolved="delta",
        method_declared="delta",
        scale="response",
        level=0.95,
        ci="wald",
    )
    g = np.repeat(np.arange(10), 8)
    facts = WiringFacts(cluster=g)
    config = build_inference_config(plan, Margins(fit_logit).adapter, facts, None)
    assert config.cluster is g


# ---------------------------------------------------------------------------
# predict query matches legacy builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        {},
        {"atexog": {"g": "a"}},
        {"over": "g"},
        {"atexog": {"g": ["a", "b"]}},
    ],
)
def test_predict_h_matches_legacy(fit_logit, scenario):
    m = Margins(fit_logit, at="overall", method="delta")
    h_old, labels_old, scen_old = _build_prediction_estimand(m, scenario, None)
    cq = compile_query(
        QuerySpec(kind="predict", scenario=scenario),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels
    assert len(scen_old) == len(cq.scenarios)


def test_predict_h_matches_legacy_log_scale(fit_logit):
    import jax.numpy as jnp

    m = Margins(
        fit_logit, at="overall", method="delta", phi=jnp.exp, phi_inv=jnp.log
    )
    scenario = {"atexog": {"g": "a"}}
    h_old, labels_old, _ = _build_prediction_estimand(m, scenario, None)
    cq = compile_query(
        QuerySpec(kind="predict", scenario=scenario),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


def test_predict_h_matches_legacy_with_transform(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")

    def transform(mu):
        return mu ** 2

    scenario = {}
    h_old, labels_old, _ = _build_prediction_estimand(m, scenario, transform)
    cq = compile_query(
        QuerySpec(kind="predict", scenario=scenario, transform=transform),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


# ---------------------------------------------------------------------------
# dydx query matches legacy builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        {},
        {"atexog": {"g": "a"}},
        {"over": "g"},
    ],
)
def test_dydx_h_matches_legacy(fit_logit, scenario):
    m = Margins(fit_logit, at="overall", method="delta")
    h_old, labels_old, scen_old = _build_slope_estimand(m, scenario, ["x1"], None)
    cq = compile_query(
        QuerySpec(kind="dydx", scenario=scenario, variables=("x1",)),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels
    assert len(scen_old) == len(cq.scenarios)


def test_dydx_h_matches_legacy_log_scale(fit_logit):
    import jax.numpy as jnp

    m = Margins(
        fit_logit, at="overall", method="delta", phi=jnp.exp, phi_inv=jnp.log
    )
    scenario = {}
    h_old, labels_old, _ = _build_slope_estimand(m, scenario, ["x1"], None)
    cq = compile_query(
        QuerySpec(kind="dydx", scenario=scenario, variables=("x1",)),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


# ---------------------------------------------------------------------------
# contrasts query matches legacy builder
# ---------------------------------------------------------------------------


def test_contrasts_h_matches_legacy(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = [
        {"atexog": {"g": "a"}},
        {"atexog": {"g": "b"}},
    ]
    weights = np.array([1.0, -1.0])
    h_old = _build_contrast_estimand(m, scenarios, weights)
    cq = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=weights),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))


def test_contrasts_dict_h_matches_legacy(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = [
        {"atexog": {"g": "a"}},
        {"atexog": {"g": "b"}},
    ]
    weights = {"risk_diff": np.array([1.0, -1.0])}
    h_old = _build_contrast_estimand(m, scenarios, weights)
    cq = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=weights),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert cq.labels == ["risk_diff"]


# ---------------------------------------------------------------------------
# evaluate query matches legacy builder
# ---------------------------------------------------------------------------


def test_evaluate_h_matches_legacy(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = [
        {"atexog": {"g": "a"}},
        {"atexog": {"g": "b"}},
    ]
    def compose(p):
        return p[0] / p[1]

    h_old = _build_evaluate_estimand(m, scenarios, compose)
    cq = compile_query(
        QuerySpec(kind="evaluate", scenarios=tuple(scenarios), compose=compose),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))


# ---------------------------------------------------------------------------
# Elasticity / WTP composed queries
# ---------------------------------------------------------------------------


def test_eyex_h_matches_composed_slope_and_prediction(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    cq = compile_query(
        QuerySpec(kind="eyex", scenario={}, variables=("x1",)),
        ctx,
    )
    slope_cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx,
    )
    pred_cq = compile_query(
        QuerySpec(kind="predict", scenario={}),
        ctx,
    )
    x_bar = float(np.asarray(ctx.base_data["x1"]).mean())
    beta = np.asarray(m.adapter.coefficients())
    expected = slope_cq.h(beta) * x_bar / pred_cq.h(beta)
    np.testing.assert_allclose(np.asarray(cq.h(beta)), np.asarray(expected), rtol=1e-12)


def test_eydx_h_matches_composed_slope_and_prediction(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    cq = compile_query(
        QuerySpec(kind="eydx", scenario={}, variables=("x1",)),
        ctx,
    )
    slope_cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx,
    )
    pred_cq = compile_query(
        QuerySpec(kind="predict", scenario={}),
        ctx,
    )
    beta = np.asarray(m.adapter.coefficients())
    expected = slope_cq.h(beta) / pred_cq.h(beta)
    np.testing.assert_allclose(np.asarray(cq.h(beta)), np.asarray(expected), rtol=1e-12)


def test_dyex_h_matches_composed_slope_and_prediction(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    cq = compile_query(
        QuerySpec(kind="dyex", scenario={}, variables=("x1",)),
        ctx,
    )
    slope_cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx,
    )
    x_bar = float(np.asarray(ctx.base_data["x1"]).mean())
    beta = np.asarray(m.adapter.coefficients())
    expected = slope_cq.h(beta) * x_bar
    np.testing.assert_array_equal(np.asarray(cq.h(beta)), np.asarray(expected))


def test_wtp_h_matches_composed_slopes(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    cq = compile_query(
        QuerySpec(kind="wtp", scenario={}, variables=("x1", "x2")),
        ctx,
    )
    attr_cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx,
    )
    price_cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x2",)),
        ctx,
    )
    beta = np.asarray(m.adapter.coefficients())
    expected = -attr_cq.h(beta) / price_cq.h(beta)
    np.testing.assert_allclose(np.asarray(cq.h(beta)), np.asarray(expected), rtol=1e-12)


# ---------------------------------------------------------------------------
# h_factory rebuilds on refit adapter
# ---------------------------------------------------------------------------


def test_h_factory_rebuilds_prediction_on_refit_adapter(fit_logit):
    m = Margins(fit_logit, at="overall", method="bootstrap", n_boot=2, rng_seed=1)
    ctx = ctx_from_margins(m)
    spec = QuerySpec(kind="predict", scenario={})
    cq = compile_query(spec, ctx)
    assert cq.h_factory is not None

    # Refit on the same data; point estimates should be identical.
    refit_adapter = m.adapter.refit(m.adapter.training_data, index=np.arange(len(m.adapter.training_data)))
    h_refit = cq.h_factory(refit_adapter)
    beta = np.asarray(m.adapter.coefficients())
    beta_refit = np.asarray(refit_adapter.coefficients())
    np.testing.assert_allclose(
        np.asarray(cq.h(beta)),
        np.asarray(h_refit(beta_refit)),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Refusal / error paths
# ---------------------------------------------------------------------------


def test_unknown_query_kind_raises():
    from pymargins._adapters import auto_detect_adapter

    rng = np.random.default_rng(1)
    n = 20
    df = pd.DataFrame({"y": rng.binomial(1, 0.5, size=n).astype(float), "x": rng.normal(size=n)})
    fit = smf.ols("y ~ x", data=df).fit()
    adapter = auto_detect_adapter(fit)
    ctx = QueryContext(
        adapter=adapter,
        base_data=adapter.training_data,
        at="overall",
        weights=None,
        phi=None,
        phi_inv=None,
        fd_step=1e-6,
        gradient_backend="autodiff",
    )
    with pytest.raises(CompileError, match="Unknown query kind"):
        compile_query(QuerySpec(kind="unknown"), ctx)


def test_dydx_requires_variables(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    with pytest.raises(ValueError, match="requires at least one variable"):
        compile_query(QuerySpec(kind="dydx"), ctx)
