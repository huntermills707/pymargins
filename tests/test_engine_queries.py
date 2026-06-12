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


@pytest.fixture
def fit_logit_num():
    """Numeric-only logit fit for at=mean/typical (string columns break aggregation)."""
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n).astype(float),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()


@pytest.fixture
def df_survival():
    """Synthetic survival data for rmst sanity tests."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    hazard = np.exp(0.5 + 0.3 * df["x1"] - 0.2 * df["x2"])
    df["T"] = rng.exponential(1.0 / hazard)
    df["E"] = (rng.random(n) < 0.8).astype(int)
    return df


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
# Probe axes: at=mean/typical, weights, multi-variable dydx, grid+over
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("at", ["mean", "typical"])
def test_predict_h_matches_legacy_at_mean_typical(fit_logit_num, at):
    m = Margins(fit_logit_num, at=at, method="delta")
    h_old, labels_old, _ = _build_prediction_estimand(m, {}, None)
    cq = compile_query(
        QuerySpec(kind="predict", scenario={}),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


@pytest.mark.parametrize("at", ["mean", "typical"])
def test_dydx_h_matches_legacy_at_mean_typical(fit_logit_num, at):
    m = Margins(fit_logit_num, at=at, method="delta")
    h_old, labels_old, _ = _build_slope_estimand(m, {}, ["x1"], None)
    cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


def test_predict_h_matches_legacy_weights(fit_logit):
    rng = np.random.default_rng(4)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    h_old, labels_old, _ = _build_prediction_estimand(m, {}, None)
    cq = compile_query(
        QuerySpec(kind="predict", scenario={}),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


def test_dydx_h_matches_legacy_weights(fit_logit):
    rng = np.random.default_rng(5)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    h_old, labels_old, _ = _build_slope_estimand(m, {}, ["x1"], None)
    cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1",)),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels


def test_dydx_h_matches_legacy_multivar(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    h_old, labels_old, scen_old = _build_slope_estimand(
        m, {}, ["x1", "x2"], None
    )
    cq = compile_query(
        QuerySpec(kind="dydx", scenario={}, variables=("x1", "x2")),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels
    assert len(scen_old) == len(cq.scenarios)


def test_predict_h_matches_legacy_grid_plus_over(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenario = {"atexog": {"x1": [0.0, 1.0]}, "over": "g"}
    h_old, labels_old, scen_old = _build_prediction_estimand(m, scenario, None)
    cq = compile_query(
        QuerySpec(kind="predict", scenario=scenario),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels
    assert len(scen_old) == len(cq.scenarios)


def test_predict_weights_plus_over_weighted_group_means(fit_logit):
    """D16 (resolves D12): weights subset per over-group -> weighted group means."""
    import jax.numpy as jnp

    rng = np.random.default_rng(6)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    ctx = ctx_from_margins(m)
    beta = np.asarray(m.adapter.coefficients())
    cq = compile_query(QuerySpec(kind="predict", scenario={"over": "g"}), ctx)
    got = np.asarray(cq.h(beta))

    # Independent expectation: weighted mean of response-scale predictions
    # within each group, groups in sorted order.
    base = ctx.base_data
    levels = sorted(base["g"].unique())
    expected = []
    for g_val in levels:
        mask = (base["g"] == g_val).to_numpy()
        X_g = m.adapter.design_matrix_from_df(base[mask])
        mu = m.adapter.predict(jnp.asarray(beta), X_g, offset=None)
        w_g = jnp.asarray(w[mask])
        expected.append(float(jnp.sum(w_g * mu) / jnp.sum(w_g)))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-12)
    assert cq.labels == [f"g={v}" for v in levels]

    # Legacy still crashes (frozen until R7); divergence is intentional (D16).
    h_old, _, _ = _build_prediction_estimand(m, {"over": "g"}, None)
    with pytest.raises(TypeError):
        h_old(beta)


def test_dydx_weights_plus_over_matches_per_group_runs(fit_logit):
    """D16: over= with weights equals stacking per-group weighted runs.

    The no-over weighted dydx path is dual-run-anchored to legacy, so the
    per-group runs serve as the oracle for the over= vector.
    """
    rng = np.random.default_rng(9)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    ctx = ctx_from_margins(m)
    beta = np.asarray(m.adapter.coefficients())
    cq = compile_query(
        QuerySpec(kind="dydx", scenario={"over": "g"}, variables=("x1",)), ctx
    )
    got = np.asarray(cq.h(beta))

    base = ctx.base_data
    expected = []
    for g_val in sorted(base["g"].unique()):
        mask = (base["g"] == g_val).to_numpy()
        sub_ctx = QueryContext(
            adapter=ctx.adapter,
            base_data=base[mask],
            at=ctx.at,
            weights=w[mask],
            phi=ctx.phi,
            phi_inv=ctx.phi_inv,
            fd_step=ctx.fd_step,
            gradient_backend=ctx.gradient_backend,
        )
        sub = compile_query(
            QuerySpec(kind="dydx", scenario={}, variables=("x1",)), sub_ctx
        )
        expected.append(float(np.asarray(sub.h(beta))))
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-12)


def test_contrasts_2d_matrix_h_matches_legacy_numbers(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = [
        {"atexog": {"x1": 0.0}},
        {"atexog": {"x1": 1.0}},
        {"atexog": {"x1": 2.0}},
    ]
    C = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    weights_dict = {f"contrast[{i}]": C[i] for i in range(C.shape[0])}
    h_old = _build_contrast_estimand(m, scenarios, weights_dict)
    cq = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=C),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    # D18 (resolves D14): raw 2D matrices normalize to named contrasts.
    assert cq.labels == ["contrast[0]", "contrast[1]"]


def test_contrasts_list_of_lists_matches_2d(fit_logit):
    """D18: list-of-lists normalizes identically to a 2D matrix."""
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = (
        {"atexog": {"x1": 0.0}},
        {"atexog": {"x1": 1.0}},
        {"atexog": {"x1": 2.0}},
    )
    C = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    cq_mat = compile_query(
        QuerySpec(kind="contrasts", scenarios=scenarios, contrast_weights=C),
        ctx_from_margins(m),
    )
    cq_lol = compile_query(
        QuerySpec(
            kind="contrasts",
            scenarios=scenarios,
            contrast_weights=[[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
        ),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(
        np.asarray(cq_mat.h(beta)), np.asarray(cq_lol.h(beta))
    )
    assert cq_lol.labels == ["contrast[0]", "contrast[1]"]


def test_contrasts_weight_length_mismatch_raises(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = ({"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}})
    with pytest.raises(ValueError, match="has 3 weights but 2 scenarios"):
        compile_query(
            QuerySpec(
                kind="contrasts",
                scenarios=scenarios,
                contrast_weights=np.array([1.0, -1.0, 0.0]),
            ),
            ctx_from_margins(m),
        )


def test_contrasts_nonfinite_weights_raise(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    scenarios = ({"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}})
    with pytest.raises(ValueError, match="must be finite"):
        compile_query(
            QuerySpec(
                kind="contrasts",
                scenarios=scenarios,
                contrast_weights=np.array([np.nan, 1.0]),
            ),
            ctx_from_margins(m),
        )


def test_contrasts_non_dict_scenario_raises(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    with pytest.raises(TypeError, match="must be a dict"):
        compile_query(
            QuerySpec(
                kind="contrasts",
                scenarios=("notadict",),
                contrast_weights=np.array([1.0]),
            ),
            ctx_from_margins(m),
        )


def test_contrasts_weights_honored_in_aggregation(fit_logit):
    """D17 (resolves D13): contrast scenario aggregation uses declared weights."""
    import jax.numpy as jnp

    rng = np.random.default_rng(7)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m_w = Margins(fit_logit, at="overall", method="delta", weights=w)
    scenarios = [{"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}}]
    cw = np.array([1.0, -1.0])
    cqw = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=cw),
        ctx_from_margins(m_w),
    )
    beta = np.asarray(m_w.adapter.coefficients())
    got = float(np.asarray(cqw.h(beta)))

    # Independent expectation: weighted mean of response-scale predictions
    # per counterfactual scenario, then the linear combination.
    base = m_w._base_data
    w_jnp = jnp.asarray(w)
    vals_w, vals_u = [], []
    for x1v in (0.0, 1.0):
        df_s = base.copy()
        df_s["x1"] = x1v
        X_s = m_w.adapter.design_matrix_from_df(df_s)
        mu = m_w.adapter.predict(jnp.asarray(beta), X_s, offset=None)
        vals_w.append(float(jnp.sum(w_jnp * mu) / jnp.sum(w_jnp)))
        vals_u.append(float(jnp.mean(mu)))
    np.testing.assert_allclose(got, vals_w[0] - vals_w[1], rtol=1e-12)

    # Legacy combines the UNWEIGHTED per-scenario means — divergence is D17.
    h_legacy = _build_contrast_estimand(m_w, scenarios, jnp.asarray(cw))
    np.testing.assert_allclose(
        float(np.asarray(h_legacy(beta))), vals_u[0] - vals_u[1], rtol=1e-12
    )

    # Unweighted construction is unchanged: still matches legacy bit-for-bit.
    m0 = Margins(fit_logit, at="overall", method="delta")
    cq0 = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=cw),
        ctx_from_margins(m0),
    )
    h0 = _build_contrast_estimand(m0, scenarios, jnp.asarray(cw))
    np.testing.assert_array_equal(np.asarray(h0(beta)), np.asarray(cq0.h(beta)))


def test_evaluate_weights_honored_in_aggregation(fit_logit):
    """D17: evaluate scenario aggregation uses declared weights."""
    import jax.numpy as jnp

    rng = np.random.default_rng(8)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    scenarios = [
        {"atexog": {"g": "a"}},
        {"atexog": {"g": "b"}},
    ]

    def compose(p):
        return p[0] / p[1]

    cq = compile_query(
        QuerySpec(kind="evaluate", scenarios=tuple(scenarios), compose=compose),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    got = float(np.asarray(cq.h(beta)))

    base = m._base_data
    w_jnp = jnp.asarray(w)
    vals_w, vals_u = [], []
    for g_val in ("a", "b"):
        df_s = base.copy()
        df_s["g"] = g_val
        X_s = m.adapter.design_matrix_from_df(df_s)
        mu = m.adapter.predict(jnp.asarray(beta), X_s, offset=None)
        vals_w.append(float(jnp.sum(w_jnp * mu) / jnp.sum(w_jnp)))
        vals_u.append(float(jnp.mean(mu)))
    np.testing.assert_allclose(got, vals_w[0] / vals_w[1], rtol=1e-12)

    # Legacy composes the UNWEIGHTED per-scenario means — divergence is D17.
    h_legacy = _build_evaluate_estimand(m, scenarios, compose)
    np.testing.assert_allclose(
        float(np.asarray(h_legacy(beta))), vals_u[0] / vals_u[1], rtol=1e-12
    )


def test_contrasts_weights_at_mean_single_row_matches_legacy(fit_logit_num):
    """Single-row scenarios (at='mean') aggregate trivially; weighted ctx
    stays bit-identical to legacy."""
    import jax.numpy as jnp

    rng = np.random.default_rng(10)
    n = len(fit_logit_num.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit_num, at="mean", method="delta", weights=w)
    scenarios = [{"atexog": {"x1": 0.0}}, {"atexog": {"x1": 1.0}}]
    cw = np.array([1.0, -1.0])
    h_old = _build_contrast_estimand(m, scenarios, jnp.asarray(cw))
    cq = compile_query(
        QuerySpec(kind="contrasts", scenarios=tuple(scenarios), contrast_weights=cw),
        ctx_from_margins(m),
    )
    beta = np.asarray(m.adapter.coefficients())
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))


def test_weights_with_data_override_scenario_refuses(fit_logit):
    """D17: weights= with a scenario whose rows can't align refuses clearly."""
    rng = np.random.default_rng(11)
    n = len(fit_logit.model.endog)
    w = np.exp(rng.normal(0, 0.3, size=n))
    m = Margins(fit_logit, at="overall", method="delta", weights=w)
    custom = m._base_data.head(5)
    with pytest.raises(ValueError, match="Weighted aggregation requires"):
        compile_query(
            QuerySpec(
                kind="contrasts",
                scenarios=({"data": custom}, {"atexog": {"x1": 1.0}}),
                contrast_weights=np.array([1.0, -1.0]),
            ),
            ctx_from_margins(m),
        )


# ---------------------------------------------------------------------------
# Multi-estimand label warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        {"atexog": {"x1": [0.0, 1.0]}},
        {"over": "g"},
    ],
)
def test_multi_estimand_label_warns(fit_logit, scenario):
    m = Margins(fit_logit, at="overall", method="delta")
    with pytest.warns(UserWarning, match=r"label='mylabel' is ignored"):
        compile_query(
            QuerySpec(kind="predict", scenario=scenario, label="mylabel"),
            ctx_from_margins(m),
        )


def test_single_estimand_label_used(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    cq = compile_query(
        QuerySpec(kind="predict", scenario={}, label="mylabel"),
        ctx_from_margins(m),
    )
    assert cq.labels == ["mylabel"]


# ---------------------------------------------------------------------------
# rmst
# ---------------------------------------------------------------------------


def test_rmst_refuses_non_survival_adapter(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    ctx = ctx_from_margins(m)
    with pytest.raises(ValueError, match="per-scenario prediction time"):
        compile_query(QuerySpec(kind="rmst", horizon=1.0), ctx)


def test_rmst_survival_sanity(df_survival):
    pytest.importorskip("lifelines")
    from lifelines import CoxPHFitter

    from pymargins._adapters.lifelines_coxph_survival import (
        LifelinesCoxPHSurvivalAdapter,
    )

    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col="T", event_col="E", formula="x1 + x2")
    adapter = LifelinesCoxPHSurvivalAdapter(
        cph, training_data=df_survival, prediction_time=1.0
    )
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
    cq = compile_query(QuerySpec(kind="rmst", horizon=1.0, n_grid=20), ctx)
    beta = np.asarray(adapter.coefficients())
    val = np.asarray(cq.h(beta))
    assert val.shape == ()
    assert np.isfinite(val)
    assert cq.h_factory is not None


# ---------------------------------------------------------------------------
# InferenceConfig pass-through
# ---------------------------------------------------------------------------


def test_config_n_sim_n_boot_pass_through(fit_logit):
    from pymargins._graph._plan import Plan

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
    config = build_inference_config(plan, Margins(fit_logit).adapter, facts, None)
    assert config.n_sim == 0
    assert config.n_boot == 0

    plan2 = Plan(
        method_resolved="bootstrap",
        method_declared="bootstrap",
        scale="response",
        level=0.95,
        ci="percentile",
        B=1000,
        n_sim=4000,
        seed=7,
    )
    config2 = build_inference_config(plan2, Margins(fit_logit).adapter, facts, None)
    assert config2.n_sim == 4000
    assert config2.n_boot == 1000


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
