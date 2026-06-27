"""R6 — GComputation on the new engine."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import steps
from pymargins._graph._compile import CompileError
from pymargins._result._graphresult import GraphResult
from pymargins.estimators._base import GComputation


def make_df(seed: int = 42, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )


def test_spec_form_ols_outcome():
    d = make_df()
    est = GComputation(steps.input(d), outcome="y ~ x + z", method="delta")
    assert est.plan.method_resolved == "delta"
    r = est.predict()
    assert isinstance(r, GraphResult)
    assert r.estimate.shape == ()


def test_spec_form_logit_outcome():
    d = make_df()
    est = GComputation(
        steps.input(d),
        outcome=("y ~ x + z", sm.families.Binomial()),
        method="delta",
        scale="response",
    )
    r = est.predict()
    assert isinstance(r, GraphResult)


def test_positional_model_implicit_input():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r = est.predict()
    assert isinstance(r, GraphResult)


def test_positional_node_without_outcome_refuses():
    d = make_df()
    with pytest.raises(CompileError, match="requires outcome="):
        GComputation(steps.input(d))


def test_predict_dydx_contrasts_evaluate_return_graphresult():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    assert isinstance(est.predict(), GraphResult)
    assert isinstance(est.dydx("x"), GraphResult)
    assert isinstance(
        est.contrasts(
            scenarios=[{"atexog": {"x": 0}}, {"atexog": {"x": 1}}],
            contrasts=np.array([[-1, 1]]),
        ),
        GraphResult,
    )
    assert isinstance(
        est.evaluate(
            scenarios=[{"atexog": {"x": 0}}, {"atexog": {"x": 1}}],
            compose=lambda x: x[1] - x[0],
        ),
        GraphResult,
    )


def test_callable_scale_pair():
    d = make_df()
    fit = smf.glm("y ~ x + z", data=d, family=sm.families.Binomial()).fit()
    phi = lambda t: 1 / (1 + jnp.exp(-t))  # noqa: E731
    phi_inv = lambda p: jnp.log(p / (1 - p))  # noqa: E731
    est = GComputation(fit, method="delta", scale=(phi, phi_inv))
    r = est.predict()
    assert isinstance(r, GraphResult)


def test_weights_known_weights():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    w = np.random.default_rng(0).uniform(0.5, 1.5, size=len(d))
    est = GComputation(fit, method="delta", weights=w)
    r = est.predict()
    assert isinstance(r, GraphResult)
    assert est.plan.weights_fingerprint is not None


def test_cluster_declaration_routed():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    cluster = np.repeat(np.arange(20), 5)
    est = GComputation(
        steps.input(d, cluster=cluster),
        outcome=fit,
        method="delta",
        vcov="cluster",
    )
    assert est.plan.method_resolved == "delta"
    r = est.predict()
    assert isinstance(r, GraphResult)


def test_survey_design_routed():
    from pymargins.survey import SurveyDesign

    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    sd = SurveyDesign(weights=np.ones(len(d)))
    est = GComputation(steps.input(d, design=sd), outcome=fit, method="delta")
    r = est.predict()
    assert isinstance(r, GraphResult)


def test_method_auto_records_kappa_reason():
    rng = np.random.default_rng(4)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 2.0 + 5.0 * x1 - 3.0 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    d = pd.DataFrame(
        {
            "y": rng.binomial(1, p).astype(float),
            "x1": x1,
            "x2": x2,
        }
    )
    fit = smf.glm("y ~ x1 + x2", data=d, family=sm.families.Binomial()).fit()
    est = GComputation(fit, method="auto")
    assert est.plan.method_resolved == "simulation"
    desc = est.plan.describe()
    assert "κ" in desc or "kappa" in desc


def test_strict_and_diagnostics_kwargs_rejected():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    with pytest.raises(TypeError):
        GComputation(fit, strict=True)
    with pytest.raises(TypeError):
        GComputation(fit, diagnostics=False)


def test_wtp_basic():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r = est.wtp("x", "z")
    assert isinstance(r, GraphResult)


def test_eyex_eydx_dyex_return_graphresult():
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    assert isinstance(est.eyex("x"), GraphResult)
    assert isinstance(est.eydx("x"), GraphResult)
    assert isinstance(est.dyex("x"), GraphResult)


def test_eyex_matches_manual_ratio():
    """eyex ≈ (dydx * x_bar) / predict at the mean."""
    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r_eyex = est.eyex("x")
    r_dydx = est.dydx("x")
    r_pred = est.predict()
    x_bar = float(d["x"].mean())
    manual = r_dydx.estimate * x_bar / r_pred.estimate
    np.testing.assert_allclose(r_eyex.estimate, manual, rtol=1e-5)


def test_eyex_se_against_analytic_delta():
    """Delta-method SE for OLS eyex matches the closed-form gradient."""
    rng = np.random.default_rng(11)
    n = 250
    d = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r = est.eyex("x")

    beta = fit.params.values
    Sigma = fit.cov_params().values
    x_idx = list(fit.params.index).index("x")
    Xbar = fit.model.exog.mean(axis=0)
    ybar = float(Xbar @ beta)
    x_bar = float(d["x"].mean())

    e_x = np.zeros(len(beta))
    e_x[x_idx] = 1.0
    grad = x_bar * (e_x * ybar - beta[x_idx] * Xbar) / ybar**2
    se_analytic = float(np.sqrt(grad @ Sigma @ grad))

    np.testing.assert_allclose(r.std_error, se_analytic, rtol=1e-5)


def test_ndarray_vcov_demotes_psi_h():
    """A user-supplied ndarray Σ̂ must not emit tier-1 influence (ψ^h)."""
    d = make_df(n=200)
    fit = smf.ols("y ~ x + z", data=d).fit()
    user_cov = np.asarray(fit.cov_params())

    # Baseline: adapter-derived Σ̂ produces a real tier-1 influence function.
    est_base = GComputation(fit, method="delta")
    r_base = est_base.predict()
    assert isinstance(r_base, GraphResult)
    assert r_base.psi_h is not None
    assert r_base.psi_h.ndim == 1
    assert len(r_base.psi_h) == len(d)

    # User-supplied Σ̂: tier-1 influence must be demoted.
    est = GComputation(fit, method="delta", vcov=user_cov)
    r = est.predict()
    assert isinstance(r, GraphResult)
    assert r.psi_h is None
    # The Plan must not retain the raw array (§4 hash / JSON hygiene).
    assert not isinstance(est.plan.vcov, np.ndarray)
    assert isinstance(est.plan.vcov, dict)
    assert est.plan.vcov.get("kind") == "user_ndarray"
    assert "fingerprint" in est.plan.vcov


def test_psi_h_includes_bread_scale_equivariance():
    """ψ^h must carry the bread Σ̂: scaling y by c scales ψ^h by exactly c.

    The per-observation influence of β̂ is ψ^β = Σ̂·score; dropping Σ̂
    mis-scales ψ^h by the covariance (≈ σ̂² for OLS), which the shape-only
    demotion check cannot see. Regression guard for D19.
    """
    rng = np.random.default_rng(7)
    n = 300
    base = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    base["y0"] = 1.0 + 2.0 * base["x"] - base["z"] + rng.normal(size=n)

    def psi_h(scale: float) -> np.ndarray:
        d = base.assign(y=scale * base["y0"])
        fit = smf.ols("y ~ x + z", data=d).fit()
        return np.asarray(GComputation(fit, method="delta").predict().psi_h)

    psi1 = psi_h(1.0)
    np.testing.assert_allclose(psi_h(10.0), 10.0 * psi1, rtol=1e-8)

    # ψ^h reproduces the (robust) SE: sqrt(Σ ψ²) ≈ std_error. Without the bread
    # this holds only when σ̂ ≈ 1; the equivariance check pins the scaling,
    # this pins the magnitude.
    fit = smf.ols("y ~ x + z", data=base.assign(y=base["y0"])).fit()
    r = GComputation(fit, method="delta").predict()
    np.testing.assert_allclose(
        np.sqrt((np.asarray(r.psi_h) ** 2).sum()), r.std_error, rtol=0.05
    )


def test_adjust_on_graphresult():
    """adjust() is duck-typed on result.test(); it must accept GraphResult."""
    from pymargins._result._test import adjust

    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r1 = est.dydx("x")
    r2 = est.dydx("z")
    adj = adjust([r1, r2], method="bonferroni")
    assert len(adj.p_adj) == 2
    assert np.all(adj.p_adj >= adj.p_raw)


def test_pool_imputations_on_graphresult():
    """Rubin pooling must accept a list of GraphResult objects."""
    from pymargins._result._pooling import pool_imputations

    d = make_df()
    fit = smf.ols("y ~ x + z", data=d).fit()
    est = GComputation(fit, method="delta")
    r1 = est.predict()
    r2 = est.predict()
    pooled = pool_imputations([r1, r2])
    assert isinstance(pooled, GraphResult)
    assert pooled.method == "pooled"
    assert pooled.imputation_diagnostic is not None
