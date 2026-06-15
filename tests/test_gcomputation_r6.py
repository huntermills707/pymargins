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
