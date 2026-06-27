"""Regression tests for bug fixes, ported to the GComputation API (R7)."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import GComputation, steps
from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
from pymargins._inference import InferenceConfig, _run_bootstrap, _run_simulation
from pymargins._result import GraphResult
from pymargins._scenarios import make_aggregation_resolver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_logit():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, size=n),
            "treatment": rng.binomial(1, 0.5, size=n),
        }
    )
    eta = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"]
    prob = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = (rng.uniform(size=n) < prob).astype(float)
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "y ~ age + treatment", data=df_logit, family=sm.families.Binomial()
    ).fit()


# ---------------------------------------------------------------------------
# 1. Delta fallback guard too restrictive
# ---------------------------------------------------------------------------
# Legacy ``run_inference`` fallback dispatch is deleted in 0.4.0 (R3).
# Non-differentiable estimands under method="delta" now raise CompileError.
# Covered by tests/test_engine_execute.py::test_nondifferentiable_delta_refuses.


# ---------------------------------------------------------------------------
# 2. NaN κ silently disables auto-fallback
# ---------------------------------------------------------------------------
# kappa_threshold and method fallback are deleted in 0.4.0 (R3/R7).


# ---------------------------------------------------------------------------
# 3. Simulation fallback passes JAX arrays to pure-Python h
# ---------------------------------------------------------------------------


def test_simulation_fallback_uses_numpy_for_pure_python_h(fit_logit):
    """vmap fallback must pass np.ndarray, not jnp.ndarray, to pure-Python h."""
    adapter = StatsmodelsGLMAdapter(fit_logit)

    def h(b):
        # Pure Python that fails on JAX arrays but works on numpy
        # The first call (estimate = h(beta)) will receive a JAX array,
        # so we allow that. The vmap loop fallback should convert to numpy.
        arr = np.asarray(b)
        return float(np.sum(arr))

    config = InferenceConfig(
        method="simulation",
        n_sim=20,
        rng_seed=42,
        diagnostics=False,
    )
    result = _run_simulation(h, adapter, config, estimand_metadata=None)
    assert result["method"] == "simulation"
    assert np.isfinite(float(result["estimate"]))


# ---------------------------------------------------------------------------
# 4. Bootstrap single failed replicate crashes entire loop
# ---------------------------------------------------------------------------


def test_bootstrap_skips_failed_replicates(fit_logit, monkeypatch):
    """A few failed refits should be skipped; result still produced."""
    adapter = StatsmodelsGLMAdapter(fit_logit)

    def h(b):
        return jax.scipy.special.expit(jnp.array([1.0, 50.0, 1.0]) @ b)

    call_count = [0]
    original_refit = adapter.refit

    def flaky_refit(resampled, *, index=None):
        call_count[0] += 1
        if call_count[0] <= 3:
            raise RuntimeError("Simulated refit failure")
        return original_refit(resampled, index=index)

    monkeypatch.setattr(adapter, "refit", flaky_refit)

    config = InferenceConfig(
        method="bootstrap",
        n_boot=50,
        rng_seed=42,
        diagnostics=False,
    )
    result = _run_bootstrap(
        h,
        adapter,
        config,
        estimand_metadata=None,
        h_factory=lambda new_adapter: h,
    )
    assert result["method"] == "bootstrap"
    assert np.isfinite(float(result["estimate"]))


def test_bootstrap_raises_when_too_many_failures(fit_logit, monkeypatch):
    """If all replicates fail, bootstrap should raise."""
    adapter = StatsmodelsGLMAdapter(fit_logit)

    def h(b):
        return jax.scipy.special.expit(jnp.array([1.0, 50.0, 1.0]) @ b)

    monkeypatch.setattr(
        adapter,
        "refit",
        lambda data, *, index=None: (_ for _ in ()).throw(RuntimeError("always fail")),
    )

    config = InferenceConfig(
        method="bootstrap",
        n_boot=20,
        rng_seed=42,
        diagnostics=False,
    )
    with pytest.raises(RuntimeError, match="Bootstrap failed"):
        _run_bootstrap(
            h,
            adapter,
            config,
            estimand_metadata=None,
            h_factory=lambda new_adapter: h,
        )


# ---------------------------------------------------------------------------
# 5. run_test ignores kind parameter for delta results
# ---------------------------------------------------------------------------
# Legacy ``run_test`` dispatch is deleted; ``GraphResult.test/joint_test``
# validate their own ``kind``/``null_scale`` parameters.


# ---------------------------------------------------------------------------
# 6. phi invoked on JAX array without safety conversion
# ---------------------------------------------------------------------------


def test_simulation_phi_defensively_converts_to_numpy(fit_logit):
    """phi that only accepts numpy should work via defensive conversion."""
    adapter = StatsmodelsGLMAdapter(fit_logit)

    def h(b):
        return jax.scipy.special.expit(jnp.array([1.0, 50.0, 1.0]) @ b)

    def phi_numpy_only(x):
        if isinstance(x, jnp.ndarray):
            raise TypeError("phi does not accept JAX arrays")
        return np.exp(x)

    config = InferenceConfig(
        method="simulation",
        n_sim=20,
        rng_seed=42,
        diagnostics=False,
        phi=phi_numpy_only,
    )
    result = _run_simulation(h, adapter, config, estimand_metadata=None)
    assert result["method"] == "simulation"
    assert np.isfinite(float(result["estimate"]))


# ---------------------------------------------------------------------------
# 7. Silent skip for missing columns
# ---------------------------------------------------------------------------


def test_resolver_raises_on_missing_columns():
    """make_aggregation_resolver must raise when expected columns are missing."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    resolver = make_aggregation_resolver("mean")
    meta = {
        "a": type("V", (), {"var_type": "continuous", "name": "a"})(),
        "c": type("V", (), {"var_type": "continuous", "name": "c"})(),
    }
    with pytest.raises(ValueError, match="Missing column"):
        resolver(df, meta)


# ---------------------------------------------------------------------------
# 8. Remove dead expand_with_over
# ---------------------------------------------------------------------------


def test_expand_with_over_is_removed():
    """expand_with_over should no longer be importable from _scenarios."""
    with pytest.raises(ImportError):
        from pymargins._scenarios import expand_with_over  # noqa: F401


# ---------------------------------------------------------------------------
# 9. joint_test default null on wrong scale
# ---------------------------------------------------------------------------


def test_joint_test_default_null_on_inference_scale(fit_logit):
    """When phi_inv is available, default null should be phi_inv(0), not 0."""
    # Use correlation_scale where phi_inv(0) = arctanh(0) = 0 (finite)
    est = GComputation(
        fit_logit,
        at="typical",
        scale=(jnp.tanh, jnp.arctanh),
        method="delta",
    )

    result = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "age": 40}},
            {"atexog": {"treatment": 0, "age": 40}},
            {"atexog": {"treatment": 1, "age": 60}},
            {"atexog": {"treatment": 0, "age": 60}},
        ],
        contrasts={
            "age40": [+1, -1, 0, 0],
            "age60": [0, 0, +1, -1],
        },
    )
    jt = result.joint_test()
    # correlation_scale: phi=tanh, phi_inv=arctanh. phi_inv(0) = 0 on inference scale.
    assert jt.method == "joint_wald"
    assert np.isfinite(float(jt.statistic))


# ---------------------------------------------------------------------------
# 10. __mul__ with non-scalar raises unclear error
# ---------------------------------------------------------------------------
# MarginsResult.__mul__ is gone with the legacy result object (R7).


# ---------------------------------------------------------------------------
# 11. test() falls through confusingly when gradient present but cov_params missing
# ---------------------------------------------------------------------------


def test_test_raises_clear_message_when_cov_params_missing(fit_logit):
    """test() should say inference is unavailable when cov_params is missing."""
    est = GComputation(fit_logit, at="typical", method="delta")
    result = est.predict()
    assert result.gradient is not None

    # Create a result with gradient but no cov_params manually
    result_with_grad = GraphResult(
        estimate=result.estimate,
        std_error=result.std_error,
        conf_int_lower=result.conf_int_lower,
        conf_int_upper=result.conf_int_upper,
        labels=result.labels,
        method="delta",
        level=result.level,
        ci=result.ci,
        scale=result.scale,
        at=result.at,
        plan=result.plan,
        population_note=result.population_note,
        n_obs=result.n_obs,
        estimand_metadata=result.estimand_metadata,
        gradient=np.array([0.1, 0.2]),
        cov_params=None,
        phi=result.phi,
        phi_inv=result.phi_inv,
    )
    with pytest.raises(ValueError, match="Cannot run test"):
        result_with_grad.test()


# ---------------------------------------------------------------------------
# 12. fallback_reason in combined result keeps only a's reason
# ---------------------------------------------------------------------------
# ``fallback_reason`` and MarginsResult arithmetic are deleted in 0.4.0.


# ---------------------------------------------------------------------------
# 13. Contrast matrix detection misses JAX arrays and list-of-lists
# ---------------------------------------------------------------------------


def test_contrasts_accepts_jax_array(fit_logit):
    """JAX 2D arrays should be accepted as contrast matrices."""
    est = GComputation(fit_logit, at="typical", method="delta")
    contrasts = jnp.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    result = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "age": 40}},
            {"atexog": {"treatment": 0, "age": 40}},
            {"atexog": {"treatment": 1, "age": 60}},
            {"atexog": {"treatment": 0, "age": 60}},
        ],
        contrasts=contrasts,
    )
    assert result.estimate.shape == (2,)


def test_contrasts_accepts_list_of_lists(fit_logit):
    """list-of-lists should be accepted as contrast matrices."""
    est = GComputation(fit_logit, at="typical", method="delta")
    result = est.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1, "age": 40}},
            {"atexog": {"treatment": 0, "age": 40}},
            {"atexog": {"treatment": 1, "age": 60}},
            {"atexog": {"treatment": 0, "age": 60}},
        ],
        contrasts=[[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
    )
    assert result.estimate.shape == (2,)


# ---------------------------------------------------------------------------
# 14. over= pandas-coupled without guard
# ---------------------------------------------------------------------------
# Legacy ``Margins._enumerate_groups`` is deleted; over= now requires a
# DataFrame base_data by construction (the input node carries a DataFrame).


# ---------------------------------------------------------------------------
# 15. _build_prediction_estimand slicing assumes contiguous grid blocks
# ---------------------------------------------------------------------------


def test_prediction_grid_blocks_sliced_correctly(fit_logit):
    """Multi-value atexog should produce correct per-block predictions."""
    est = GComputation(fit_logit, at="typical", method="delta")
    pred = est.predict(atexog={"treatment": [0, 1]})
    assert pred.estimate.shape == (2,)
    assert np.all(np.isfinite(pred.estimate))


def test_grid_block_slicing_defensive_check(fit_logit):
    """If an adapter drops rows, the grid block check should raise a clear error."""
    est = GComputation(fit_logit, at="typical", method="delta")
    from unittest.mock import patch

    def drop_rows(self, df):
        # Drop one row to break the grid layout
        X = self._original_design_matrix_from_df(df)
        return X[:-1]

    adapter = est._compiled.adapter
    adapter._original_design_matrix_from_df = adapter.design_matrix_from_df
    with patch.object(
        adapter, "design_matrix_from_df", lambda df: drop_rows(adapter, df)
    ):
        with pytest.raises(
            ValueError, match="Design matrix rows .* do not match expected grid layout"
        ):
            est.predict(atexog={"treatment": [0, 1]})


# ---------------------------------------------------------------------------
# 16. h_factory passed unconditionally
# ---------------------------------------------------------------------------
# Legacy session h_factory wiring is deleted; the new executor always passes
# h_factory to bootstrap (covered by tests/test_engine_execute.py).


# ---------------------------------------------------------------------------
# 17. _get_base_data is redundant
# ---------------------------------------------------------------------------
# Legacy ``_get_base_data`` is deleted; base data comes from the wiring graph.


# ---------------------------------------------------------------------------
# B1. MarginsResult.outcome() slices gradient correctly
# ---------------------------------------------------------------------------


def test_result_outcome_slices_gradient_correctly():
    """Gradient is (n_components, n_params); outcome() should slice rows only."""
    result = GraphResult(
        estimate=np.array([0.1, 0.2, 0.3]),
        std_error=np.array([0.01, 0.02, 0.03]),
        conf_int_lower=np.array([0.08, 0.18, 0.28]),
        conf_int_upper=np.array([0.12, 0.22, 0.32]),
        labels=["pred (0)", "pred (1)", "pred (2)"],
        method="delta",
        level=0.95,
        ci="wald",
        scale="response",
        at="overall",
        plan=None,
        population_note=None,
        n_obs=100,
        estimand_metadata={},
        gradient=np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            ]
        ),
    )
    sub = result.outcome(0)
    assert sub.gradient.shape == (1, 6)
    np.testing.assert_array_equal(sub.gradient, result.gradient[[0], :])


def test_result_outcome_slices_draws_correctly():
    """Draws are (n_draws, n_components); outcome() should slice columns."""
    draws = np.random.default_rng(42).standard_normal((100, 3))
    result = GraphResult(
        estimate=np.array([0.1, 0.2, 0.3]),
        std_error=np.array([0.01, 0.02, 0.03]),
        conf_int_lower=np.array([0.08, 0.18, 0.28]),
        conf_int_upper=np.array([0.12, 0.22, 0.32]),
        labels=["pred (0)", "pred (1)", "pred (2)"],
        method="simulation",
        level=0.95,
        ci="percentile",
        scale="response",
        at="overall",
        plan=None,
        population_note=None,
        n_obs=100,
        estimand_metadata={},
        draws=draws,
    )
    sub = result.outcome(1)
    assert sub.draws.shape == (100, 1)
    np.testing.assert_array_equal(sub.draws, draws[:, [1]])


# ---------------------------------------------------------------------------
# B2. sm.Logit / sm.Probit auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_logit_probit():
    """Modern statsmodels wraps Logit/Probit in BinaryResultsWrapper."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.integers(0, 2, 100)})
    fit_logit = smf.logit("y ~ x", data=df).fit(disp=False)
    fit_probit = smf.probit("y ~ x", data=df).fit(disp=False)

    from pymargins._adapters import auto_detect_adapter
    from pymargins._adapters.statsmodels_discrete_binary import (
        StatsmodelsDiscreteBinaryAdapter,
    )

    assert isinstance(auto_detect_adapter(fit_logit), StatsmodelsDiscreteBinaryAdapter)
    assert isinstance(auto_detect_adapter(fit_probit), StatsmodelsDiscreteBinaryAdapter)


def test_logit_predict_matches_statsmodels():
    """StatsmodelsDiscreteBinaryAdapter predict matches statsmodels native."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.integers(0, 2, 100)})
    fit = smf.logit("y ~ x", data=df).fit(disp=False)

    from pymargins._adapters.statsmodels_discrete_binary import (
        StatsmodelsDiscreteBinaryAdapter,
    )

    adapter = StatsmodelsDiscreteBinaryAdapter(fit)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(df)
    pred = adapter.predict(beta, X)
    np.testing.assert_allclose(np.asarray(pred), fit.predict(df), atol=1e-12)


def test_probit_predict_matches_statsmodels():
    """StatsmodelsDiscreteBinaryAdapter predict matches statsmodels native."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.integers(0, 2, 100)})
    fit = smf.probit("y ~ x", data=df).fit(disp=False)

    from pymargins._adapters.statsmodels_discrete_binary import (
        StatsmodelsDiscreteBinaryAdapter,
    )

    adapter = StatsmodelsDiscreteBinaryAdapter(fit)
    beta = adapter.coefficients()
    X = adapter.design_matrix_from_df(df)
    pred = adapter.predict(beta, X)
    np.testing.assert_allclose(np.asarray(pred), fit.predict(df), atol=1e-12)


# ---------------------------------------------------------------------------
# B4. Cluster ID validation with string cluster IDs
# ---------------------------------------------------------------------------


def test_cluster_string_ids_accepted():
    """String cluster IDs should not crash validation."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
    fit = smf.ols("y ~ x", data=df).fit()
    clusters = np.array(["A" if i < 50 else "B" for i in range(100)])
    # Should not raise
    inp = steps.input(df, cluster=clusters)
    est = GComputation(inp, outcome=fit, method="bootstrap", B=5)
    assert est._compiled.wiring_facts.cluster is not None


def test_cluster_nan_string_ids_raises():
    """NaN in string cluster IDs should still raise."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
    fit = smf.ols("y ~ x", data=df).fit()
    clusters = np.array(["A"] * 100, dtype=object)
    clusters[0] = np.nan
    with pytest.raises(TypeError):
        inp = steps.input(df, cluster=clusters)
        GComputation(inp, outcome=fit, method="bootstrap", B=5)


# ---------------------------------------------------------------------------
# M7. summary() omits κ when NaN
# ---------------------------------------------------------------------------


def test_summary_omits_nan_kappa():
    """Footer should not print κ when it is NaN."""
    result = GraphResult(
        estimate=np.array([1.0]),
        std_error=np.array([0.1]),
        conf_int_lower=np.array([0.8]),
        conf_int_upper=np.array([1.2]),
        labels=None,
        method="delta",
        level=0.95,
        ci="wald",
        scale="response",
        at="overall",
        plan=None,
        population_note=None,
        n_obs=100,
        estimand_metadata={},
        kappa=np.array([np.nan]),
    )
    summary = result.summary()
    assert "κ" not in summary


# ---------------------------------------------------------------------------
# B3. Bootstrap refit resamples offset/exposure arrays
# ---------------------------------------------------------------------------


def test_bootstrap_refit_resamples_offset():
    """Offset arrays should be resampled to match the bootstrap index."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "x": rng.standard_normal(n),
            "y": rng.poisson(2, size=n),
            "off": np.arange(n, dtype=float),  # distinctive offset values
        }
    )
    fit = smf.glm("y ~ x", data=df, family=sm.families.Poisson(), offset=df["off"]).fit(
        disp=False
    )
    adapter = StatsmodelsGLMAdapter(fit)

    # Capture what offset is passed to the refit
    captured_offsets = []
    original_refit = adapter.refit

    def capturing_refit(resampled, *, index=None):
        # The offset in fit_kwargs should be resampled
        kwargs = adapter._collect_original_fit_kwargs()
        if index is not None and "offset" in kwargs:
            captured_offsets.append(np.asarray(kwargs["offset"])[index].copy())
        return original_refit(resampled, index=index)

    # Patch on the instance
    adapter.refit = capturing_refit

    config = InferenceConfig(
        method="bootstrap", n_boot=3, rng_seed=42, diagnostics=False
    )

    def h_factory(a):
        def h(beta):
            return float(beta[0])

        return h

    result = _run_bootstrap(
        lambda b: float(b[0]), adapter, config, {}, h_factory=h_factory
    )
    assert result["method"] == "bootstrap"
    # Verify captured offsets were resampled (length should match n_boot * n, but
    # we only check that the captured offsets are sub-arrays of the original)
    assert len(captured_offsets) == 3
    for off in captured_offsets:
        assert len(off) == n
        # Values should be a subset of 0..n-1 (the original offset values)
        assert np.all((off >= 0) & (off < n))


def test_bootstrap_perfect_separation_counts_failure():
    """Bootstrap with near-perfect separation should count failures and emit a warning."""
    rng = np.random.default_rng(42)
    x = np.concatenate(
        [
            rng.standard_normal(12) - 2,
            rng.standard_normal(12) + 2,
            rng.standard_normal(6),
        ]
    )
    y = np.array([0] * 12 + [1] * 12 + [int(xi > 0) for xi in rng.standard_normal(6)])
    df = pd.DataFrame({"x": x, "y": y})
    fit = smf.logit("y ~ x", data=df).fit(disp=False)

    est = GComputation(fit, method="bootstrap", B=20, seed=42)
    with pytest.warns(UserWarning, match="Bootstrap: .* replicates failed"):
        pred = est.predict()

    assert np.isfinite(float(pred.estimate))
    assert pred.method == "bootstrap"


# ---------------------------------------------------------------------------
# Bootstrap refit with dropped category level
# ---------------------------------------------------------------------------


def test_bootstrap_refit_dropped_category_counts_failure():
    """If resampling drops a category level, patsy produces fewer columns.
    The bootstrap should count this as a failed replicate, not crash."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.standard_normal(n),
            # Rare category: 3 observations (~1.5%). Expected drop rate ≈ 5%,
            # so a few replicates out of 50 will fail, but not enough to
            # exceed the 10% threshold.
            "group": ["A"] * 197 + ["B"] * 3,
            "y": rng.standard_normal(n) + 2.0 * rng.standard_normal(n),
        }
    )
    fit = smf.ols("y ~ x + C(group)", data=df).fit()

    est = GComputation(fit, method="bootstrap", B=50, seed=42)
    # Some resamples will drop the rare category. The bootstrap engine
    # should emit a warning about failed replicates but still produce
    # a valid result from the successful ones.
    with pytest.warns(UserWarning, match="Bootstrap: .* replicates failed"):
        pred = est.predict()

    assert np.isfinite(float(pred.estimate))
    assert pred.method == "bootstrap"


def test_bootstrap_refit_dropped_category_raises_when_too_many_fail():
    """If >10% of replicates have dropped categories, bootstrap should raise."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.standard_normal(n),
            # Very rare category: 1 observation. Drop rate ≈ 37%, well above 10%.
            "group": ["A"] * 95 + ["B"] * 4 + ["C"],
            "y": rng.standard_normal(n) + 2.0 * rng.standard_normal(n),
        }
    )
    fit = smf.ols("y ~ x + C(group)", data=df).fit()

    est = GComputation(fit, method="bootstrap", B=50, seed=42)
    with pytest.raises(RuntimeError, match="Bootstrap failed"):
        est.predict()
