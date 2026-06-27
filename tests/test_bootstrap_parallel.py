"""Tests for bootstrap parallelization."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation, steps


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    df["y"] = 0.5 + 0.8 * df["x1"] - 0.4 * df["x2"] + rng.standard_normal(n)
    return df


@pytest.fixture
def fit(df):
    return smf.ols("y ~ x1 + x2", data=df).fit()


# ---------------------------------------------------------------------------
# Reproducibility: serial vs parallel gives identical results
# ---------------------------------------------------------------------------


def test_parallel_reproducible(fit):
    m1 = GComputation(fit, method="bootstrap", B=100, seed=42, n_jobs=1)
    m2 = GComputation(fit, method="bootstrap", B=100, seed=42, n_jobs=2)
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)
    np.testing.assert_allclose(res1.std_error, res2.std_error)


def test_parallel_n_jobs_minus_one(fit):
    m = GComputation(fit, method="bootstrap", B=50, seed=42, n_jobs=-1)
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)


# ---------------------------------------------------------------------------
# Parallel with cluster bootstrap
# ---------------------------------------------------------------------------


def test_parallel_cluster_bootstrap(fit, df):
    inp = steps.input(df, cluster=df.index % 10)
    m1 = GComputation(
        inp,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=42,
        n_jobs=1,
    )
    m2 = GComputation(
        inp,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=42,
        n_jobs=2,
    )
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Parallel with block bootstrap
# ---------------------------------------------------------------------------


def test_parallel_block_bootstrap(fit, df):
    inp = steps.input(df, block=10)
    m1 = GComputation(
        inp,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=42,
        n_jobs=1,
    )
    m2 = GComputation(
        inp,
        outcome=fit,
        method="bootstrap",
        B=100,
        seed=42,
        n_jobs=2,
    )
    res1 = m1.dydx("x1")
    res2 = m2.dydx("x1")
    np.testing.assert_allclose(res1.estimate, res2.estimate)
    np.testing.assert_allclose(res1.conf_int_lower, res2.conf_int_lower)
    np.testing.assert_allclose(res1.conf_int_upper, res2.conf_int_upper)


# ---------------------------------------------------------------------------
# Parallel with predictions and contrasts
# ---------------------------------------------------------------------------


def test_parallel_predictions(fit):
    m = GComputation(fit, method="bootstrap", B=50, seed=42, n_jobs=2)
    res = m.predict(atexog={"x1": [0, 1]})
    assert np.all(np.isfinite(res.estimate))
    assert np.all(np.isfinite(res.conf_int_lower))
    assert np.all(np.isfinite(res.conf_int_upper))


def test_parallel_contrasts(fit):
    m = GComputation(fit, method="bootstrap", B=50, seed=42, n_jobs=2)
    res = m.contrasts(
        scenarios=[
            {"atexog": {"x1": 1}},
            {"atexog": {"x1": 0}},
        ],
        contrasts=[+1, -1],
    )
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.conf_int_lower)
    assert np.isfinite(res.conf_int_upper)


# ---------------------------------------------------------------------------
# Smoke test: n_jobs > 1 with threadpool_limits
# ---------------------------------------------------------------------------


def test_n_jobs_two_respects_threadpool_limits(fit):
    """Bootstrap with n_jobs=2 should complete and return finite results."""
    m = GComputation(fit, method="bootstrap", B=50, seed=42, n_jobs=2)
    res = m.dydx("x1")
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.std_error)
    assert np.isfinite(res.conf_int_lower)
    assert np.isfinite(res.conf_int_upper)


# ---------------------------------------------------------------------------
# ProcessPoolExecutor pickle fallback
# ---------------------------------------------------------------------------


def test_parallel_fallback_when_unpicklable():
    """If the adapter cannot be pickled, parallel bootstrap should fall back
    to ThreadPoolExecutor with a RuntimeWarning."""
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    from pymargins._adapter import ModelAdapter
    from pymargins._inference import InferenceConfig, _run_bootstrap

    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    class UnpicklableAdapter(ModelAdapter):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        @property
        def supports_jax_autodiff(self):
            return True

        @property
        def supported_inference_methods(self):
            return {"delta", "simulation", "bootstrap"}

        @property
        def gradient_backend_recommendation(self):
            return "autodiff"

        def coefficients(self):
            return np.asarray(self._wrapped.params)

        def covariance(self, vcov_spec=None):
            return np.asarray(self._wrapped.cov_params())

        def predict(self, beta, X, offset=None):
            import jax.numpy as jnp

            return jnp.asarray(X) @ jnp.asarray(beta)

        def design_matrix_from_df(self, df):
            return np.asarray(df[["x"]])

        def variable_metadata(self):
            return {"x": {"type": "continuous"}}

        def column_index_of_variable(self, name):
            return 0

        def refit(self, data, index=None):
            return self

        @property
        def training_data(self):
            return df

        def __reduce__(self):
            raise TypeError("Cannot pickle mock adapter")

    adapter = UnpicklableAdapter(fit)

    config = InferenceConfig(
        method="bootstrap",
        level=0.95,
        phi=None,
        phi_inv=None,
        kappa_threshold=float("inf"),
        gradient_backend="autodiff",
        fd_step=1e-6,
        n_sim=4000,
        n_boot=10,
        n_jobs=2,
        rng_seed=42,
        diagnostics=False,
        cov_params=adapter.covariance(),
    )

    def h(beta):
        import jax.numpy as jnp

        return jnp.sum(beta)

    def h_factory(new_adapter):
        return h

    with pytest.warns(RuntimeWarning, match="cannot be pickled"):
        result = _run_bootstrap(
            h,
            adapter,
            config,
            {"kind": "prediction"},
            h_factory=h_factory,
        )
    assert result is not None
    assert len(result["draws"]) == 10
