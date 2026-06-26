"""Tests for pymargins._inference engine paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter
from pymargins._inference import (
    InferenceConfig,
    _run_simulation,
)


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


def test_simulation_vmap_path_executes(fit_logit, monkeypatch):
    """Bug 1 regression: jax.vmap must actually be called in _run_simulation."""
    adapter = StatsmodelsGLMAdapter(fit_logit)

    def h(b):
        return jax.scipy.special.expit(jnp.array([1.0, 50.0, 1.0]) @ b)

    config = InferenceConfig(
        method="simulation", n_sim=100, rng_seed=42, diagnostics=True
    )

    call_count = [0]
    original_vmap = jax.vmap

    def counting_vmap(f):
        def wrapper(*args, **kwargs):
            call_count[0] += 1
            return original_vmap(f)(*args, **kwargs)

        return wrapper

    # Monkeypatch jax.vmap inside _inference module only
    import pymargins._inference as _inf_mod

    monkeypatch.setattr(_inf_mod.jax, "vmap", counting_vmap)

    result = _run_simulation(h, adapter, config, estimand_metadata=None)

    assert call_count[0] > 0, "jax.vmap was never called in _run_simulation"
    assert result["method"] == "simulation"
    assert np.isfinite(float(result["estimate"]))


def test_is_jax_differentiable_detects_vmap_failure():
    """Probe must reject estimands that pass jax.grad but fail under vmap.

    A Python `if b[0] > 0:` evaluates concretely under jax.grad (single
    point) but raises TracerBoolConversionError under vmap. The probe
    must agree with the engine's actual trace pattern.
    """
    from pymargins._estimands import is_jax_differentiable

    def h(b):
        if b[0] > 0:
            return jnp.exp(b[0])
        return jnp.exp(b[1])

    beta = jnp.array([1.0, 2.0])
    assert is_jax_differentiable(h, beta) is False


def test_is_jax_differentiable_accepts_clean_estimand():
    """Probe must accept estimands that trace cleanly through vmap and hessian."""
    from pymargins._estimands import is_jax_differentiable

    def h(b):
        return jax.scipy.special.expit(jnp.sum(b))

    beta = jnp.array([0.5, -0.3, 1.0])
    assert is_jax_differentiable(h, beta) is True
