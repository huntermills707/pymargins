"""Targeted tests for remaining coverage gaps in _inference/_bootstrap.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pymargins._inference._bootstrap import (
    _compute_acceleration_jackknife,
    _generate_resample_indices,
    _refit_replicate_task,
    _run_bootstrap,
    _try_fast_path,
)

# ---------------------------------------------------------------------------
# _generate_resample_indices block bootstrap paths
# ---------------------------------------------------------------------------


def test_resample_indices_block_moving():
    """Cover block bootstrap moving type."""
    idx = _generate_resample_indices(
        n_obs=20, n_boot=5, rng_seed=1, block_size=4, block_type="moving"
    )
    assert len(idx) == 5
    for arr in idx:
        assert len(arr) >= 20


def test_resample_indices_block_circular():
    """Cover block bootstrap circular type."""
    idx = _generate_resample_indices(
        n_obs=20, n_boot=5, rng_seed=1, block_size=4, block_type="circular"
    )
    assert len(idx) == 5
    for arr in idx:
        assert len(arr) >= 20


def test_resample_indices_block_nonoverlapping():
    """Cover block bootstrap nonoverlapping type."""
    idx = _generate_resample_indices(
        n_obs=20, n_boot=5, rng_seed=1, block_size=5, block_type="nonoverlapping"
    )
    assert len(idx) == 5
    for arr in idx:
        assert len(arr) >= 20


def test_resample_indices_block_size_too_large():
    """Cover block bootstrap block_size > n_obs - still works via modulo (line 114-116 is dead code)."""
    idx = _generate_resample_indices(
        n_obs=5, n_boot=1, rng_seed=1, block_size=10, block_type="nonoverlapping"
    )
    assert len(idx) == 1
    assert len(idx[0]) >= 5


# ---------------------------------------------------------------------------
# _compute_acceleration_jackknife
# ---------------------------------------------------------------------------


def test_jackknife_cluster_too_many_units():
    """Cover jackknife n_units > 200 early return."""
    adapter = MagicMock()
    h = MagicMock()
    h_factory = MagicMock()
    data = pd.DataFrame({"x": range(250)})
    cluster_ids = np.repeat(range(250), 1)
    z0, a = _compute_acceleration_jackknife(
        adapter, h_factory, h, data, cluster_ids, None
    )
    assert z0 is None
    assert a is None


def test_jackknife_block_bootstrap_returns_none():
    """Cover jackknife block_size not None early return."""
    adapter = MagicMock()
    h = MagicMock()
    h_factory = MagicMock()
    data = pd.DataFrame({"x": range(10)})
    z0, a = _compute_acceleration_jackknife(
        adapter, h_factory, h, data, None, block_size=4
    )
    assert z0 is None
    assert a is None


def test_jackknife_n_obs_too_large():
    """Cover jackknife n_obs > 200 early return."""
    adapter = MagicMock()
    h = MagicMock()
    h_factory = MagicMock()
    data = pd.DataFrame({"x": range(250)})
    z0, a = _compute_acceleration_jackknife(adapter, h_factory, h, data, None, None)
    assert z0 is None
    assert a is None


# ---------------------------------------------------------------------------
# _refit_replicate_task error paths
# ---------------------------------------------------------------------------


def test_refit_replicate_task_matching_rematch_error():
    """Cover _refit_replicate_task matching rematch exception (lines 325-330)."""
    adapter = MagicMock()
    matching = MagicMock()
    matching.rematch.side_effect = RuntimeError("rematch failed")
    data = pd.DataFrame({"x": [1, 2, 3]})
    result = _refit_replicate_task(
        (0, np.array([0, 1, 2])),
        adapter=adapter,
        data=data,
        matching=matching,
    )
    assert result[0] == 0
    assert result[1] is None
    assert isinstance(result[2], RuntimeError)


def test_refit_replicate_task_assertion_error_propagates():
    """Cover _refit_replicate_task assertion error propagation (lines 351-354)."""
    adapter = MagicMock()
    adapter.refit.side_effect = AssertionError("bad")
    data = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(AssertionError):
        _refit_replicate_task(
            (0, np.array([0, 1, 2])),
            adapter=adapter,
            data=data,
            matching=None,
        )


def test_refit_replicate_task_param_count_mismatch():
    """Cover _refit_replicate_task parameter count mismatch."""
    adapter = MagicMock()
    adapter.refit.return_value = MagicMock()
    adapter.refit.return_value.coefficients.return_value = np.array([1.0, 2.0])
    adapter.coefficients.return_value = np.array([1.0])
    data = pd.DataFrame({"x": [1, 2, 3]})
    result = _refit_replicate_task(
        (0, np.array([0, 1, 2])),
        adapter=adapter,
        data=data,
        matching=None,
    )
    assert result[0] == 0
    assert result[1] is None
    assert isinstance(result[2], ValueError)


# ---------------------------------------------------------------------------
# _try_fast_path
# ---------------------------------------------------------------------------


def test_try_fast_path_loop_engine():
    """Cover _try_fast_path with engine='loop'."""

    adapter = MagicMock()
    adapter.supports_jax_autodiff = True
    config = MagicMock()
    config.bootstrap_config = {"engine": "loop"}

    def h(beta):
        return beta[0]

    result = _try_fast_path(h, adapter, config)
    assert result is None


def test_try_fast_path_no_autodiff():
    """Cover _try_fast_path without autodiff support."""

    adapter = MagicMock()
    adapter.supports_jax_autodiff = False
    config = MagicMock()
    config.bootstrap_config = {}

    def h(beta):
        return beta[0]

    result = _try_fast_path(h, adapter, config)
    assert result is None


def test_try_fast_path_not_kernel_partial():
    """Cover _try_fast_path when h is not a kernel partial."""
    adapter = MagicMock()
    adapter.supports_jax_autodiff = True
    config = MagicMock()
    config.bootstrap_config = {}

    def h(beta):
        return beta[0]

    result = _try_fast_path(h, adapter, config)
    assert result is None


# ---------------------------------------------------------------------------
# _run_bootstrap validation errors
# ---------------------------------------------------------------------------


def test_run_bootstrap_no_h_factory():
    """Cover _run_bootstrap h_factory missing error (lines 896-900)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="h_factory"):
        _run_bootstrap(h, adapter, config, {})


def test_run_bootstrap_no_training_data():
    """Cover _run_bootstrap training_data missing error (lines 902-908)."""
    from pymargins._inference._config import InferenceConfig

    class BadAdapter:
        @property
        def training_data(self):
            raise NotImplementedError("no training data")

    adapter = BadAdapter()
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
    )

    def h(beta):
        return beta[0]

    with pytest.raises(NotImplementedError, match="training_data"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_cluster_and_block_size():
    """Cover _run_bootstrap cluster+block_size mutual exclusion (lines 936-940)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        cluster=np.array([1, 1, 2]),
        block_size=2,
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="mutually exclusive"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_cluster_length_mismatch():
    """Cover _run_bootstrap cluster length mismatch (lines 944-948)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        cluster=np.array([1, 1]),
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="cluster IDs length"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_cluster_nan():
    """Cover _run_bootstrap cluster NaN values (lines 949-950)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        cluster=np.array([1, np.nan, 2]),
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="NaN"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_block_size_negative():
    """Cover _run_bootstrap invalid block_size (lines 957-958)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        block_size=0,
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="block_size"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_block_size_too_large():
    """Cover _run_bootstrap block_size > n_obs (lines 959-964)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        block_size=10,
    )

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="block_size"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


def test_run_bootstrap_unsupported_ci_method():
    """Cover _run_bootstrap unsupported ci_method (lines 930-934)."""
    from pymargins._inference._config import InferenceConfig

    adapter = MagicMock()
    adapter.training_data = pd.DataFrame({"x": [1, 2, 3]})
    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
    )
    config.bootstrap_config = {"ci_method": "invalid"}

    def h(beta):
        return beta[0]

    with pytest.raises(ValueError, match="ci_method"):
        _run_bootstrap(h, adapter, config, {}, h_factory=lambda a: h)


# ---------------------------------------------------------------------------
# _run_bootstrap basic CI method with phi error
# ---------------------------------------------------------------------------


def test_run_bootstrap_basic_with_phi():
    """Cover basic CI method with phi error (lines 1081-1088)."""
    import jax.numpy as jnp

    from pymargins._inference._config import InferenceConfig

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": np.random.randn(30),
            "y": np.random.randn(30),
        }
    )
    import statsmodels.api as sm

    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter

    adapter = StatsmodelsOLSAdapter(fit, training_data=df)

    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    config.bootstrap_config = {"ci_method": "basic"}

    def h_factory(a):
        def h(beta):
            return beta[0]

        return h

    with pytest.raises(ValueError, match="basic bootstrap"):
        _run_bootstrap(h_factory(adapter), adapter, config, {}, h_factory=h_factory)


# ---------------------------------------------------------------------------
# _run_bootstrap studentized CI method with phi error
# ---------------------------------------------------------------------------


def test_run_bootstrap_studentized_with_phi():
    """Cover studentized CI method with phi error (lines 1113-1120)."""
    import jax.numpy as jnp

    from pymargins._inference._config import InferenceConfig

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": np.random.randn(30),
            "y": np.random.randn(30),
        }
    )
    import statsmodels.api as sm

    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter

    adapter = StatsmodelsOLSAdapter(fit, training_data=df)

    config = InferenceConfig(
        method="bootstrap",
        n_boot=10,
        phi=jnp.exp,
        phi_inv=jnp.log,
    )
    config.bootstrap_config = {"ci_method": "studentized"}

    def h_factory(a):
        def h(beta):
            return beta[0]

        return h

    with pytest.raises(ValueError, match="studentized bootstrap"):
        _run_bootstrap(h_factory(adapter), adapter, config, {}, h_factory=h_factory)


# ---------------------------------------------------------------------------
# _run_bootstrap all replicates fail
# ---------------------------------------------------------------------------


def test_run_bootstrap_all_fail():
    """Cover all bootstrap replicates failing - hits threshold error first (lines 1032-1035)."""
    from pymargins._inference._config import InferenceConfig

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": np.random.randn(10),
            "y": np.random.randn(10),
        }
    )
    import statsmodels.api as sm

    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    from pymargins._adapters.statsmodels_ols import StatsmodelsOLSAdapter

    adapter = StatsmodelsOLSAdapter(fit, training_data=df)

    config = InferenceConfig(
        method="bootstrap",
        n_boot=3,
    )

    def h_factory(a):
        def h(beta):
            # Always fails
            raise ValueError("always fails")

        return h

    with pytest.raises(RuntimeError, match="threshold"):
        _run_bootstrap(h_factory(adapter), adapter, config, {}, h_factory=h_factory)
