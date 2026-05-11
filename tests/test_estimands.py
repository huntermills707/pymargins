"""Tests for pymargins._estimands."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

from pymargins._estimands import (
    make_prediction_estimand,
    make_slope_estimand,
    make_linear_combination_estimand,
    make_evaluate_estimand,
)


class DummyAdapter:
    """Minimal adapter for testing estimands."""

    def __init__(self, multi_output=False):
        self.multi_output = multi_output

    def predict(self, beta, X, offset=None):
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        if self.multi_output:
            # Simulate (n_rows, 2) output
            return jnp.stack([eta, eta * 2], axis=1)
        return eta

    def design_matrix_from_df(self, df):
        # Simple identity design matrix for testing
        return jnp.asarray(df.values)


# ---------------------------------------------------------------------------
# make_prediction_estimand
# ---------------------------------------------------------------------------

def test_prediction_estimand_zero_weights_raises():
    """Zero-sum weights must raise ValueError."""
    adapter = DummyAdapter()
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    h = make_prediction_estimand(adapter, X, aggregate="overall", weights=jnp.array([0.0, 0.0]))
    beta = jnp.array([0.5, -0.3])
    with pytest.raises(ValueError, match="weights must not sum to zero"):
        h(beta)


def test_prediction_estimand_negative_weights_raises():
    """Negative weights must raise ValueError."""
    adapter = DummyAdapter()
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    h = make_prediction_estimand(adapter, X, aggregate="overall", weights=jnp.array([1.0, -1.0]))
    beta = jnp.array([0.5, -0.3])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        h(beta)


def test_prediction_estimand_multi_output_mean():
    """Multi-output predictions should average over rows, keeping outputs."""
    adapter = DummyAdapter(multi_output=True)
    X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = make_prediction_estimand(adapter, X, aggregate="overall")
    beta = jnp.array([1.0, 2.0])
    out = h(beta)
    # eta = [1, 2]; multi-output = [[1, 2], [2, 4]]; mean over rows = [1.5, 3.0]
    np.testing.assert_allclose(out, jnp.array([1.5, 3.0]), rtol=1e-10)


# ---------------------------------------------------------------------------
# make_slope_estimand
# ---------------------------------------------------------------------------

def test_slope_estimand_fd_step_validation():
    """Non-positive fd_step must raise ValueError."""
    adapter = DummyAdapter()
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    with pytest.raises(ValueError, match="fd_step must be positive"):
        make_slope_estimand(adapter, df, "x", fd_step=0.0)
    with pytest.raises(ValueError, match="fd_step must be positive"):
        make_slope_estimand(adapter, df, "x", fd_step=-1e-6)


def test_slope_estimand_zero_weights_raises():
    """Zero-sum weights in slope estimand must raise ValueError."""
    adapter = DummyAdapter()
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    h = make_slope_estimand(adapter, df, "x", aggregate="overall", weights=jnp.array([0.0, 0.0]))
    beta = jnp.array([0.5, -0.3])
    with pytest.raises(ValueError, match="weights must not sum to zero"):
        h(beta)


def test_slope_estimand_negative_weights_raises():
    """Negative weights in slope estimand must raise ValueError."""
    adapter = DummyAdapter()
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    h = make_slope_estimand(adapter, df, "x", aggregate="overall", weights=jnp.array([1.0, -1.0]))
    beta = jnp.array([0.5, -0.3])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        h(beta)


def test_slope_estimand_multi_output():
    """Slope estimand must handle multi-output predictions."""
    adapter = DummyAdapter(multi_output=True)
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    h = make_slope_estimand(adapter, df, "x", aggregate="overall")
    beta = jnp.array([0.5, -0.3])
    out = h(beta)
    # Should not crash and should return shape (2,)
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# make_linear_combination_estimand
# ---------------------------------------------------------------------------

def test_linear_combination_multi_output_mean():
    """Per-scenario aggregation should keep output dimensions for multi-output."""
    adapter = DummyAdapter(multi_output=True)
    X1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    X2 = jnp.array([[1.0, 1.0]])
    weights = jnp.array([1.0, -1.0])
    h = make_linear_combination_estimand(
        adapter, [X1, X2], weights, scenario_aggregate="overall"
    )
    beta = jnp.array([1.0, 2.0])
    out = h(beta)
    # X1 mean multi-output = [1.5, 3.0]; X2 mean multi-output = [3.0, 6.0]
    # contrast = [1.5-3.0, 3.0-6.0] = [-1.5, -3.0]
    np.testing.assert_allclose(out, jnp.array([-1.5, -3.0]), rtol=1e-10)


def test_linear_combination_empty_offsets():
    """Empty list scenario_offsets should be preserved, not replaced."""
    adapter = DummyAdapter()
    X1 = jnp.array([[1.0, 0.0]])
    X2 = jnp.array([[0.0, 1.0]])
    weights = jnp.array([1.0, -1.0])
    # Passing empty list for scenario_offsets should raise because length mismatches
    with pytest.raises((ValueError, IndexError)):
        h = make_linear_combination_estimand(
            adapter, [X1, X2], weights, scenario_offsets=[]
        )
        h(jnp.array([0.5, -0.3]))


def test_linear_combination_zero_scenario_weights_raises():
    """Zero-sum scenario_weights must raise ValueError."""
    adapter = DummyAdapter()
    X1 = jnp.array([[1.0, 0.0]])
    X2 = jnp.array([[0.0, 1.0]])
    weights = jnp.array([1.0, -1.0])
    h = make_linear_combination_estimand(
        adapter, [X1, X2], weights,
        scenario_aggregate="weighted",
        scenario_weights=[jnp.array([1.0, -1.0]), jnp.array([1.0, 1.0])],
    )
    beta = jnp.array([0.5, -0.3])
    with pytest.raises(ValueError, match="scenario_weights must not sum to zero"):
        h(beta)


# ---------------------------------------------------------------------------
# make_evaluate_estimand
# ---------------------------------------------------------------------------

def test_evaluate_estimand_multi_output_mean():
    """Per-scenario aggregation in evaluate should keep output dimensions."""
    adapter = DummyAdapter(multi_output=True)
    X1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    X2 = jnp.array([[1.0, 1.0]])
    h = make_evaluate_estimand(
        adapter, [X1, X2], lambda preds: preds[0] - preds[1], scenario_aggregate="overall"
    )
    beta = jnp.array([1.0, 2.0])
    out = h(beta)
    # X1 mean multi-output = [1.5, 3.0]; X2 mean multi-output = [3.0, 6.0]
    # diff = [-1.5, -3.0]
    np.testing.assert_allclose(out, jnp.array([-1.5, -3.0]), rtol=1e-10)


def test_evaluate_estimand_empty_offsets():
    """Empty list scenario_offsets should be preserved, not replaced."""
    adapter = DummyAdapter()
    X1 = jnp.array([[1.0, 0.0]])
    X2 = jnp.array([[0.0, 1.0]])
    with pytest.raises((ValueError, IndexError)):
        h = make_evaluate_estimand(
            adapter, [X1, X2], lambda preds: preds.sum(), scenario_offsets=[]
        )
        h(jnp.array([0.5, -0.3]))
