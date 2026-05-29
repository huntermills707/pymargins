from __future__ import annotations

import warnings
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from .._adapter import ModelAdapter
from .._delta import delta_wald_test
from .._estimands import is_jax_differentiable
from ._bootstrap import _run_bootstrap
from ._config import InferenceConfig
from ._delta import _run_delta
from ._simulation import _run_simulation


def run_inference(
    h: Callable,
    adapter: ModelAdapter,
    config: InferenceConfig,
    *,
    estimand_metadata: dict | None = None,
    h_factory: Callable[[ModelAdapter], Callable] | None = None,
) -> dict:
    """Compute estimate, SE, CI, and diagnostics for an estimand.

    Top-level dispatch. Validates that the requested method is supported by
    the adapter, runs the appropriate inference path, and returns a dict of
    results (which Margins wraps into a MarginsResult).

    Parameters
    ----------
    h : callable
        Estimand function on the inference scale.

    adapter : ModelAdapter
        Provides β̂ (via coefficients) and predict.

    config : InferenceConfig
        Inference-time settings.

    estimand_metadata : dict, optional
        Bookkeeping passed through to the result (variable names, scenario
        labels, etc.) for output formatting.

    Returns
    -------
    result_data : dict
        Keys for assembling a MarginsResult: estimate, std_error,
        conf_int_lower, conf_int_upper, method, level, kappa, gradient,
        draws, fallback_triggered, etc.
    """
    method = config.method
    supported = adapter.supported_inference_methods

    if method not in supported:
        raise ValueError(
            f"Adapter {type(adapter).__name__} does not support method "
            f"'{method}'. Supported: {sorted(supported)}."
        )

    if method == "delta":
        beta = adapter.coefficients()
        if not is_jax_differentiable(h, beta):
            if "simulation" in supported:
                # Auto-route to simulation with a warning marker in the result
                warnings.warn(
                    "Estimand is not JAX-differentiable; falling back to simulation.",
                    UserWarning,
                    stacklevel=3,
                )
                return _run_simulation(
                    h,
                    adapter,
                    config,
                    estimand_metadata,
                    fallback_reason="non_differentiable",
                )
            elif "bootstrap" in supported and h_factory is not None:
                warnings.warn(
                    "Estimand is not JAX-differentiable; falling back to bootstrap.",
                    UserWarning,
                    stacklevel=3,
                )
                return _run_bootstrap(
                    h,
                    adapter,
                    config,
                    estimand_metadata,
                    fallback_reason="non_differentiable",
                    h_factory=h_factory,
                )
            else:
                raise ValueError(
                    "Estimand is not JAX-differentiable, and no fallback "
                    "method is available."
                )
        return _run_delta(h, adapter, config, estimand_metadata)

    elif method == "simulation":
        return _run_simulation(h, adapter, config, estimand_metadata)

    elif method == "bootstrap":
        return _run_bootstrap(
            h, adapter, config, estimand_metadata, h_factory=h_factory
        )

    else:
        raise ValueError(f"Unknown method: {method!r}")


def run_test(
    estimate: np.ndarray,
    grad: np.ndarray | None,
    cov_params: jnp.ndarray | None,
    draws: np.ndarray | None,
    *,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    method: str = "wald",
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis test on a result, dispatching by method.

    For delta-based results (gradient available), uses Wald test on the
    inference scale. For simulation-based results (draws available), uses
    the empirical distribution of draws to compute a tail probability.

    Parameters
    ----------
    estimate : array
        Point estimate(s) on the inference scale.

    grad : array, optional
        ∇h. Required for Wald tests on delta results.

    cov_params : jax array, optional
        Σ̂. Required for Wald tests.

    draws : array, optional
        Estimand draws. Used for simulation/bootstrap-based tests.

    null_value : float, default 0.0
        Hypothesized value on the inference scale.

    alternative : str
        "two-sided", "greater", or "less".

    method : str, default "wald"
        Test type. Currently only "wald" implemented.

    Returns
    -------
    (statistic, p_value) : tuple of arrays
    """
    if method != "wald":
        raise NotImplementedError(f"Test method {method!r} is not implemented.")

    if grad is not None and cov_params is not None:
        return delta_wald_test(
            jnp.asarray(estimate),
            jnp.asarray(grad),
            cov_params,
            null_value=null_value,
            alternative=alternative,
        )
    elif draws is not None:
        # Empirical p-value from draws: compare observed estimate to the
        # simulated sampling distribution.
        estimate = np.asarray(estimate)
        null_value = np.asarray(null_value)
        draws = np.asarray(draws)
        if alternative == "two-sided":
            # Two-sided p-value via 2*min(tail probabilities).
            # Correct for asymmetric simulation distributions.
            p_left = np.mean(draws <= null_value, axis=0)
            p_right = np.mean(draws >= null_value, axis=0)
            p = np.clip(2.0 * np.minimum(p_left, p_right), min=None, max=1.0)
        elif alternative == "greater":
            # H1: effect > null. Under H0, draws are centered at estimate;
            # the null lies to the left. Small p when null is in the lower
            # tail of the simulated distribution.
            p = np.mean(draws <= null_value, axis=0)
        elif alternative == "less":
            # H1: effect < null. Small p when null is in the upper tail.
            p = np.mean(draws >= null_value, axis=0)
        else:
            raise ValueError(f"Unknown alternative: {alternative!r}")
        # Return the observed estimate minus null as the effect-size statistic
        statistic = estimate - null_value
        return statistic, np.asarray(p)
    else:
        raise ValueError("Cannot run test: result has neither gradient nor draws.")
