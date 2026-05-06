"""
pymargins._estimands

Internal estimand construction. An "estimand" is a pure function h(β) that
the inference engine differentiates and evaluates. This module builds these
functions from user-facing arguments (variables, scenarios, contrasts, etc.).

This layer is internal — users do not construct estimand functions directly.
The Margins entry-point methods (predict, dydx, contrasts, evaluate) call
into this module to assemble the appropriate h(β) for the inference engine.

Design
------
Estimands are pure functions (not classes) returning scalars or vectors.
This keeps composition trivial: linear combinations are weighted sums of
function outputs; nonlinear compositions are arbitrary functions of function
outputs. JAX handles the autodiff plumbing.

The estimand functions returned here are on the **inference scale** — they
incorporate phi_inv if the session uses a non-identity transform, so that
the output is suitable for delta-method computation. The session's phi is
applied by the inference engine after CI construction, not inside the
estimand function.
"""

from __future__ import annotations
from typing import Callable, Optional, Union, Any
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Prediction estimand (level quantity)
# ---------------------------------------------------------------------------

def make_prediction_estimand(
    adapter,
    X: jnp.ndarray,
    *,
    aggregate: str = "overall",
    weights: Optional[jnp.ndarray] = None,
    phi_inv: Optional[Callable] = None,
    offset: Optional[jnp.ndarray] = None,
    transform: Optional[Callable] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Construct h(β) for an adjusted prediction.

    The returned function evaluates predictions at the design rows X,
    optionally applies a per-row user transform, aggregates per the rule,
    and lifts to the inference scale via phi_inv.

    Parameters
    ----------
    adapter : ModelAdapter
        Provides predict(beta, X, offset).

    X : jax array of shape (n_rows, n_features)
        Design matrix for evaluation. May be:
          - One row: single prediction at a scenario
          - Many rows: overall averaging over the rows
          - One row per scenario: vector-valued estimand

    aggregate : str, default "overall"
        How to combine across rows:
          "overall"   : mean over rows (AAP)
          "none"      : no averaging; output is per-row vector
          "weighted"  : weighted mean using `weights` argument
        For at-typical / at-means predictions, X has one row and aggregate
        is irrelevant.

    weights : jax array of shape (n_rows,), optional
        Aggregation weights. Used when aggregate="weighted". For overall
        averaging, falls back to uniform weights.

    phi_inv : callable, optional
        Forward transform from reporting scale to inference scale. Applied
        after transform and averaging. None for identity scale.

    offset : jax array of shape (n_rows,), optional
        Offset for the prediction, passed to adapter.predict.

    transform : callable, optional
        Per-row mapping applied to predictions before aggregation. Receives
        the per-row prediction array ``μ`` and returns an array of matching
        shape (or broadcastable). Distinct from the cross-scenario
        composition used by ``make_evaluate_estimand``.

    Returns
    -------
    h : callable (beta) -> scalar or vector
        Inference-scale estimand suitable for jax.grad and the delta engine.
    """
    def h(beta):
        mu = adapter.predict(beta, X, offset=offset)
        if transform is not None:
            mu = transform(mu)
        if aggregate == "overall":
            # For multi-output, average over rows (axis=0), keeping outputs
            value = jnp.mean(mu, axis=0) if mu.ndim > 1 else jnp.mean(mu)
        elif aggregate == "weighted":
            if weights is None:
                value = jnp.mean(mu, axis=0) if mu.ndim > 1 else jnp.mean(mu)
            else:
                if mu.ndim > 1:
                    # weights (n_rows,) broadcast against mu (n_rows, n_outputs)
                    value = jnp.sum(weights[:, None] * mu, axis=0) / jnp.sum(weights)
                else:
                    value = jnp.sum(weights * mu) / jnp.sum(weights)
        elif aggregate == "none":
            # Squeeze singleton row dimension for scalar estimands
            value = mu[0] if mu.shape[0] == 1 else mu
        else:
            raise ValueError(f"Unknown aggregate rule: {aggregate!r}")
        if phi_inv is not None:
            value = phi_inv(value)
        return value

    return h


# ---------------------------------------------------------------------------
# Slope estimand (∂μ/∂x_j)
# ---------------------------------------------------------------------------

def make_slope_estimand(
    adapter,
    X: jnp.ndarray,
    var_index: int,
    *,
    aggregate: str = "overall",
    weights: Optional[jnp.ndarray] = None,
    phi_inv: Optional[Callable] = None,
    offset: Optional[jnp.ndarray] = None,
    transform: Optional[Callable] = None,
    backend: str = "autodiff",
    fd_step: float = 1e-6,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Construct h(β) for a slope (marginal effect of a continuous variable).

    Computes ∂μ/∂x_j per row via autodiff (or FD) over the design matrix
    direction, aggregates per rule, applies phi_inv. The double-derivative
    structure (inner ∂/∂x for the slope, outer ∂/∂β for inference) is
    handled cleanly by JAX's nested differentiation.

    Parameters
    ----------
    adapter : ModelAdapter
        Provides predict, must support differentiation w.r.t. its X argument.
        For LinearPredictionAdapter and GLMAdapter the slope is exact via
        autodiff. For WrappedFDAdapter the slope uses the same FD mechanism
        as gradients.

    X : jax array of shape (n_rows, n_features)
        Evaluation points.

    var_index : int
        Column index of the slope variable in X. The adapter's
        variable_metadata() can map a variable name to its index in X.

    aggregate : str, default "overall"
        See make_prediction_estimand.

    weights, phi_inv, offset : as in make_prediction_estimand.

    transform : callable, optional
        Per-row mapping applied to slopes before averaging.

    backend : str, default "autodiff"
        Gradient backend for the inner ∂/∂x derivative. Passed to
        directional_derivative. Should match the session's gradient_backend.

    fd_step : float, default 1e-6
        FD step for the inner derivative when backend is "fd" or "wrapped_fd".

    Returns
    -------
    h : callable (beta) -> scalar or vector
    """
    from ._gradients import directional_derivative

    def slope_at_row(beta, x_row):
        # Directional derivative of prediction at this row along x[var_index]
        def predict_at_x(x):
            return adapter.predict(beta, x[None, :], offset=offset)[0]
        direction = jnp.zeros_like(x_row).at[var_index].set(1.0)
        return directional_derivative(
            predict_at_x, x_row, direction,
            backend=backend, fd_step=fd_step,
        )

    def h(beta):
        slopes = jax.vmap(slope_at_row, in_axes=(None, 0))(beta, X)
        if transform is not None:
            slopes = transform(slopes)
        if aggregate == "overall":
            value = jnp.mean(slopes, axis=0) if slopes.ndim > 1 else jnp.mean(slopes)
        elif aggregate == "weighted":
            if weights is None:
                value = jnp.mean(slopes, axis=0) if slopes.ndim > 1 else jnp.mean(slopes)
            else:
                if slopes.ndim > 1:
                    value = jnp.sum(weights[:, None] * slopes, axis=0) / jnp.sum(weights)
                else:
                    value = jnp.sum(weights * slopes) / jnp.sum(weights)
        elif aggregate == "none":
            value = slopes
        else:
            raise ValueError(f"Unknown aggregate rule: {aggregate!r}")
        if phi_inv is not None:
            value = phi_inv(value)
        return value

    return h


# ---------------------------------------------------------------------------
# Linear combination across scenarios
# ---------------------------------------------------------------------------

def make_linear_combination_estimand(
    adapter,
    scenarios_X: list[jnp.ndarray],
    weights: Union[jnp.ndarray, dict[str, jnp.ndarray]],
    *,
    phi_inv: Optional[Callable] = None,
    scenario_offsets: Optional[list[Optional[jnp.ndarray]]] = None,
    scenario_aggregate: str = "overall",
    scenario_weights: Optional[list[Optional[jnp.ndarray]]] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Construct h(β) for a linear combination of predictions across scenarios.

    Each scenario is its own design matrix; the prediction at each scenario
    is aggregated per scenario_aggregate (e.g., averaged across rows for
    overall averaging), and the resulting per-scenario predictions are
    linearly combined with the supplied weights.

    Supports either a single weight vector (returns scalar h) or a dict
    of named contrasts (returns vector h with one component per contrast).

    Parameters
    ----------
    adapter : ModelAdapter
        Provides predict.

    scenarios_X : list of jax arrays
        One design matrix per scenario. Lengths can differ across scenarios
        (each may have its own row count for overall averaging).

    weights : jax array or dict
        - If 1D array of shape (n_scenarios,): single contrast.
        - If dict mapping names to 1D arrays: multiple named contrasts;
          all weight vectors must have the same length as scenarios_X.

    phi_inv : callable, optional
        Forward transform to inference scale. Applied to the combined
        per-scenario predictions before returning.

    scenario_offsets : list of (jax array | None), optional
        Per-scenario offsets, same length as scenarios_X.

    scenario_aggregate : str, default "overall"
        How to aggregate within each scenario before linear combination.

    scenario_weights : list of (jax array | None), optional
        Per-scenario aggregation weights. Used when scenario_aggregate
        is "weighted".

    Returns
    -------
    h : callable (beta) -> scalar or vector
        Scalar for single weight vector; vector for dict of contrasts.
    """
    n_scenarios = len(scenarios_X)
    offsets = scenario_offsets if scenario_offsets else [None] * n_scenarios
    sw = scenario_weights if scenario_weights else [None] * n_scenarios

    def per_scenario_value(beta, X, offset, w):
        mu = adapter.predict(beta, X, offset=offset)
        if scenario_aggregate == "overall":
            return jnp.mean(mu)
        elif scenario_aggregate == "weighted":
            if w is None:
                return jnp.mean(mu)
            return jnp.sum(w * mu) / jnp.sum(w)
        elif scenario_aggregate == "none":
            # Means a single-row scenario; just return scalar
            return mu[0] if mu.shape[0] == 1 else mu
        else:
            raise ValueError(f"Unknown scenario_aggregate: {scenario_aggregate!r}")

    def h(beta):
        # Compute per-scenario predictions (may be scalar each)
        scenario_values = jnp.stack([
            per_scenario_value(beta, scenarios_X[i], offsets[i], sw[i])
            for i in range(n_scenarios)
        ])

        # Apply phi_inv to per-scenario values BEFORE the linear combination
        # (the combination is on the inference scale, so we need to be there
        #  already). For log-scale: log(p1) - log(p0), not log(p1 - p0).
        if phi_inv is not None:
            scenario_values = phi_inv(scenario_values)

        if isinstance(weights, dict):
            # Multiple contrasts: stack their values
            weight_matrix = jnp.stack([
                jnp.asarray(weights[name]) for name in weights
            ])
            return weight_matrix @ scenario_values
        else:
            return jnp.dot(jnp.asarray(weights), scenario_values)

    return h


# ---------------------------------------------------------------------------
# Arbitrary nonlinear composition
# ---------------------------------------------------------------------------

def make_evaluate_estimand(
    adapter,
    scenarios_X: list[jnp.ndarray],
    compose: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    phi_inv: Optional[Callable] = None,
    scenario_offsets: Optional[list[Optional[jnp.ndarray]]] = None,
    scenario_aggregate: str = "overall",
    scenario_weights: Optional[list[Optional[jnp.ndarray]]] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Construct h(β) for an arbitrary differentiable composition of scenario
    predictions.

    Like make_linear_combination_estimand but the user supplies a JAX-
    compatible callable that takes the per-scenario prediction vector and
    returns a scalar or vector. Used for nonlinear compositions: NNT
    (1/(p_a - p_b)), custom utility functions, ratios across non-paired
    scenarios.

    The compose function MUST be JAX-compatible (use jnp ops, no Python
    conditionals on tracer values) for delta-method inference. Non-JAX
    compose functions are detected and routed to simulation/bootstrap by
    the inference engine.

    Parameters
    ----------
    adapter : ModelAdapter
        Provides predict.

    scenarios_X : list of jax arrays
        Per-scenario design matrices.

    compose : callable
        Function (scenario_predictions) -> scalar or vector. Receives a 1D
        array of length n_scenarios with the (aggregated) prediction from
        each scenario. Must be JAX-compatible if delta-method inference is
        intended.

    phi_inv : callable, optional
        Forward transform to inference scale. Applied to the compose output.

    scenario_offsets, scenario_aggregate, scenario_weights : as in
        make_linear_combination_estimand.

    Returns
    -------
    h : callable (beta) -> scalar or vector
    """
    n_scenarios = len(scenarios_X)
    offsets = scenario_offsets if scenario_offsets else [None] * n_scenarios
    sw = scenario_weights if scenario_weights else [None] * n_scenarios

    def per_scenario_value(beta, X, offset, w):
        mu = adapter.predict(beta, X, offset=offset)
        if scenario_aggregate == "overall":
            return jnp.mean(mu)
        elif scenario_aggregate == "weighted":
            if w is None:
                return jnp.mean(mu)
            return jnp.sum(w * mu) / jnp.sum(w)
        elif scenario_aggregate == "none":
            return mu[0] if mu.shape[0] == 1 else mu
        else:
            raise ValueError(f"Unknown scenario_aggregate: {scenario_aggregate!r}")

    def h(beta):
        scenario_values = jnp.stack([
            per_scenario_value(beta, scenarios_X[i], offsets[i], sw[i])
            for i in range(n_scenarios)
        ])
        result = compose(scenario_values)
        if phi_inv is not None:
            result = phi_inv(result)
        return result

    return h


# ---------------------------------------------------------------------------
# Differentiability check
# ---------------------------------------------------------------------------

def is_jax_differentiable(h: Callable, beta: jnp.ndarray) -> bool:
    """Test whether an estimand function is JAX-differentiable at beta.

    Used by the inference engine to decide whether delta-method inference
    is possible. If h contains operations JAX can't trace (Python
    conditionals on values, NumPy ops without conversion, etc.), this
    returns False and the engine falls back to simulation or bootstrap.

    Parameters
    ----------
    h : callable
        Estimand function.

    beta : jax array
        Test point.

    Returns
    -------
    differentiable : bool
        True if jax.grad(h)(beta) succeeds.
    """
    import jax
    out = h(beta)
    # Narrow to JAX tracer/concretization errors and TypeError.
    # Avoid catching bare Exception so genuine programming errors in
    # estimand factories propagate rather than being silently treated as
    # non-differentiable.
    jax_tracer_errors: tuple = (TypeError,)
    if hasattr(jax, "errors"):
        tracer_errs = []
        for name in (
            "TracerIntegerConversionError",
            "TracerArrayConversionError",
            "TracerTupleConversionError",
            "ConcretizationTypeError",
            "UnexpectedTracerError",
        ):
            if hasattr(jax.errors, name):
                tracer_errs.append(getattr(jax.errors, name))
        if tracer_errs:
            jax_tracer_errors = tuple(tracer_errs)
    try:
        if jnp.ndim(out) == 0:
            jax.grad(h)(beta)
        else:
            jax.jacobian(h)(beta)
        return True
    except jax_tracer_errors:
        return False


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Build a prediction estimand for AAP
----------------------------------------------

    from pymargins._estimands import make_prediction_estimand

    df = adapter.training_data  # full design
    X = adapter.design_matrix_from_df(df)
    h = make_prediction_estimand(
        adapter, X,
        aggregate="overall",
        phi_inv=None,  # identity scale
    )

    # Pass h to the gradient + delta machinery
    from pymargins._gradients import gradient
    grad = gradient(h, beta_hat)


Example 2: Build a slope estimand for AME
-----------------------------------------

    from pymargins._estimands import make_slope_estimand

    var_idx = adapter.variable_metadata()["age"].index_in_X  # adapter helper
    h = make_slope_estimand(
        adapter, X, var_index=var_idx,
        aggregate="overall",
    )


Example 3: Build a linear combination for diff-in-diff
------------------------------------------------------

    from pymargins._estimands import make_linear_combination_estimand

    df_TT = adapter.training_data.iloc[:1].copy()
    df_TT["treat"] = 1; df_TT["post"] = 1
    df_TC = adapter.training_data.iloc[:1].copy()
    df_TC["treat"] = 1; df_TC["post"] = 0
    df_CT = adapter.training_data.iloc[:1].copy()
    df_CT["treat"] = 0; df_CT["post"] = 1
    df_CC = adapter.training_data.iloc[:1].copy()
    df_CC["treat"] = 0; df_CC["post"] = 0

    X_TT = adapter.design_matrix_from_df(df_TT)
    X_TC = adapter.design_matrix_from_df(df_TC)
    X_CT = adapter.design_matrix_from_df(df_CT)
    X_CC = adapter.design_matrix_from_df(df_CC)

    h = make_linear_combination_estimand(
        adapter,
        scenarios_X=[X_TT, X_TC, X_CT, X_CC],
        weights=jnp.array([+1.0, -1.0, -1.0, +1.0]),
        phi_inv=None,
    )


Example 4: Build a log-scale ratio estimand
-------------------------------------------

    # Session has phi=exp, phi_inv=log

    h = make_linear_combination_estimand(
        adapter,
        scenarios_X=[X_treated, X_control],
        weights=jnp.array([+1.0, -1.0]),
        phi_inv=jnp.log,  # log-scale: contrast is log(p_treat) - log(p_control)
    )
    # h(beta) returns log(p_treat) - log(p_control); the engine exponentiates
    # CI endpoints via phi=exp for reporting as RR.


Example 5: Build an NNT estimand via evaluate
---------------------------------------------

    from pymargins._estimands import make_evaluate_estimand

    h = make_evaluate_estimand(
        adapter,
        scenarios_X=[X_control, X_treated],
        compose=lambda preds: 1.0 / (preds[0] - preds[1]),
        phi_inv=None,
    )
    # h(beta) returns 1/(p_control - p_treated) — the NNT
"""
