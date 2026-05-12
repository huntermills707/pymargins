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
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Kernel marker — used by the bootstrap path to detect module-level kernels
# that can be differentiated with a stable function identity.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _aggregate(mu, aggregate, weights):
    """Aggregate per-row predictions according to the rule."""
    if aggregate == "overall" or aggregate == "weighted":
        if weights is None:
            return jnp.mean(mu, axis=0) if mu.ndim > 1 else jnp.mean(mu)
        if not jnp.all(jnp.isfinite(weights)):
            raise ValueError("weights must be finite (no NaN or Inf)")
        if jnp.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if jnp.sum(weights) == 0:
            raise ValueError("weights must not sum to zero")
        if mu.ndim > 1:
            return jnp.sum(weights[:, None] * mu, axis=0) / jnp.sum(weights)
        return jnp.sum(weights * mu) / jnp.sum(weights)
    elif aggregate == "none":
        return mu[0] if mu.shape[0] == 1 else mu
    else:
        raise ValueError(f"Unknown aggregate rule: {aggregate!r}")


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
    return partial(
        prediction_kernel,
        predict_fn=adapter.predict,
        X=X,
        offset=offset,
        transform=transform,
        aggregate=aggregate,
        weights=weights,
        phi_inv=phi_inv,
    )


def prediction_kernel(
    beta,
    predict_fn,
    X,
    offset,
    transform,
    aggregate,
    weights,
    phi_inv,
):
    """Module-level kernel for prediction estimands.

    Using a top-level function (rather than a closure) lets JAX cache the
    compiled gradient across bootstrap replicates when the underlying
    ``predict_fn`` primitive is stable (e.g. cached GLM JVP wrappers).
    """
    mu = predict_fn(beta, X, offset=offset)
    if transform is not None:
        mu = transform(mu)
    value = _aggregate(mu, aggregate, weights)
    if phi_inv is not None:
        value = phi_inv(value)
    return value

prediction_kernel.__pymargins_kernel__ = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Slope estimand (∂μ/∂x_j)
# ---------------------------------------------------------------------------

def make_slope_estimand(
    adapter,
    df,
    var_name: str,
    *,
    aggregate: str = "overall",
    weights: Optional[jnp.ndarray] = None,
    phi_inv: Optional[Callable] = None,
    offset: Optional[jnp.ndarray] = None,
    transform: Optional[Callable] = None,
    fd_step: float = 1e-6,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Construct h(β) for the *total* marginal effect ∂μ/∂v.

    Computes the slope as a data-side central-difference: the source
    DataFrame's column ``var_name`` is perturbed by ±ε, the design matrix
    is rebuilt through ``adapter.design_matrix_from_df`` (so patsy
    regenerates every interaction, polynomial, spline, and ``I(...)``
    transform that depends on ``var_name``), and the prediction is
    central-differenced. This matches the semantics of R's
    ``marginaleffects::slopes()`` and Stata's ``margins, dydx()``: a
    request for the marginal effect of ``v`` returns the *total*
    derivative ∂μ/∂v including chain-rule contributions from every
    derived design column, not just the partial w.r.t. one column.

    The two perturbed design matrices are computed once at construction
    and captured by the closure; ``h(β)`` then only re-evaluates predict.
    β-gradients flow through predict via JAX as usual.

    Parameters
    ----------
    adapter : ModelAdapter
        Provides ``predict`` and ``design_matrix_from_df``.

    df : pd.DataFrame or TabularData
        Evaluation data. One row → slope at that row; multiple rows →
        per-row slopes aggregated per ``aggregate``. Must contain
        ``var_name`` as a column.

    var_name : str
        Source variable to perturb. Should be continuous; categorical or
        binary variables must be filtered out upstream (use ``contrasts()``
        instead). Patsy/formulaic re-evaluates every term that depends on
        this column.

    aggregate, weights, phi_inv, offset, transform : as in make_prediction_estimand.

    fd_step : float, default 1e-6
        Relative FD step. The actual perturbation per row is
        ``fd_step * max(1, |v_i|)`` so it stays well-conditioned for both
        small (e.g., probabilities) and large (e.g., income) magnitudes.

    Returns
    -------
    h : callable (beta) -> scalar or vector
    """
    from ._tabular import as_tabular, to_pandas_if_needed

    df = as_tabular(df)

    if var_name not in df.columns:
        raise ValueError(
            f"Variable {var_name!r} not in df.columns: {list(df.columns)}"
        )

    v = np.asarray(df[var_name], dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError(
            f"Variable {var_name!r} has non-numeric or NaN values; "
            "dydx() is only defined for finite numeric columns."
        )

    if fd_step <= 0:
        raise ValueError(f"fd_step must be positive, got {fd_step}")

    eps = fd_step * np.maximum(1.0, np.abs(v))  # shape (n_rows,)

    df_plus = df.with_column(var_name, v + eps)
    df_minus = df.with_column(var_name, v - eps)

    Xp = adapter.design_matrix_from_df(to_pandas_if_needed(df_plus))
    Xm = adapter.design_matrix_from_df(to_pandas_if_needed(df_minus))
    eps_jax = jnp.asarray(eps)

    return partial(
        slope_kernel,
        predict_fn=adapter.predict,
        Xp=Xp,
        Xm=Xm,
        offset=offset,
        transform=transform,
        aggregate=aggregate,
        weights=weights,
        phi_inv=phi_inv,
        eps_jax=eps_jax,
    )


def slope_kernel(
    beta,
    predict_fn,
    Xp,
    Xm,
    offset,
    transform,
    aggregate,
    weights,
    phi_inv,
    eps_jax,
):
    """Module-level kernel for slope estimands."""
    mu_p = predict_fn(beta, Xp, offset=offset)
    mu_m = predict_fn(beta, Xm, offset=offset)
    if mu_p.ndim > 1:
        slopes = (mu_p - mu_m) / (2.0 * eps_jax[:, None])
    else:
        slopes = (mu_p - mu_m) / (2.0 * eps_jax)
    if transform is not None:
        slopes = transform(slopes)
    value = _aggregate(slopes, aggregate, weights)
    if phi_inv is not None:
        value = phi_inv(value)
    return value

slope_kernel.__pymargins_kernel__ = True  # type: ignore[attr-defined]


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
    offsets = scenario_offsets if scenario_offsets is not None else [None] * n_scenarios
    sw = scenario_weights if scenario_weights is not None else [None] * n_scenarios

    return partial(
        linear_combination_kernel,
        predict_fn=adapter.predict,
        scenarios_X=scenarios_X,
        offsets=offsets,
        sw=sw,
        weights=weights,
        phi_inv=phi_inv,
        scenario_aggregate=scenario_aggregate,
    )


def per_scenario_kernel(beta, predict_fn, X, offset, w, scenario_aggregate):
    """Module-level kernel for a single scenario's aggregated prediction."""
    mu = predict_fn(beta, X, offset=offset)
    if scenario_aggregate in ("overall", "weighted"):
        if w is None:
            return jnp.mean(mu, axis=0) if mu.ndim > 1 else jnp.mean(mu)
        if not jnp.all(jnp.isfinite(w)):
            raise ValueError("scenario_weights must be finite (no NaN or Inf)")
        if jnp.any(w < 0):
            raise ValueError("scenario_weights must be non-negative")
        if jnp.sum(w) == 0:
            raise ValueError("scenario_weights must not sum to zero")
        return jnp.sum(w * mu, axis=0) / jnp.sum(w) if mu.ndim > 1 else jnp.sum(w * mu) / jnp.sum(w)
    elif scenario_aggregate == "none":
        return mu[0] if mu.shape[0] == 1 else mu
    else:
        raise ValueError(f"Unknown scenario_aggregate: {scenario_aggregate!r}")


def linear_combination_kernel(
    beta,
    predict_fn,
    scenarios_X,
    offsets,
    sw,
    weights,
    phi_inv,
    scenario_aggregate,
):
    """Module-level kernel for linear-combination estimands."""
    scenario_values = jnp.stack([
        per_scenario_kernel(beta, predict_fn, scenarios_X[i], offsets[i], sw[i], scenario_aggregate)
        for i in range(len(scenarios_X))
    ])
    if phi_inv is not None:
        scenario_values = phi_inv(scenario_values)
    if isinstance(weights, dict):
        weight_matrix = jnp.stack([
            jnp.asarray(weights[name]) for name in weights
        ])
        return weight_matrix @ scenario_values
    return jnp.dot(jnp.asarray(weights), scenario_values)

linear_combination_kernel.__pymargins_kernel__ = True  # type: ignore[attr-defined]


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
    offsets = scenario_offsets if scenario_offsets is not None else [None] * n_scenarios
    sw = scenario_weights if scenario_weights is not None else [None] * n_scenarios

    return partial(
        evaluate_kernel,
        predict_fn=adapter.predict,
        scenarios_X=scenarios_X,
        offsets=offsets,
        sw=sw,
        compose=compose,
        phi_inv=phi_inv,
        scenario_aggregate=scenario_aggregate,
    )


def evaluate_kernel(
    beta,
    predict_fn,
    scenarios_X,
    offsets,
    sw,
    compose,
    phi_inv,
    scenario_aggregate,
):
    """Module-level kernel for arbitrary nonlinear composition estimands."""
    scenario_values = jnp.stack([
        per_scenario_kernel(beta, predict_fn, scenarios_X[i], offsets[i], sw[i], scenario_aggregate)
        for i in range(len(scenarios_X))
    ])
    result = compose(scenario_values)
    if phi_inv is not None:
        result = phi_inv(result)
    return result

evaluate_kernel.__pymargins_kernel__ = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Differentiability check
# ---------------------------------------------------------------------------

def is_jax_differentiable(h: Callable, beta: jnp.ndarray) -> bool:
    """Test whether an estimand function is JAX-differentiable at beta.

    Used by the inference engine to decide whether delta-method inference
    is possible. If h contains operations JAX can't trace (Python
    conditionals on values, NumPy ops without conversion, etc.), this
    returns False and the engine falls back to simulation or bootstrap.

    The probe mirrors the trace patterns the engine actually uses:
    ``jax.vmap`` (for simulation draws) and ``jax.hessian`` (for the κ
    diagnostic). A function may pass single-point ``jax.grad`` while still
    failing under vmap-batched tracing — for example a Python ``if`` on a
    component of ``b`` evaluates to a concrete boolean at scalar inputs but
    raises ``TracerBoolConversionError`` under vmap.

    Parameters
    ----------
    h : callable
        Estimand function.

    beta : jax array
        Test point.

    Returns
    -------
    differentiable : bool
        True if both ``jax.vmap(h)`` and ``jax.hessian(h)`` (or
        ``jax.jacobian`` for vector estimands) succeed at ``beta``.
    """
    import jax
    out = h(beta)
    # Narrow to JAX tracer/concretization errors. Avoid catching bare
    # Exception so genuine programming errors in estimand factories
    # propagate rather than being silently treated as non-differentiable.
    jax_tracer_errors: tuple = (TypeError,)
    if hasattr(jax, "errors"):
        tracer_errs = []
        for name in (
            "TracerIntegerConversionError",
            "TracerArrayConversionError",
            "TracerBoolConversionError",
            "TracerTupleConversionError",
            "ConcretizationTypeError",
            "UnexpectedTracerError",
        ):
            if hasattr(jax.errors, name):
                tracer_errs.append(getattr(jax.errors, name))
        if tracer_errs:
            jax_tracer_errors = tuple(tracer_errs)
    try:
        # Probe vmap (used by _run_simulation and delta_simulation_disagreement).
        jax.vmap(h)(jnp.stack([beta, beta]))
        # Probe second-order (used by κ via _kappa.kappa → jax.hessian).
        if jnp.ndim(out) == 0:
            jax.hessian(h)(beta)
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
