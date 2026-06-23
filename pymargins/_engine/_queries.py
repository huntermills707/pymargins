"""Query construction: spec -> estimand.

Design \u00a74.2/\u00a74.8, req \u00a72. Added in 0.4.0 (R2).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from pymargins._adapter import ModelAdapter
from pymargins._estimands import (
    make_evaluate_estimand,
    make_linear_combination_estimand,
    make_prediction_estimand,
    make_slope_estimand,
)
from pymargins._inference import InferenceConfig
from pymargins._scenarios import (
    _auto_label_from_atexog,
    expand_scenario,
    make_aggregation_resolver,
)
from pymargins._soundness._predicates import CompileError
from pymargins._tabular import to_pandas_if_needed


@dataclass(frozen=True)
class QueryContext:
    """Everything query construction may read. No session anywhere."""

    adapter: ModelAdapter
    base_data: Any  # wiring point-execution output (post-match/trim)
    at: str
    weights: np.ndarray | None
    phi: Callable | None
    phi_inv: Callable | None
    fd_step: float
    gradient_backend: str


@dataclass(frozen=True)
class QuerySpec:
    """A user-facing query request."""

    kind: str
    scenario: Mapping | None = None
    variables: tuple[str, ...] | None = None
    scenarios: tuple[Mapping, ...] | None = None
    contrast_weights: Any | None = None
    compose: Callable | None = None
    transform: Callable | None = None
    label: str | None = None
    outcome: int | tuple[int, ...] | None = None
    horizon: float | None = None
    n_grid: int = 80


@dataclass(frozen=True)
class CompiledQuery:
    """A compiled estimand ready for the executor."""

    h: Callable
    h_factory: Callable | None
    labels: list[str] | None
    scenarios: list[dict]
    estimand_metadata: dict


@dataclass(frozen=True)
class WiringFacts:
    """Dependence and pipeline facts extracted from a wiring graph.

    This dataclass is owned by the query layer in R2; R5 moves the
    extraction logic into ``_graph/_compile.py`` but keeps the type here
    so that ``build_inference_config`` has a stable contract.
    """

    design: Any | None = None
    cluster: Any | None = None
    block: int | None = None
    block_type: str = "moving"
    matching: Any | None = None
    transforms: list | None = None
    population_note: str | None = None


# ---------------------------------------------------------------------------
# Scale resolution
# ---------------------------------------------------------------------------


def resolve_scale(scale: str | tuple | None) -> tuple[Callable | None, Callable | None]:
    """Return (phi, phi_inv) for a named or explicit scale.

    Named scales:
      - "response", "identity" -> (None, None)
      - "log"                  -> (jnp.exp, jnp.log)
      - "logit"                -> (expit, logit)
      - "probit"               -> (ndtr, ndtri)

    An explicit tuple (phi, phi_inv) passes through unchanged.
    """
    if scale is None or scale == "response" or scale == "identity":
        return None, None
    if isinstance(scale, tuple) and len(scale) == 2:
        return scale
    if scale == "log":
        return jnp.exp, jnp.log
    if scale == "logit":
        from jax.scipy.special import expit, logit

        return expit, logit
    if scale == "probit":
        from jax.scipy.special import ndtr, ndtri

        return ndtr, ndtri
    raise CompileError(
        f"Unknown scale: {scale!r}. Supported: response, identity, log, logit, probit."
    )


# ---------------------------------------------------------------------------
# Bootstrap-weight subsetting
# ---------------------------------------------------------------------------


def _bootstrap_weights_for_adapter(ctx: QueryContext, adapter: ModelAdapter | None = None):
    """Return session weights, subsetted by bootstrap resample index if needed."""
    weights = ctx.weights
    adapter = adapter if adapter is not None else ctx.adapter
    if weights is not None and hasattr(adapter, "_pymargins_bootstrap_idx"):
        idx = adapter._pymargins_bootstrap_idx
        if idx is not None:
            weights = np.asarray(weights)[idx]
    return weights


# ---------------------------------------------------------------------------
# Scenario adapter (per-scenario prediction_time)
# ---------------------------------------------------------------------------


def _scenario_adapter(adapter: ModelAdapter, scenario: dict):
    """Return a (possibly cloned) adapter parameterized for one scenario."""
    t = scenario.get("prediction_time")
    if t is None:
        return adapter
    if not hasattr(adapter, "with_prediction_time"):
        raise ValueError(
            f"Scenario carries 'prediction_time' but adapter "
            f"{type(adapter).__name__} does not support per-scenario time. "
            "Use a time-aware adapter or remove 'prediction_time' from the scenario."
        )
    return adapter.with_prediction_time(t)


# ---------------------------------------------------------------------------
# Atom helpers (ported from margins/_atoms.py, session-free)
# ---------------------------------------------------------------------------


def _enumerate_groups(
    scenario: dict,
    base_data,
    variable_metadata: dict,
    *,
    track_positions: bool = False,
):
    """Resolve ``scenario['over']`` into (group_label, df, positions) triples.

    ``positions`` is the positional row index of each group within
    ``base_data``, computed only when ``track_positions`` is set — it is
    what lets per-observation aggregation weights be subset per over-group
    (ledger D16). Positional (not label) indexing keeps the mapping correct
    for bootstrap-resampled frames with duplicate index labels.
    """
    over_spec = scenario.get("over")
    if over_spec is None:
        return [(None, base_data, None)], None
    over_keys = [over_spec] if isinstance(over_spec, str) else list(over_spec)
    unknown = set(over_keys) - set(variable_metadata.keys())
    if unknown:
        raise ValueError(
            f"Unknown over variable(s): {sorted(unknown)}. "
            f"Known variables: {sorted(variable_metadata.keys())}."
        )
    if not hasattr(base_data, "groupby"):
        raise TypeError(
            f"over= requires base_data to support groupby, got {type(base_data).__name__}"
        )
    if track_positions:
        df = to_pandas_if_needed(base_data)
        tmp = df.assign(__pymargins_pos=np.arange(len(df)))
        groups = [
            (g, gdf.drop(columns="__pymargins_pos"), gdf["__pymargins_pos"].to_numpy())
            for g, gdf in tmp.groupby(over_keys, sort=True)
        ]
    else:
        groups = [(g, gdf, None) for g, gdf in base_data.groupby(over_keys, sort=True)]
    if not groups:
        raise ValueError(
            f"over={over_keys!r} produced no groups; base data may be empty."
        )
    return groups, over_keys


def _format_atom_label(
    group_label,
    over_keys: list[str] | None,
    suffix: str | None,
) -> str | None:
    """Build a stable label for one atom of a stacked estimand."""
    parts: list[str] = []
    if over_keys is not None:
        gl = group_label if isinstance(group_label, tuple) else (group_label,)
        parts.extend(f"{k}={v}" for k, v in zip(over_keys, gl, strict=False))
    if suffix is not None:
        parts.append(suffix)
    return ", ".join(parts) if parts else None


def _finalize_atoms(atoms: list[tuple[str | None, Callable]]):
    """Reduce a list of (label, h_atom) pairs to (h, labels)."""
    if len(atoms) == 1:
        label = atoms[0][0]
        return atoms[0][1], ([label] if label is not None else None)
    individual_h = [h for _, h in atoms]
    labels = [lab for lab, _ in atoms]

    def h_vector(beta):
        return jnp.stack([hi(beta) for hi in individual_h])

    return h_vector, labels


# ---------------------------------------------------------------------------
# Refit context + h_factory helper
# ---------------------------------------------------------------------------


def _refit_context(ctx: QueryContext, new_adapter: ModelAdapter) -> QueryContext:
    """Build a context for a refit (bootstrap) adapter."""
    return QueryContext(
        adapter=new_adapter,
        base_data=new_adapter.training_data,
        at=ctx.at,
        weights=ctx.weights,
        phi=ctx.phi,
        phi_inv=ctx.phi_inv,
        fd_step=ctx.fd_step,
        gradient_backend=ctx.gradient_backend,
    )


def _h_factory_for(spec: QuerySpec, ctx: QueryContext) -> Callable:
    """Return h_factory(new_adapter) -> h for bootstrap re-execution."""

    def h_factory(new_adapter: ModelAdapter) -> Callable:
        return compile_query(spec, _refit_context(ctx, new_adapter)).h

    return h_factory


# ---------------------------------------------------------------------------
# Per-kind builders
# ---------------------------------------------------------------------------


def _set_outcome_shape(meta: dict, n_atoms: int, adapter: ModelAdapter) -> None:
    """Record multi-outcome layout when the adapter exposes it."""
    n_outcomes = getattr(adapter, "n_outcomes", 1)
    if n_outcomes > 1:
        labels = getattr(adapter, "outcome_labels", None)
        if labels is None:
            labels = [str(i) for i in range(n_outcomes)]
        meta["_outcome_shape"] = {
            "n_atoms": n_atoms,
            "n_outcomes": n_outcomes,
            "outcome_labels": list(labels),
        }


def _build_prediction_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Construct the prediction estimand for predict() calls."""
    adapter = ctx.adapter
    estimand_adapter = _scenario_adapter(adapter, dict(spec.scenario or {}))
    base_data = ctx.base_data
    var_meta = adapter.variable_metadata()
    weights = _bootstrap_weights_for_adapter(ctx)

    scenario = dict(spec.scenario or {})
    groups, over_keys = _enumerate_groups(
        scenario, base_data, var_meta, track_positions=weights is not None
    )

    sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
    atoms: list[tuple[str | None, Callable]] = []
    scenarios: list[dict] = []

    for group_label, group_df, group_pos in groups:
        # D16: aggregation weights are per-row of base_data; subset them to
        # the group's rows so the group estimand is a weighted mean over the
        # group only (marginaleffects by=+wts= / Stata over() [pw=] semantics).
        w_g = (
            np.asarray(weights)[group_pos]
            if weights is not None and group_pos is not None
            else weights
        )
        resolver = make_aggregation_resolver(ctx.at, w_g)
        df, meta = expand_scenario(sub_scenario, group_df, resolver, var_meta)
        X = adapter.design_matrix_from_df(to_pandas_if_needed(df))
        n_grid = meta.get("n_grid_points", 1)
        rows_per = meta.get("rows_per_grid_point", len(df))
        if n_grid > 1 and X.shape[0] != n_grid * rows_per:
            raise ValueError(
                f"Design matrix rows ({X.shape[0]}) do not match expected grid layout "
                f"({n_grid} x {rows_per} = {n_grid * rows_per})."
            )

        for i in range(n_grid):
            start = i * rows_per
            end = (i + 1) * rows_per
            if end > X.shape[0]:
                raise ValueError(f"Grid block {i} would exceed design matrix rows.")
            X_i = X[start:end]
            agg_kind = "overall" if ctx.at == "overall" else ("none" if X_i.shape[0] == 1 else "overall")
            h_atom = make_prediction_estimand(
                estimand_adapter,
                X_i,
                aggregate=agg_kind,
                weights=jnp.asarray(w_g) if w_g is not None else None,
                phi_inv=ctx.phi_inv,
                transform=spec.transform,
            )
            if n_grid > 1:
                grid_rows = meta.get("grid_rows", [])
                grid_keys = meta.get("atexog_keys", [])
                grid_row = grid_rows[i] if i < len(grid_rows) else ()
                if grid_row and grid_keys:
                    grid_suffix = ", ".join(
                        f"{k}={v}" for k, v in zip(grid_keys, grid_row, strict=False)
                    )
                else:
                    grid_suffix = (
                        _auto_label_from_atexog(sub_scenario.get("atexog"))
                        or f"grid[{i}]"
                    )
            else:
                grid_suffix = _auto_label_from_atexog(sub_scenario.get("atexog"))
            label = _format_atom_label(group_label, over_keys, grid_suffix)
            atoms.append((label, h_atom))

            scen: dict = {}
            if over_keys is not None:
                gl = group_label if isinstance(group_label, tuple) else (group_label,)
                for k, v in zip(over_keys, gl, strict=False):
                    scen[k] = v
                scen["_over_values"] = {ok: gl[i] for i, ok in enumerate(over_keys)}
            if n_grid > 1:
                grid_rows = meta.get("grid_rows", [])
                grid_keys = meta.get("atexog_keys", [])
                grid_row = grid_rows[i] if i < len(grid_rows) else ()
                for k, v in zip(grid_keys, grid_row, strict=False):
                    scen[k] = v
            else:
                atexog = sub_scenario.get("atexog", {})
                if atexog:
                    for k, v in atexog.items():
                        if not isinstance(v, list):
                            scen[k] = v
            scenarios.append(scen)

    h, labels = _finalize_atoms(atoms)
    if spec.label is not None:
        if labels is None or len(labels) == 1:
            labels = [spec.label]

    meta = {"kind": "prediction", "at": ctx.at}
    if labels is not None:
        meta["labels"] = labels
    if scenarios:
        meta["scenarios"] = scenarios
    if scenario.get("over") is not None:
        over = scenario["over"]
        meta["over"] = [over] if isinstance(over, str) else list(over)
    _over_values = [s.get("_over_values") for s in scenarios]
    if any(v is not None for v in _over_values):
        meta["_over_values"] = _over_values
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome
    _set_outcome_shape(meta, len(atoms), adapter)

    return CompiledQuery(
        h=h,
        h_factory=_h_factory_for(spec, ctx),
        labels=labels,
        scenarios=scenarios,
        estimand_metadata=meta,
    )


def _build_slope_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Construct the slope estimand for dydx() calls."""
    adapter = ctx.adapter
    estimand_adapter = _scenario_adapter(adapter, dict(spec.scenario or {}))
    base_data = ctx.base_data
    var_meta = adapter.variable_metadata()
    weights = _bootstrap_weights_for_adapter(ctx)

    scenario = dict(spec.scenario or {})
    var_list = list(spec.variables or ())
    if not var_list:
        raise ValueError("dydx() requires at least one variable")

    for v in var_list:
        adapter.column_index_of_variable(v)

    groups, over_keys = _enumerate_groups(
        scenario, base_data, var_meta, track_positions=weights is not None
    )
    sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
    atoms: list[tuple[str | None, Callable]] = []
    scenarios: list[dict] = []

    for group_label, group_df, group_pos in groups:
        # D16: subset per-row aggregation weights to the group's rows.
        w_g = (
            np.asarray(weights)[group_pos]
            if weights is not None and group_pos is not None
            else weights
        )
        resolver = make_aggregation_resolver(ctx.at, w_g)
        df, meta = expand_scenario(sub_scenario, group_df, resolver, var_meta)
        agg_kind = "overall" if ctx.at == "overall" else ("none" if len(df) == 1 else "overall")

        base_scen: dict = {}
        if over_keys is not None:
            gl = group_label if isinstance(group_label, tuple) else (group_label,)
            for k, v in zip(over_keys, gl, strict=False):
                base_scen[k] = v
            base_scen["_over_values"] = {ok: gl[i] for i, ok in enumerate(over_keys)}
        atexog = sub_scenario.get("atexog", {})
        if atexog and meta.get("n_grid_points", 1) == 1:
            for k, v in atexog.items():
                if not isinstance(v, list):
                    base_scen[k] = v

        for var_name in var_list:
            h_atom = make_slope_estimand(
                estimand_adapter,
                df,
                var_name,
                aggregate=agg_kind,
                weights=jnp.asarray(w_g) if w_g is not None else None,
                phi_inv=ctx.phi_inv,
                transform=spec.transform,
                fd_step=ctx.fd_step,
            )
            atexog_label = _auto_label_from_atexog(sub_scenario.get("atexog"))
            suffix = f"{atexog_label}, {var_name}" if atexog_label else var_name
            label = _format_atom_label(group_label, over_keys, suffix)
            atoms.append((label, h_atom))
            scenarios.append(base_scen.copy())

    h, labels = _finalize_atoms(atoms)
    if spec.label is not None:
        if labels is None or len(labels) == 1:
            labels = [spec.label]

    meta = {"kind": "slope", "variables": var_list, "at": ctx.at}
    if labels is not None:
        meta["labels"] = labels
    if scenarios:
        meta["scenarios"] = scenarios
    if scenario.get("over") is not None:
        over = scenario["over"]
        meta["over"] = [over] if isinstance(over, str) else list(over)
    _over_values = [s.get("_over_values") for s in scenarios]
    if any(v is not None for v in _over_values):
        meta["_over_values"] = _over_values
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome
    _set_outcome_shape(meta, len(atoms), adapter)

    return CompiledQuery(
        h=h,
        h_factory=_h_factory_for(spec, ctx),
        labels=labels,
        scenarios=scenarios,
        estimand_metadata=meta,
    )


def _normalize_contrast_weights(weights_arg, scenarios):
    """Normalize the contrasts argument into the dict-or-vector forms accepted
    by ``make_linear_combination_estimand`` (legacy session normalization,
    ported per ledger D18)."""
    if weights_arg is None:
        raise ValueError("contrasts() requires contrast_weights")
    if isinstance(weights_arg, dict):
        weights_arg = {name: jnp.asarray(w) for name, w in weights_arg.items()}
        labels = list(weights_arg.keys())
    elif isinstance(weights_arg, (np.ndarray, jnp.ndarray)) and weights_arg.ndim == 2:
        weights_arg = {
            f"contrast[{i}]": jnp.asarray(weights_arg[i])
            for i in range(weights_arg.shape[0])
        }
        labels = list(weights_arg.keys())
    elif (
        isinstance(weights_arg, list) and weights_arg and isinstance(weights_arg[0], list)
    ):
        contrasts_arr = jnp.asarray(weights_arg)
        if contrasts_arr.ndim != 2:
            raise ValueError(
                f"list-of-lists contrast must be 2D after conversion, got {contrasts_arr.ndim}D"
            )
        weights_arg = {
            f"contrast[{i}]": contrasts_arr[i]
            for i in range(contrasts_arr.shape[0])
        }
        labels = list(weights_arg.keys())
    else:
        weights_arg = jnp.asarray(weights_arg)
        labels = [scenarios[0].get("label", "contrast")]
    return weights_arg, labels


def _validate_contrast_weights(weights_arg, n_scenarios: int) -> None:
    """Length and finiteness validation for normalized contrast weights."""
    _weights_to_check = []
    if isinstance(weights_arg, dict):
        for name, w in weights_arg.items():
            if w.shape[0] != n_scenarios:
                raise ValueError(
                    f"Contrast {name!r} has {w.shape[0]} weights but "
                    f"{n_scenarios} scenarios were provided."
                )
            _weights_to_check.append(w)
    else:
        if weights_arg.shape[0] != n_scenarios:
            raise ValueError(
                f"Contrast has {weights_arg.shape[0]} weights but "
                f"{n_scenarios} scenarios were provided."
            )
        _weights_to_check.append(weights_arg)
    for w in _weights_to_check:
        if not jnp.all(jnp.isfinite(w)):
            raise ValueError("Contrast weights must be finite (no NaN or Inf)")


def _scenario_weights_for(weights, scenarios_X):
    """Per-scenario aggregation weights for contrast/evaluate estimands (D17).

    Declared weights align row-wise with each scenario's expanded design
    matrix. A single-row scenario (at="mean"/"typical") aggregates
    trivially, so it carries None; any other row-count mismatch cannot be
    weighted coherently and refuses.

    Validation happens here, at query construction time, so the kernel itself
    can remain JAX-traceable.
    """
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite (no NaN or Inf)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(w) == 0:
        raise ValueError("weights must not sum to zero")
    out = []
    for i, X in enumerate(scenarios_X):
        n_rows = X.shape[0]
        if n_rows == w.shape[0]:
            out.append(w)
        elif n_rows == 1:
            out.append(None)
        else:
            raise ValueError(
                f"Scenario {i} expands to {n_rows} rows but weights= has length "
                f"{w.shape[0]}. Weighted aggregation requires the scenario rows "
                "to align with the declared per-observation weights; "
                "data-override or grid scenarios with a different row count are "
                "not compatible with weights=."
            )
    return out


def _build_contrast_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Construct a linear combination estimand for contrasts() calls."""
    adapter = ctx.adapter
    base_data = ctx.base_data
    scenarios = list(spec.scenarios or [])
    if len(scenarios) == 0:
        raise ValueError("contrasts() requires at least one scenario")

    # Scenario type check runs before normalization: the vector-contrast
    # label default reads scenarios[0], which must already be a dict.
    for i, s in enumerate(scenarios):
        if not isinstance(s, dict):
            raise TypeError(
                f"Each scenario must be a dict, got {type(s).__name__} at index {i}"
            )

    weights_arg, labels = _normalize_contrast_weights(spec.contrast_weights, scenarios)
    _validate_contrast_weights(weights_arg, len(scenarios))

    weights = _bootstrap_weights_for_adapter(ctx)
    scenarios_X = []
    scenario_predict_fns = []
    any_per_scenario_predict = False
    for scenario in scenarios:
        df, _ = expand_scenario(
            scenario,
            base_data=base_data,
            aggregation_resolver=make_aggregation_resolver(ctx.at, weights),
            variable_metadata=adapter.variable_metadata(),
        )
        scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))
        scen_adapter = _scenario_adapter(adapter, scenario)
        if scen_adapter is not adapter:
            any_per_scenario_predict = True
        scenario_predict_fns.append(scen_adapter.predict)

    h = make_linear_combination_estimand(
        adapter,
        scenarios_X=scenarios_X,
        weights=weights_arg,
        phi_inv=ctx.phi_inv,
        scenario_weights=_scenario_weights_for(weights, scenarios_X),
        scenario_predict_fns=scenario_predict_fns if any_per_scenario_predict else None,
    )

    if spec.label is not None:
        if len(labels) == 1:
            labels = [spec.label]

    meta = {
        "kind": "contrasts",
        "labels": labels,
        "scenarios": scenarios,
        "at": ctx.at,
    }
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome
    n_atoms = len(weights_arg) if isinstance(weights_arg, dict) else 1
    _set_outcome_shape(meta, n_atoms, adapter)

    return CompiledQuery(
        h=h,
        h_factory=_h_factory_for(spec, ctx),
        labels=labels,
        scenarios=scenarios,
        estimand_metadata=meta,
    )


def _build_evaluate_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Construct an arbitrary composition estimand for evaluate() calls."""
    adapter = ctx.adapter
    base_data = ctx.base_data
    scenarios = list(spec.scenarios or [])
    compose = spec.compose
    if compose is None:
        raise ValueError("evaluate() requires compose")
    if not callable(compose):
        raise TypeError(f"compose must be callable, got {type(compose).__name__}")

    weights = _bootstrap_weights_for_adapter(ctx)
    scenarios_X = []
    scenario_predict_fns = []
    any_per_scenario_predict = False
    for scenario in scenarios:
        df, _ = expand_scenario(
            scenario,
            base_data=base_data,
            aggregation_resolver=make_aggregation_resolver(ctx.at, weights),
            variable_metadata=adapter.variable_metadata(),
        )
        scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))
        scen_adapter = _scenario_adapter(adapter, scenario)
        if scen_adapter is not adapter:
            any_per_scenario_predict = True
        scenario_predict_fns.append(scen_adapter.predict)

    h = make_evaluate_estimand(
        adapter,
        scenarios_X=scenarios_X,
        compose=compose,
        phi_inv=ctx.phi_inv,
        scenario_weights=_scenario_weights_for(weights, scenarios_X),
        scenario_predict_fns=scenario_predict_fns if any_per_scenario_predict else None,
    )

    labels = [
        s.get("label", _auto_label_from_atexog(s.get("atexog")) or f"scenario[{i}]")
        for i, s in enumerate(scenarios)
    ]
    if spec.label is not None:
        if len(labels) == 1:
            labels = [spec.label]

    meta = {
        "kind": "evaluate",
        "labels": labels,
        "scenarios": scenarios,
        "at": ctx.at,
    }
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome
    _set_outcome_shape(meta, 1, adapter)

    return CompiledQuery(
        h=h,
        h_factory=_h_factory_for(spec, ctx),
        labels=labels,
        scenarios=scenarios,
        estimand_metadata=meta,
    )


def _build_elasticity_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Build eyex/eydx/dyex as a composed estimand."""
    kind = spec.kind
    if kind not in ("eyex", "eydx", "dyex"):
        raise ValueError(f"Unknown elasticity kind: {kind!r}")
    variable = (spec.variables or (None,))[0]
    if variable is None:
        raise ValueError(f"{kind}() requires a single variable")

    x_bar = float(np.asarray(ctx.base_data[variable]).mean())
    clip_near_zero = 1e-12

    def fn(slope, pred):
        if kind == "eyex":
            return slope * x_bar / jnp.where(
                jnp.abs(pred) < clip_near_zero,
                jnp.sign(pred) * clip_near_zero,
                pred,
            )
        if kind == "eydx":
            return slope / jnp.where(
                jnp.abs(pred) < clip_near_zero,
                jnp.sign(pred) * clip_near_zero,
                pred,
            )
        return slope * x_bar

    scenario = dict(spec.scenario or {})
    slope_spec = QuerySpec(
        kind="dydx",
        scenario=scenario,
        variables=(variable,),
        transform=spec.transform,
    )
    pred_spec = QuerySpec(
        kind="predict",
        scenario=scenario,
        transform=spec.transform,
    )
    slope_cq = _build_slope_query(slope_spec, ctx)
    pred_cq = _build_prediction_query(pred_spec, ctx)

    def h(beta):
        return fn(slope_cq.h(beta), pred_cq.h(beta))

    def h_factory(new_adapter: ModelAdapter) -> Callable:
        new_ctx = _refit_context(ctx, new_adapter)
        new_slope_cq = _build_slope_query(slope_spec, new_ctx)
        new_pred_cq = _build_prediction_query(pred_spec, new_ctx)

        def h_refit(beta):
            return fn(new_slope_cq.h(beta), new_pred_cq.h(beta))

        return h_refit

    labels = [spec.label or f"{kind}({variable})"]
    meta = {
        "kind": "elasticity",
        "subkind": kind,
        "variable": variable,
        "at": ctx.at,
        "labels": labels,
    }
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome

    return CompiledQuery(
        h=h,
        h_factory=h_factory,
        labels=labels,
        scenarios=slope_cq.scenarios,
        estimand_metadata=meta,
    )


def _build_wtp_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Willingness to pay: -(dU/dattribute) / (dU/dprice)."""
    variables = spec.variables
    if variables is None or len(variables) != 2:
        raise ValueError("wtp() requires exactly two variables: attribute, price")
    attribute, price = variables
    scenario = dict(spec.scenario or {})

    def fn(slope_attr, slope_price):
        return -slope_attr / slope_price

    attr_spec = QuerySpec(
        kind="dydx",
        scenario=scenario,
        variables=(attribute,),
        transform=spec.transform,
    )
    price_spec = QuerySpec(
        kind="dydx",
        scenario=scenario,
        variables=(price,),
        transform=spec.transform,
    )
    attr_cq = _build_slope_query(attr_spec, ctx)
    price_cq = _build_slope_query(price_spec, ctx)

    def h(beta):
        return fn(attr_cq.h(beta), price_cq.h(beta))

    def h_factory(new_adapter: ModelAdapter) -> Callable:
        new_ctx = _refit_context(ctx, new_adapter)
        new_attr_cq = _build_slope_query(attr_spec, new_ctx)
        new_price_cq = _build_slope_query(price_spec, new_ctx)

        def h_refit(beta):
            return fn(new_attr_cq.h(beta), new_price_cq.h(beta))

        return h_refit

    labels = [spec.label or f"WTP({attribute})"]
    meta = {
        "kind": "wtp",
        "attribute": attribute,
        "price": price,
        "at": ctx.at,
        "labels": labels,
    }
    if spec.outcome is not None:
        meta["outcome"] = spec.outcome

    return CompiledQuery(
        h=h,
        h_factory=h_factory,
        labels=labels,
        scenarios=attr_cq.scenarios,
        estimand_metadata=meta,
    )


def _build_rmst_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Restricted mean survival time via trapezoidal integration."""
    adapter = ctx.adapter
    if not hasattr(adapter, "with_prediction_time"):
        raise ValueError(
            f"Adapter {type(adapter).__name__} does not support "
            f"per-scenario prediction time; rmst() requires a time-aware survival adapter."
        )
    horizon = spec.horizon
    if horizon is None:
        raise ValueError("rmst() requires horizon")
    n_grid = spec.n_grid
    times = np.linspace(0.0, float(horizon), int(n_grid))
    scenario = dict(spec.scenario or {})
    scenarios = [
        {**scenario, "prediction_time": float(t)} for t in times
    ]
    times_jax = jnp.asarray(times)

    def compose(surv):
        dt = jnp.diff(times_jax)
        return jnp.sum(0.5 * (surv[:-1] + surv[1:]) * dt)

    eval_spec = QuerySpec(
        kind="evaluate",
        scenarios=tuple(scenarios),
        compose=compose,
    )
    return _build_evaluate_query(eval_spec, ctx)


# ---------------------------------------------------------------------------
# Public compile dispatcher
# ---------------------------------------------------------------------------


def compile_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery:
    """Compile a QuerySpec into a CompiledQuery."""
    kind = spec.kind
    if kind == "predict":
        cq = _build_prediction_query(spec, ctx)
    elif kind == "dydx":
        cq = _build_slope_query(spec, ctx)
    elif kind in ("eyex", "eydx", "dyex"):
        cq = _build_elasticity_query(spec, ctx)
    elif kind == "contrasts":
        cq = _build_contrast_query(spec, ctx)
    elif kind == "evaluate":
        cq = _build_evaluate_query(spec, ctx)
    elif kind == "wtp":
        cq = _build_wtp_query(spec, ctx)
    elif kind == "rmst":
        cq = _build_rmst_query(spec, ctx)
    else:
        raise CompileError(
            f"Unknown query kind: {spec.kind!r}. Supported: predict, dydx, "
            "eyex, eydx, dyex, contrasts, evaluate, wtp, rmst."
        )

    if spec.label is not None and cq.labels is not None and len(cq.labels) > 1:
        warnings.warn(
            f"label={spec.label!r} is ignored when atexog or over "
            "produces multiple estimands",
            UserWarning,
            stacklevel=2,
        )
    return cq


# ---------------------------------------------------------------------------
# InferenceConfig builder
# ---------------------------------------------------------------------------


def _resolve_vcov_spec(plan, wiring_facts: WiringFacts):
    """Replicate the legacy vcov_spec resolution rule."""
    if plan.vcov is not None:
        return plan.vcov
    if wiring_facts.cluster is not None:
        return {"type": "cluster", "groups": wiring_facts.cluster}
    if wiring_facts.design is not None:
        return {"type": "survey", "design": wiring_facts.design}
    return None


def build_inference_config(
    plan,
    adapter: ModelAdapter,
    wiring_facts: WiringFacts,
    banks,  # noqa: ARG001
    *,
    n_jobs: int = 1,
    progress_bar: bool = False,
    frozen_cov: np.ndarray | None = None,
    phi: Any | None = None,
    phi_inv: Any | None = None,
) -> InferenceConfig:
    """Build a doctrine-shaped InferenceConfig from a Plan.

    Banks are accepted for API symmetry but injection happens in the
    executor (R3); this function leaves ``all_idx``/``all_states``/
    ``all_states_failures``/``sim_draws`` as None.

    ``frozen_cov`` is the estimator's pre-computed Σ̂. When omitted it is
    computed once from the adapter; the executor always passes it so Σ̂
    is resolved exactly once per estimator.
    """
    if phi is None or phi_inv is None:
        phi, phi_inv = resolve_scale(plan.scale)
    if frozen_cov is None:
        vcov_spec = _resolve_vcov_spec(plan, wiring_facts)
        frozen_cov = adapter.covariance(vcov_spec)

    strata = None
    if wiring_facts.design is not None:
        strata = getattr(wiring_facts.design, "strata", None)

    bootstrap_config = None
    if plan.method_resolved == "bootstrap":
        bootstrap_config = {
            "ci_method": plan.ci or "percentile",
            "block_type": wiring_facts.block_type,
        }

    return InferenceConfig(
        method=plan.method_resolved,
        level=plan.level,
        phi=phi,
        phi_inv=phi_inv,
        kappa_threshold=float("inf"),
        gradient_backend=plan.gradient_backend,
        fd_step=plan.fd_step,
        n_sim=plan.n_sim,
        n_boot=plan.B,
        n_jobs=n_jobs,
        rng_seed=plan.seed,
        diagnostics=True,
        cov_params=frozen_cov,
        cluster=wiring_facts.cluster,
        block_size=wiring_facts.block,
        strata=strata,
        survey_design=wiring_facts.design,
        matching=wiring_facts.matching,
        transforms=wiring_facts.transforms,
        bootstrap_config=bootstrap_config,
        all_idx=None,
        all_states=None,
        all_states_failures=None,
        sim_draws=None,
        progress_bar=progress_bar,
    )
