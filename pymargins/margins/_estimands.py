"""Estimand builders for predictions, slopes, contrasts, and evaluate."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from .._adapter import ModelAdapter
from .._estimands import (
    make_evaluate_estimand,
    make_linear_combination_estimand,
    make_prediction_estimand,
    make_slope_estimand,
)
from .._scenarios import (
    _auto_label_from_atexog,
    expand_scenario,
    make_aggregation_resolver,
)


def _bootstrap_weights_for_adapter(session, adapter: ModelAdapter | None = None):
    """Return session weights, subsetted by bootstrap resample index if needed.

    During bootstrap, the adapter may carry the resample index on
    ``_pymargins_bootstrap_idx``.  When present, subset aggregation weights
    so they match the resampled evaluation data.
    """
    weights = session.weights
    adapter = adapter if adapter is not None else session.adapter
    if weights is not None and hasattr(adapter, "_pymargins_bootstrap_idx"):
        idx = adapter._pymargins_bootstrap_idx
        if idx is not None:
            import numpy as np
            weights = np.asarray(weights)[idx]
    return weights


def _get_base_data(session, adapter: ModelAdapter | None = None):
    """Get base data from an adapter, applying matching if active.

    For the original session adapter, ``matching.matched_data`` is used
    when a matching object was provided. For bootstrap replicates, the
    adapter's ``training_data`` already reflects the rematched subset
    (set during ``refit()``), so we return it verbatim.
    """
    adapter = adapter if adapter is not None else session.adapter
    if session.matching is not None and adapter is session.adapter:
        return session.matching.matched_data
    try:
        return adapter.training_data
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"Adapter {type(adapter).__name__} does not expose training_data. "
            "Bootstrap inference and scenario expansion require it."
        ) from exc


def _build_prediction_estimand(
    session,
    scenario: dict,
    transform: Callable | None,
    adapter: ModelAdapter | None = None,
) -> tuple[Callable, list[str] | None, list[dict]]:
    """Construct the prediction estimand for predict() calls.

    Resolves the scenario into a design matrix using the session's
    ``at`` setting, then wraps it in ``make_prediction_estimand`` with
    ``phi_inv`` applied to lift onto the inference scale.

    When the scenario produces multiple atoms (over-stratification, an
    atexog grid, or both), returns a stacked vector estimand with one
    component per (group × grid point) and the corresponding labels.
    Returns ``(h, None)`` for the single-atom case.

    Also returns a list of scenario dicts, one per atom, containing the
    atexog and over values for plot-ready DataFrame construction.
    """
    adapter = adapter if adapter is not None else session.adapter
    # A scalar prediction_time at this level applies to every atom this
    # call produces (groups × grid). Per-scenario times for multi-time
    # curves go through contrasts(scenarios=[...]) instead.
    estimand_adapter = _scenario_adapter(adapter, scenario)
    base_data = _get_base_data(session, adapter)
    var_meta = adapter.variable_metadata()
    weights = _bootstrap_weights_for_adapter(session, adapter)
    resolver = make_aggregation_resolver(session.at, weights)
    from ._atoms import _enumerate_groups, _finalize_atoms, _format_atom_label

    groups, over_keys = _enumerate_groups(session, scenario, base_data, var_meta)

    sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
    atoms: list[tuple[str | None, Callable]] = []
    scenarios: list[dict] = []

    for group_label, group_df in groups:
        df, meta = expand_scenario(
            sub_scenario,
            group_df,
            resolver,
            var_meta,
        )
        from .._tabular import to_pandas_if_needed

        X = adapter.design_matrix_from_df(to_pandas_if_needed(df))
        n_grid = meta.get("n_grid_points", 1)
        rows_per = meta.get("rows_per_grid_point", len(df))
        if n_grid > 1 and X.shape[0] != n_grid * rows_per:
            raise ValueError(
                f"Design matrix rows ({X.shape[0]}) do not match expected grid layout "
                f"({n_grid} × {rows_per} = {n_grid * rows_per}). The adapter may have dropped rows."
            )

        for i in range(n_grid):
            start = i * rows_per
            end = (i + 1) * rows_per
            if end > X.shape[0]:
                raise ValueError(
                    f"Grid block {i} would exceed design matrix rows "
                    f"({end} > {X.shape[0]}). The adapter's "
                    "design_matrix_from_df may have dropped rows."
                )
            X_i = X[start:end]
            if session.at == "overall":
                agg_kind = "overall"
            else:
                agg_kind = "none" if X_i.shape[0] == 1 else "overall"
            h_atom = make_prediction_estimand(
                estimand_adapter,
                X_i,
                aggregate=agg_kind,
                weights=jnp.asarray(weights)
                if weights is not None
                else None,
                phi_inv=session.phi_inv,
                transform=transform,
            )
            if n_grid > 1:
                grid_row = (
                    meta.get("grid_rows", [])[i]
                    if i < len(meta.get("grid_rows", []))
                    else ()
                )
                grid_keys = meta.get("atexog_keys", [])
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
            label = _format_atom_label(session, group_label, over_keys, grid_suffix)
            atoms.append((label, h_atom))

            # Build scenario dict for this atom
            scen = {}
            if over_keys is not None:
                gl = group_label if isinstance(group_label, tuple) else (group_label,)
                for k, v in zip(over_keys, gl, strict=False):
                    scen[k] = v
                scen["_over_values"] = {ok: gl[i] for i, ok in enumerate(over_keys)}
            if n_grid > 1:
                grid_row = (
                    meta.get("grid_rows", [])[i]
                    if i < len(meta.get("grid_rows", []))
                    else ()
                )
                grid_keys = meta.get("atexog_keys", [])
                for k, v in zip(grid_keys, grid_row, strict=False):
                    scen[k] = v
            else:
                atexog = sub_scenario.get("atexog", {})
                if atexog:
                    for k, v in atexog.items():
                        if not isinstance(v, list):
                            scen[k] = v
            scenarios.append(scen)

    h, labels = _finalize_atoms(session, atoms)
    return h, labels, scenarios


def _build_slope_estimand(
    session,
    scenario: dict,
    var_list: list[str],
    transform: Callable | None,
    adapter: ModelAdapter | None = None,
) -> tuple[Callable, list[str] | None, list[dict]]:
    """Construct the slope estimand for dydx() calls.

    Produces one atom per (over-group × variable). With a single
    variable and no ``over``, returns a scalar estimand. Otherwise
    returns a stacked vector estimand with one component per atom.

    Also returns a list of scenario dicts, one per atom.
    """
    adapter = adapter if adapter is not None else session.adapter
    estimand_adapter = _scenario_adapter(adapter, scenario)
    base_data = _get_base_data(session, adapter)
    var_meta = adapter.variable_metadata()
    weights = _bootstrap_weights_for_adapter(session, adapter)
    resolver = make_aggregation_resolver(session.at, weights)
    from ._atoms import _enumerate_groups, _finalize_atoms, _format_atom_label

    groups, over_keys = _enumerate_groups(session, scenario, base_data, var_meta)

    # Type-check each variable up front. column_index_of_variable raises
    # for categorical/binary/discrete, which is the contract we want for
    # dydx(). The returned index is unused — slopes are now data-side
    # central differences (R/Stata-style total derivatives).
    for v in var_list:
        adapter.column_index_of_variable(v)

    sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
    atoms: list[tuple[str | None, Callable]] = []
    scenarios: list[dict] = []

    for group_label, group_df in groups:
        df, meta = expand_scenario(
            sub_scenario,
            group_df,
            resolver,
            var_meta,
        )
        if session.at == "overall":
            agg_kind = "overall"
        else:
            agg_kind = "none" if len(df) == 1 else "overall"

        # Build base scenario dict for this group (shared across variables)
        base_scen = {}
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
                weights=jnp.asarray(weights)
                if weights is not None
                else None,
                phi_inv=session.phi_inv,
                transform=transform,
                fd_step=session.fd_step,
            )
            atexog_label = _auto_label_from_atexog(sub_scenario.get("atexog"))
            suffix = f"{atexog_label}, {var_name}" if atexog_label else var_name
            label = _format_atom_label(session, group_label, over_keys, suffix)
            atoms.append((label, h_atom))
            scenarios.append(base_scen.copy())

    h, labels = _finalize_atoms(session, atoms)
    return h, labels, scenarios


def _build_contrast_estimand(
    session,
    scenarios: list[dict],
    weights_arg,
    adapter: ModelAdapter | None = None,
) -> Callable:
    """Construct a linear combination estimand for contrasts() calls."""
    adapter = adapter if adapter is not None else session.adapter
    base_data = _get_base_data(session, adapter)

    scenarios_X = []
    scenario_predict_fns = []
    any_per_scenario_predict = False
    weights = _bootstrap_weights_for_adapter(session, adapter)
    for scenario in scenarios:
        df, _ = expand_scenario(
            scenario,
            base_data=base_data,
            aggregation_resolver=make_aggregation_resolver(
                session.at,
                weights,
            ),
            variable_metadata=adapter.variable_metadata(),
        )
        from .._tabular import to_pandas_if_needed

        scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))
        scen_adapter = _scenario_adapter(adapter, scenario)
        if scen_adapter is not adapter:
            any_per_scenario_predict = True
        scenario_predict_fns.append(scen_adapter.predict)

    return make_linear_combination_estimand(
        adapter,
        scenarios_X=scenarios_X,
        weights=weights_arg,
        phi_inv=session.phi_inv,
        scenario_predict_fns=scenario_predict_fns if any_per_scenario_predict else None,
    )


def _scenario_adapter(adapter, scenario: dict):
    """Return a (possibly cloned) adapter parameterized for one scenario.

    Currently honors the ``prediction_time`` scenario key on adapters that
    expose ``with_prediction_time`` (e.g., the lifelines Cox survival
    adapter). Returns the original adapter when no scenario-level override
    is requested.
    """
    t = scenario.get("prediction_time")
    if t is None:
        return adapter
    if not hasattr(adapter, "with_prediction_time"):
        raise ValueError(
            f"Scenario carries 'prediction_time' but adapter "
            f"{type(adapter).__name__} does not support per-scenario time. "
            "Use a time-aware adapter (e.g., LifelinesCoxPHSurvivalAdapter) "
            "or remove 'prediction_time' from the scenario."
        )
    return adapter.with_prediction_time(t)


def _build_evaluate_estimand(
    session,
    scenarios: list[dict],
    compose: Callable,
    adapter: ModelAdapter | None = None,
) -> Callable:
    """Construct an arbitrary composition estimand for evaluate() calls."""
    adapter = adapter if adapter is not None else session.adapter
    base_data = _get_base_data(session, adapter)

    scenarios_X = []
    scenario_predict_fns = []
    any_per_scenario_predict = False
    weights = _bootstrap_weights_for_adapter(session, adapter)
    for scenario in scenarios:
        df, _ = expand_scenario(
            scenario,
            base_data=base_data,
            aggregation_resolver=make_aggregation_resolver(
                session.at,
                weights,
            ),
            variable_metadata=adapter.variable_metadata(),
        )
        from .._tabular import to_pandas_if_needed

        scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))
        scen_adapter = _scenario_adapter(adapter, scenario)
        if scen_adapter is not adapter:
            any_per_scenario_predict = True
        scenario_predict_fns.append(scen_adapter.predict)

    return make_evaluate_estimand(
        adapter,
        scenarios_X=scenarios_X,
        compose=compose,
        phi_inv=session.phi_inv,
        scenario_predict_fns=scenario_predict_fns if any_per_scenario_predict else None,
    )
