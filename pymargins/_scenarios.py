"""
pymargins._scenarios

Build counterfactual design matrices from user-facing scenario specifications
(atexog, over). This is the layer between the user's "what counterfactual
do I want to evaluate" and the model's "design matrix to plug in".

A scenario specification is a dict with optional keys:
  'atexog' : {variable: value | list_of_values}  — values for exogenous variables
  'over'   : str | list[str]                     — subgroup variable(s)
  'data'   : pd.DataFrame | TabularData          — explicit row(s) to use
  'label'  : str                                  — output identifier

When unspecified variables exist: filled per the session's `at` setting
(overall → use observed values row-wise; typical → use the typical value; etc.).

Design questions handled here
-----------------------------
- Cardinality: a single dict produces n_rows = (rows in original data, for
  overall) or 1 (for typical etc.); a dict with list-valued atexog produces
  more rows by Cartesian product.
- Group construction for `over`: replicate the design within each group
  level, with the group variable substituted in.
- Formula expansion: factor encoding, interactions, splines must match the
  model's design exactly. Adapters provide framework-specific implementations
  via design_matrix_from_df().
"""

from __future__ import annotations
from typing import Optional, Union, Any
from itertools import product
import numpy as np
import pandas as pd

from ._tabular import TabularData, PandasTabular, as_tabular, concat_tables


# ---------------------------------------------------------------------------
# Scenario expansion
# ---------------------------------------------------------------------------

def expand_scenario(
    scenario: dict,
    base_data: Union[pd.DataFrame, TabularData],
    aggregation_resolver,
    variable_metadata: dict,
) -> tuple[TabularData, dict]:
    """Expand a scenario specification into a concrete TabularData of rows.

    The output has one row per evaluation point. The exact row
    structure depends on the scenario:

    - Empty scenario, overall: returns base_data unchanged
      (every observed row is used).
    - Empty scenario, typical: returns 1 row with each variable at its
      typical value.
    - {"atexog": {"x": 5}}, overall: returns base_data with column x
      replaced by 5.
    - {"atexog": {"x": [1, 2, 3]}}: returns rows tripled, one set per value of x.
    - {"atexog": {"x": [1, 2, 3], "y": [10, 20]}}: Cartesian product → 6 sets
      of rows.
    - {"data": custom_df}: returns custom_df verbatim (advanced override).

    Parameters
    ----------
    scenario : dict
        The user's scenario spec.

    base_data : pd.DataFrame or TabularData
        The training data (or representative point dataframe) used to fill
        variables not specified in atexog.

    aggregation_resolver : callable (data, variable_metadata) -> TabularData
        Resolves the session's `at` setting into either:
          - "overall": returns base_data (all observed rows)
          - "typical" / "mean" / etc.: returns a 1-row DataFrame
            with each variable at its representative value

    variable_metadata : dict[str, VariableInfo]
        Per-variable type/level info from the adapter.

    Returns
    -------
    expanded : TabularData
        Concrete rows for evaluation.

    metadata : dict
        Bookkeeping info: list of original variable values (for output
        labeling), `at` strategy used, etc.

    Raises
    ------
    ValueError
        If a variable in atexog is not recognized by the model.
    """
    # Normalize to TabularData
    if isinstance(base_data, pd.DataFrame):
        base_data = as_tabular(base_data)

    if "data" in scenario:
        # Explicit override: use the provided DataFrame directly
        return as_tabular(scenario["data"]).copy(), {"strategy": "explicit"}

    atexog = scenario.get("atexog", {}) or {}

    # Validate: known variables
    unknown = set(atexog.keys()) - set(variable_metadata.keys())
    if unknown:
        raise ValueError(
            f"Unknown variable(s): {sorted(unknown)}. "
            f"Known: {sorted(variable_metadata.keys())}."
        )

    # Resolve aggregation: produces either base_data (overall) or a 1-row df
    background = aggregation_resolver(base_data, variable_metadata)

    # Materialize atexog values into a Cartesian-product grid
    grid_vars = {k: _to_list(v) for k, v in atexog.items()}
    if grid_vars:
        grid_keys = list(grid_vars.keys())
        grid_values = [grid_vars[k] for k in grid_keys]
        grid_rows = list(product(*grid_values))
    else:
        grid_keys = []
        grid_rows = [()]  # single empty tuple → no expansion

    # For each grid row, build a copy of background with atexog applied
    pieces = []
    for grid_row in grid_rows:
        block = background
        for k, v in zip(grid_keys, grid_row):
            block = block.with_column(k, v)
        pieces.append(block)

    expanded = concat_tables(pieces)

    metadata = {
        "strategy": "expanded",
        "atexog_keys": grid_keys,
        "grid_rows": grid_rows,
        "n_grid_points": len(grid_rows),
        "rows_per_grid_point": len(background),
    }

    return expanded, metadata


# ---------------------------------------------------------------------------
# Aggregation resolver
# ---------------------------------------------------------------------------

def make_aggregation_resolver(at, weights=None):
    """Build a resolver function that turns the session's `at` setting
    into either base_data (overall) or a representative 1-row DataFrame.

    Parameters
    ----------
    at : str or dict or callable
        One of:
          - "overall"   : per-row, no averaging; resolver returns base_data
          - "typical"   : type-aware: median continuous, mode discrete
          - "mean"      : mean of all variables
          - "median"    : median of all
          - "mode"      : mode of all (errors on continuous)
          - dict          : per-variable specification
          - callable      : (data) -> 1-row representative DataFrame

    weights : array-like, optional
        Weights for computing weighted means/medians/modes. If None,
        uniform weights are used.

    Returns
    -------
    resolver : callable (data, variable_metadata) -> TabularData
        Accepts the data and metadata, returns either the original data
        (overall) or a 1-row DataFrame at the representative point.
    """
    if at == "overall":
        return lambda data, meta: data

    if callable(at):
        return lambda data, meta: as_tabular(at(data))

    def resolver(data, meta):
        columns = data.columns
        missing = [var_name for var_name in meta.keys() if var_name not in columns]
        if missing:
            raise ValueError(
                f"Missing column(s) in data: {sorted(missing)}. "
                f"Available: {sorted(columns)}."
            )
        result = {}
        for var_name, info in meta.items():
            col = data[var_name]
            spec = _resolve_var_spec(at, var_name, info)
            result[var_name] = _summarize_column(col, spec, info, weights)
        return as_tabular(pd.DataFrame([result]))

    return resolver


def _resolve_var_spec(at, var_name, info):
    """Determine the per-variable summary spec from the `at` setting."""
    if isinstance(at, dict):
        if var_name in at:
            return at[var_name]
        if "_default" in at:
            return at["_default"]
        # Fall through to type-aware default
        at = "typical"

    if at == "typical":
        if info.var_type == "continuous":
            return "median"
        else:
            return "mode"
    elif at == "mean":
        return "mean"
    elif at == "median":
        return "median"
    elif at == "mode":
        return "mode"
    return at


def _summarize_column(col, spec, info, weights):
    """Compute the summary value for a single column per the spec."""
    if spec == "mean":
        return _weighted_mean(col, weights)
    elif spec == "median":
        return _weighted_median(col, weights)
    elif spec == "mode":
        if info.var_type == "continuous":
            raise ValueError(
                f"Mode requested for continuous variable '{info.name}'. "
                "Mode is undefined for continuous distributions; use 'median' "
                "or specify a value via at=."
            )
        return _weighted_mode(col, weights)
    elif isinstance(spec, str) and spec.startswith("p") and spec[1:].isdigit():
        # Percentile: "p25", "p75", etc.
        q = int(spec[1:]) / 100.0
        return _weighted_quantile(col, q, weights)
    elif spec == "min":
        return float(np.min(col))
    elif spec == "max":
        return float(np.max(col))
    elif callable(spec):
        return spec(col)
    else:
        raise ValueError(f"Unknown variable summary spec: {spec!r}")


# ---------------------------------------------------------------------------
# Weighted summary helpers (NumPy-based, since this isn't in the autodiff path)
# ---------------------------------------------------------------------------

def _weighted_mean(col, weights):
    if weights is None:
        return float(np.mean(col))
    return float(np.average(np.asarray(col), weights=np.asarray(weights)))


def _weighted_median(col, weights):
    return _weighted_quantile(col, 0.5, weights)


def _weighted_quantile(col, q, weights):
    arr = np.asarray(col)
    if weights is None:
        return float(np.quantile(arr, q))
    w = np.asarray(weights)
    order = np.argsort(arr)
    arr_s = arr[order]
    w_s = w[order]
    cum = np.cumsum(w_s) / np.sum(w_s)
    idx = np.searchsorted(cum, q)
    return float(arr_s[min(idx, len(arr_s) - 1)])


def _weighted_mode(col, weights):
    """Compute the mode of a column.

    Parameters
    ----------
    col : array-like
        Column values (numpy array, pandas Series, or similar).
    weights : array-like, optional
        Weights for weighted mode.

    Returns
    -------
    mode_value
        The most frequent value (or the value with the highest weight).
    """
    arr = np.asarray(col)
    if weights is None:
        # Use np.unique with return_counts for numpy arrays
        unique_vals, counts = np.unique(arr, return_counts=True)
        return unique_vals[np.argmax(counts)]
    # Weighted mode: sum weights per unique value
    w = np.asarray(weights)
    unique_vals = np.unique(arr)
    weighted_counts = np.array([w[arr == uv].sum() for uv in unique_vals])
    return unique_vals[np.argmax(weighted_counts)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(value):
    """Normalize a value or list-of-values to a list."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def _auto_label_from_atexog(atexog: Optional[dict]) -> Optional[str]:
    """Build a readable label from an atexog dict, e.g. {'x': 1, 'y': 2} → 'x=1, y=2'."""
    if not atexog:
        return None
    return ", ".join(f"{k}={v}" for k, v in atexog.items())
