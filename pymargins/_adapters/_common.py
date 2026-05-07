"""Common helpers shared across statsmodels-based adapters.

These functions encapsulate framework-agnostic logic (design-matrix
construction, variable lookup, metadata inference) that is identical
across OLS, GLM, and other statsmodels adapters.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import jax.numpy as jnp

from .._adapter import VariableInfo


def extract_training_data(results, training_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Resolve training data from explicit argument or model attribute."""
    if training_data is not None:
        return training_data
    if hasattr(results.model, "data") and hasattr(results.model.data, "frame"):
        return results.model.data.frame
    raise ValueError(
        "training_data must be provided when the model wasn't fit "
        "via the formula API (no results.model.data.frame available)."
    )


def design_matrix_from_df(results, exog_names: list[str], df: pd.DataFrame) -> jnp.ndarray:
    """Build a design matrix from a DataFrame using the model's formula."""
    if hasattr(results.model.data, "design_info"):
        from patsy import dmatrix
        design_info = results.model.data.design_info
        X_np = np.asarray(dmatrix(design_info, df, return_type="matrix"))
        return jnp.asarray(X_np)
    # Array-fit fallback: align columns and auto-inject intercept if needed
    aligned = df.reindex(columns=exog_names)
    if "const" in exog_names or "Intercept" in exog_names:
        intercept_name = "const" if "const" in exog_names else "Intercept"
        if intercept_name not in df.columns:
            aligned = aligned.copy()
            aligned[intercept_name] = 1.0
    # Reorder to match exog_names exactly
    aligned = aligned[exog_names]
    return jnp.asarray(aligned.values)


def column_index_of_variable(
    exog_names: list[str],
    variable_metadata: dict[str, VariableInfo],
    variable_name: str,
) -> int:
    """Return the index of ``variable_name`` in the design matrix.

    For categorical or discrete variables this raises ``ValueError``
    because ``dydx()`` is undefined for them.
    """
    meta = variable_metadata.get(variable_name)
    if meta is not None and meta.var_type in ("categorical", "binary", "discrete"):
        raise ValueError(
            f"Variable {variable_name!r} is {meta.var_type}; "
            f"use contrasts() for discrete contrasts, not dydx()."
        )

    if variable_name in exog_names:
        return exog_names.index(variable_name)

    prefix_patterns = [
        f"C({variable_name})[T.",
        f"C({variable_name})[",
        f"{variable_name}[",
        f"{variable_name}.",
        f"{variable_name}:",
    ]
    for pat in prefix_patterns:
        for i, name in enumerate(exog_names):
            if name.startswith(pat):
                return i

    infix_patterns = [
        f":{variable_name}",
        f"I({variable_name}",
    ]
    for pat in infix_patterns:
        for i, name in enumerate(exog_names):
            if pat in name:
                return i

    raise ValueError(
        f"Cannot locate variable {variable_name!r} in design matrix. "
        f"exog_names: {exog_names}"
    )


def build_variable_metadata(training_data: pd.DataFrame) -> dict[str, VariableInfo]:
    """Extract per-variable metadata from the training data."""
    metadata = {}
    for col in training_data.columns:
        series = training_data[col]
        var_type = _infer_variable_type(series)
        metadata[col] = VariableInfo(
            name=col,
            var_type=var_type,
            levels=(list(series.unique()) if var_type in ("binary", "categorical") else None),
            support=((float(series.min()), float(series.max()))
                     if pd.api.types.is_numeric_dtype(series) else None),
        )
    return metadata


def _infer_variable_type(series: pd.Series) -> str:
    if series.dtype == bool:
        return "binary"
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    unique = series.dropna().unique()
    if len(unique) == 2:
        return "binary"
    return "continuous"
