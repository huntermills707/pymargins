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
    if hasattr(results, "model") and hasattr(results.model, "data") and hasattr(results.model.data, "frame"):
        frame = results.model.data.frame
        if frame is not None:
            return frame
    raise ValueError(
        "training_data must be provided when the model wasn't fit "
        "via the formula API (no results.model.data.frame available)."
    )


def design_matrix_from_df(results, exog_names: list[str], df: pd.DataFrame) -> jnp.ndarray:
    """Build a design matrix from a DataFrame using the model's formula."""
    if hasattr(results, "model") and results.model is not None and hasattr(results.model.data, "design_info"):
        from patsy import dmatrix
        design_info = results.model.data.design_info
        X_np = np.asarray(dmatrix(design_info, df, return_type="matrix"))
        return jnp.asarray(X_np)
    # Array-fit fallback: align columns and auto-inject intercept if needed
    aligned = df.reindex(columns=exog_names)
    # Detect missing columns that became NaN after reindexing
    missing_cols = [col for col in exog_names if col not in df.columns and col not in ("const", "Intercept")]
    if missing_cols:
        raise ValueError(
            f"Missing columns required by the model's exog_names: {missing_cols}. "
            f"Available columns: {list(df.columns)}."
        )
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

    For categorical or binary variables this raises ``ValueError``
    because ``dydx()`` is undefined for them.
    """
    meta = variable_metadata.get(variable_name)
    if meta is not None and meta.var_type in ("categorical", "binary"):
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


def validate_vcov_spec(vcov_spec, adapter_name: str = "Adapter") -> None:
    """Validate that a vcov specification is supported at attach time.

    Note: this function does not know the parameter count, so it cannot
    validate that a user-supplied ndarray is square or matches the model
    dimensions. Adapters should perform shape checks separately if needed.

    Raises ValueError with a clear message if the spec is not supported.
    """
    if vcov_spec is None:
        return

    if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
        return

    if isinstance(vcov_spec, str):
        spec_lower = vcov_spec.lower()
        if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
            return
        raise ValueError(
            f"{adapter_name} does not support vcov={vcov_spec!r}. "
            f"Supported strings: 'HC0', 'HC1', 'HC2', 'HC3'."
        )

    if isinstance(vcov_spec, dict):
        kind = vcov_spec.get("type")
        if kind == "cluster":
            groups = vcov_spec.get("groups")
            if groups is None:
                raise ValueError(
                    f"{adapter_name}: cluster vcov requires 'groups' in the spec dict."
                )
            if hasattr(groups, "__len__") and len(groups) == 0:
                raise ValueError(
                    f"{adapter_name}: cluster vcov 'groups' must not be empty."
                )
            return
        raise ValueError(
            f"{adapter_name} does not support vcov dict with type={kind!r}. "
            f"Supported dict: {{'type': 'cluster', 'groups': ...}}."
        )

    raise ValueError(
        f"{adapter_name} does not support vcov spec of type {type(vcov_spec).__name__}."
    )
