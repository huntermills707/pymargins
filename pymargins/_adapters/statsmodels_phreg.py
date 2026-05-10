"""
pymargins._adapters.statsmodels_phreg

Concrete adapter for statsmodels PHReg (Cox proportional hazards).

Predicts on the hazard-ratio scale: HR = exp(X β).
Delta-method SEs are valid on this scale because the baseline hazard
does not enter the hazard ratio.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    extract_training_data,
    design_matrix_from_df,
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class StatsmodelsPHRegAdapter(ModelAdapter):
    """Adapter for statsmodels PHReg (Cox proportional hazards) results.

    Predicts hazard ratios: HR = exp(X β).

    Parameters
    ----------
    results : fitted statsmodels PHReg result object

    training_data : pd.DataFrame, optional
        The data the model was fit on. Required because PHReg has no
        formula API and does not store training data.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)

    @property
    def training_data(self):
        return self._training_data

    @property
    def supports_jax_autodiff(self) -> bool:
        return True

    @property
    def supported_inference_methods(self) -> set[str]:
        return {"delta", "simulation", "bootstrap"}

    @property
    def gradient_backend_recommendation(self) -> str:
        return "autodiff"

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="StatsmodelsPHRegAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.cov_params())

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            raise ValueError(
                f"StatsmodelsPHRegAdapter does not support vcov={vcov_spec!r}. "
                f"Use None (default cov_params) or a user-supplied ndarray."
            )

        if isinstance(vcov_spec, dict):
            raise ValueError(
                f"StatsmodelsPHRegAdapter does not support vcov dict. "
                f"Use None (default cov_params) or a user-supplied ndarray."
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        eta = jnp.asarray(X) @ beta
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        return jnp.exp(eta)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        return design_matrix_from_df(self.results, self._exog_names, df)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._exog_names, self.variable_metadata(), variable_name,
        )

    def variable_metadata(self) -> dict[str, VariableInfo]:
        if not hasattr(self, "_variable_metadata"):
            self._variable_metadata = build_variable_metadata(self.training_data)
        return self._variable_metadata

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsPHRegAdapter":
        from statsmodels.duration.hazard_regression import PHReg

        # Find duration (endog) and status columns
        endog_name, status_name = self._find_survival_columns(resampled_data)

        exog_cols = [c for c in self._exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}."
            )

        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        status = resampled_data[status_name].values

        kwargs = {"status": status}
        if index is not None:
            entry = getattr(self.results.model, "entry", None)
            if entry is not None:
                kwargs["entry"] = np.asarray(entry)[index]
        new_results = PHReg(endog, exog_df, **kwargs).fit()
        return StatsmodelsPHRegAdapter(new_results, training_data=resampled_data)

    def _find_survival_columns(self, df: pd.DataFrame) -> tuple[str, str]:
        """Find the duration (endog) and event indicator (status) columns.

        Heuristic: look for columns not in exog_names. The duration column
        should be positive continuous; the status column should be binary.
        """
        non_exog_cols = [c for c in df.columns if c not in self._exog_names]
        if len(non_exog_cols) < 2:
            raise ValueError(
                f"Need at least 2 non-covariate columns (duration + status). "
                f"Found: {non_exog_cols}"
            )

        # Find binary column for status
        status_candidates = []
        for col in non_exog_cols:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, True, False}):
                status_candidates.append(col)

        if len(status_candidates) == 0:
            raise ValueError("Could not find binary event indicator column.")
        if len(status_candidates) == 1:
            status_name = status_candidates[0]
            # Duration is the other non-exog column
            duration_candidates = [c for c in non_exog_cols if c != status_name]
            if len(duration_candidates) == 1:
                return duration_candidates[0], status_name
            # Multiple candidates — pick the one with positive continuous values
            for col in duration_candidates:
                if df[col].dropna().min() > 0:
                    return col, status_name
            return duration_candidates[0], status_name

        # Multiple status candidates — pick the one with fewer unique values
        status_name = min(status_candidates, key=lambda c: len(df[c].unique()))
        duration_candidates = [c for c in non_exog_cols if c != status_name]
        for col in duration_candidates:
            if df[col].dropna().min() > 0:
                return col, status_name
        return duration_candidates[0], status_name
