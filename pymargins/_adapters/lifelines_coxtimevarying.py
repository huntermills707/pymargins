"""
pymargins._adapters.lifelines_coxtimevarying

Concrete adapter for lifelines CoxTimeVaryingFitter.

Predicts on the partial-hazard scale: PH = exp(X @ beta).

CoxTimeVaryingFitter models time-varying covariates using (start, stop]
intervals. The partial hazard has the same interpretation as standard Cox PH:
exp(X @ beta) is the multiplicative effect on the hazard. There is no
baseline survival stored, so only the hazard-ratio scale is supported.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
    validate_vcov_spec,
)


class LifelinesCoxTimeVaryingAdapter(ModelAdapter):
    """Adapter for lifelines CoxTimeVaryingFitter results.

    Predicts partial hazards: PH = exp(X @ beta).

    Parameters
    ----------
    results : fitted lifelines CoxTimeVaryingFitter

    training_data : pd.DataFrame, optional
        The data the model was fit on. Required because CoxTimeVaryingFitter
        does not store the full training DataFrame.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.params_.index)
        self._x_mean = getattr(results, "_norm_mean", None)
        if self._x_mean is not None:
            self._x_mean = jnp.asarray(self._x_mean.values)
        self._id_col = getattr(results, "id_col", None)
        self._event_col = getattr(results, "event_col", None)
        self._start_col = getattr(results, "start_col", None)
        self._stop_col = getattr(results, "stop_col", None)
        self._formula = getattr(results, "formula", None)

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
        validate_vcov_spec(vcov, adapter_name="LifelinesCoxTimeVaryingAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params_.values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.variance_matrix_)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            raise ValueError(
                f"LifelinesCoxTimeVaryingAdapter does not support vcov={vcov_spec!r}. "
                f"Use None (default variance_matrix_) or a user-supplied ndarray."
            )

        if isinstance(vcov_spec, dict):
            raise ValueError(
                "LifelinesCoxTimeVaryingAdapter does not support vcov dict. "
                "Use None (default variance_matrix_) or a user-supplied ndarray."
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        X_arr = jnp.asarray(X)
        if self._x_mean is not None:
            X_arr = X_arr - self._x_mean
        eta = X_arr @ beta
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        return jnp.exp(eta)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        if self._formula is not None:
            from patsy import dmatrix

            X_np = np.asarray(dmatrix(self._formula, df, return_type="matrix"))
            # Drop intercept if present — Cox PH absorbs intercept into baseline hazard
            if X_np.shape[1] > len(self._exog_names):
                X_np = X_np[:, 1:]
            return jnp.asarray(X_np)
        return design_matrix_from_df(self.results, self._exog_names, df)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._exog_names,
            self.variable_metadata(),
            variable_name,
        )

    def variable_metadata(self) -> dict[str, VariableInfo]:
        if not hasattr(self, "_variable_metadata"):
            self._variable_metadata = build_variable_metadata(self.training_data)
        return self._variable_metadata

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> LifelinesCoxTimeVaryingAdapter:
        from lifelines import CoxTimeVaryingFitter

        # Reset index to handle bootstrap resampling with replacement
        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "id_col": self._id_col,
            "event_col": self._event_col,
            "start_col": self._start_col,
            "stop_col": self._stop_col,
        }
        if self._formula is not None:
            kwargs["formula"] = self._formula

        new_ctv = CoxTimeVaryingFitter()
        new_ctv.fit(df, **kwargs)
        return LifelinesCoxTimeVaryingAdapter(new_ctv, training_data=df)
