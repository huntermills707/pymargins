"""
pymargins._adapters.lifelines_cox_timevarying_hr

Concrete adapter for lifelines CoxTimeVaryingFitter on the hazard-ratio
(partial hazard) scale.

Predicts partial hazard:
    PH = exp((X - X_mean) @ beta)

This is pure JAX — exact autodiff, delta-method SEs valid.

Not auto-detected because CoxTimeVaryingFitter shares its class with
LifelinesCoxTimeVaryingAdapter (survival-probability scale). Users must
construct this adapter explicitly and pass via adapter=.
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


class LifelinesCoxTimeVaryingHRAdapter(ModelAdapter):
    """Adapter for lifelines CoxTimeVaryingFitter on the partial-hazard scale.

    Predicts partial hazards: PH = exp((X - X_mean) @ beta).

    Parameters
    ----------
    results : fitted lifelines CoxTimeVaryingFitter

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.params_.index)
        self._x_mean = getattr(results, "_norm_mean", None)
        if self._x_mean is not None:
            self._x_mean = jnp.asarray(self._x_mean.values)
        self._id_col = getattr(results, "id_col", None)
        self._start_col = getattr(results, "start_col", None)
        self._stop_col = getattr(results, "stop_col", None)
        self._event_col = getattr(results, "event_col", None)

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
        validate_vcov_spec(vcov, adapter_name="LifelinesCoxTimeVaryingHRAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params_.values)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.variance_matrix_.values)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesCoxTimeVaryingHRAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
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
        missing_cols = [col for col in self._exog_names if col not in df.columns and col not in ("const", "Intercept")]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        aligned = df.reindex(columns=self._exog_names)
        return jnp.asarray(aligned.values)

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

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "LifelinesCoxTimeVaryingHRAdapter":
        from lifelines import CoxTimeVaryingFitter

        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "id_col": self._id_col,
            "start_col": self._start_col,
            "stop_col": self._stop_col,
            "event_col": self._event_col,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        new_fitter = CoxTimeVaryingFitter(penalizer=0.1)
        new_fitter.fit(df, **kwargs)
        return LifelinesCoxTimeVaryingHRAdapter(new_fitter, training_data=df)
