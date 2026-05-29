"""
pymargins._adapters.lifelines_loglogistic_aft

Concrete adapter for lifelines LogLogisticAFTFitter.

Predicts survival probability at a fixed time:
    S(t|x) = 1 / (1 + (t / α(x))^β)
    where α(x) = exp(X @ beta_alpha)
    and   β    = exp(beta_beta)

This is pure JAX — exact autodiff, delta-method SEs valid.
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
    extract_training_data,
    validate_vcov_spec,
)


class LifelinesLogLogisticAFTAdapter(ModelAdapter):
    """Adapter for lifelines LogLogisticAFTFitter results.

    Predicts survival probability at a fixed time:
        S(t|x) = 1 / (1 + (t / α(x))^β)

    Parameters
    ----------
    results : fitted lifelines LogLogisticAFTFitter

    training_data : pd.DataFrame, optional
        The data the model was fit on.

    prediction_time : float, optional
        The time at which to evaluate survival probabilities.
        Defaults to the median observed event time.
    """

    def __init__(
        self,
        results,
        training_data: pd.DataFrame | None = None,
        prediction_time: float | None = None,
    ):
        self.results = results
        self._training_data = extract_training_data(results, training_data)

        # Extract alpha coefficients (scale parameter)
        alpha_params = results.params_.loc["alpha_"]
        self._alpha_names = list(alpha_params.index)
        self._has_intercept = "Intercept" in self._alpha_names

        # Extract beta coefficient (shape parameter)
        beta_params = results.params_.loc["beta_"]
        self._beta_names = list(beta_params.index)

        # Prediction time
        if prediction_time is not None:
            self._prediction_time = float(prediction_time)
        else:
            self._prediction_time = self._compute_default_time(results)

        self._duration_col = getattr(results, "duration_col", None)
        self._event_col = getattr(results, "event_col", None)

    def _compute_default_time(self, results) -> float:
        """Use median observed event time as default."""
        durations = results.durations
        events = results.event_observed
        observed_durations = durations[events]
        if len(observed_durations) == 0:
            return float(durations.median())
        return float(np.median(observed_durations))

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
        validate_vcov_spec(vcov, adapter_name="LifelinesLogLogisticAFTAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        # Flatten: [alpha_coeffs..., beta_coeffs...]
        alpha_vals = self.results.params_.loc["alpha_"].values
        beta_vals = self.results.params_.loc["beta_"].values
        return jnp.asarray(np.concatenate([alpha_vals, beta_vals]))

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            vm = self.results.variance_matrix_
            # Reorder to match flattened coefficients:
            # [alpha_x1, alpha_x2, ..., alpha_intercept, beta_intercept]
            alpha_names = self._alpha_names
            beta_names = self._beta_names
            idx_order = [("alpha_", name) for name in alpha_names] + [
                ("beta_", name) for name in beta_names
            ]
            cov = vm.loc[idx_order, idx_order].values
            return jnp.asarray(cov)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesLogLogisticAFTAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        p = X.shape[1]
        beta_alpha = beta[:p]
        beta_beta = beta[p]
        alpha = jnp.exp(jnp.asarray(X) @ beta_alpha)
        if offset is not None:
            alpha = alpha * jnp.exp(jnp.asarray(offset))
        beta_param = jnp.exp(beta_beta)
        t = self._prediction_time
        return 1.0 / (1.0 + (t / alpha) ** beta_param)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        # Build design matrix from alpha_ covariates
        missing_cols = [
            col
            for col in self._alpha_names
            if col not in df.columns and col not in ("const", "Intercept")
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        aligned = df.reindex(columns=self._alpha_names)
        if self._has_intercept and "Intercept" not in df.columns:
            aligned = aligned.copy()
            aligned["Intercept"] = 1.0
        # Reorder to match alpha_names exactly
        aligned = aligned[self._alpha_names]
        return jnp.asarray(aligned.values)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._alpha_names,
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
    ) -> LifelinesLogLogisticAFTAdapter:
        from lifelines import LogLogisticAFTFitter

        # Reset index to handle bootstrap resampling with replacement
        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }

        new_fitter = LogLogisticAFTFitter()
        new_fitter.fit(df, **kwargs)
        return LifelinesLogLogisticAFTAdapter(
            new_fitter,
            training_data=df,
            prediction_time=self._prediction_time,
        )
