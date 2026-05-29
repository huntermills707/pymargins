"""
pymargins._adapters.lifelines_aalen_additive

Concrete adapter for lifelines AalenAdditiveFitter.

Predicts survival probability at a fixed time using the fitted cumulative
hazard function. Uses WrappedFDAdapter because the cumulative hazard is
nonparametric and time-varying — no closed-form JAX expression.

AalenAdditiveFitter does not have static params_ or variance_matrix_.
Coefficients are extracted from the last row of cumulative_hazards_, and
covariance is approximated from the last row of cumulative_variance_.

Delta-method SEs are INVALID. Only bootstrap inference is supported.
"""

from __future__ import annotations

import copy
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import VariableInfo, WrappedFDAdapter
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    extract_training_data,
    validate_vcov_spec,
)


class LifelinesAalenAdditiveAdapter(WrappedFDAdapter):
    """Adapter for lifelines AalenAdditiveFitter.

    Predicts survival probability at a fixed time. Uses finite-difference
    JVP because the cumulative hazard is nonparametric and time-varying.

    Parameters
    ----------
    results : fitted lifelines AalenAdditiveFitter

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

        # Extract covariate names from cumulative_hazards_ columns
        cum_haz = results.cumulative_hazards_
        self._exog_names = list(cum_haz.columns)
        self._has_intercept = "Intercept" in self._exog_names

        self._duration_col = getattr(results, "duration_col", None)
        self._event_col = getattr(results, "event_col", None)

        if prediction_time is not None:
            self._prediction_time = float(prediction_time)
        else:
            self._prediction_time = self._compute_default_time(results)

        self._fd_step = 1e-6

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
    def supported_inference_methods(self) -> set[str]:
        return {"bootstrap"}

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="LifelinesAalenAdditiveAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        # Use the last row of cumulative_hazards_ as the effective coefficients
        cum_haz = self.results.cumulative_hazards_
        return jnp.asarray(cum_haz.iloc[-1].values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            # Build a diagonal covariance from cumulative_variance_ last row
            cum_var = self.results.cumulative_variance_
            last_var = cum_var.iloc[-1].values
            # Ensure non-negative
            last_var = np.maximum(last_var, 0.0)
            return jnp.asarray(np.diag(last_var))

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesAalenAdditiveAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Compute survival probability at prediction_time.

        Note: beta_np is ignored because AalenAdditiveFitter uses its
        internal cumulative_hazards_ for prediction. Bootstrap refit
        ensures the model state is updated.
        """
        df_eval = pd.DataFrame(np.asarray(X), columns=self._exog_names)
        fitter = copy.copy(self.results)
        pred = fitter.predict_survival_function(df_eval, times=[self._prediction_time])
        # Extract the row closest to prediction_time
        idx = pred.index.get_indexer([self._prediction_time], method="nearest")[0]
        values = pred.iloc[idx].values
        # Clip to [0, 1] since Aalen model can produce values slightly outside
        return np.clip(values, 0.0, 1.0)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        missing_cols = [
            col
            for col in self._exog_names
            if col not in df.columns and col not in ("const", "Intercept")
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        aligned = df.reindex(columns=self._exog_names)
        if self._has_intercept and "Intercept" not in df.columns:
            aligned = aligned.copy()
            aligned["Intercept"] = 1.0
        aligned = aligned[self._exog_names]
        return jnp.asarray(aligned.values)

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
    ) -> LifelinesAalenAdditiveAdapter:
        from lifelines import AalenAdditiveFitter

        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }

        new_fitter = AalenAdditiveFitter()
        new_fitter.fit(df, **kwargs)
        return LifelinesAalenAdditiveAdapter(
            new_fitter,
            training_data=df,
            prediction_time=self._prediction_time,
        )
