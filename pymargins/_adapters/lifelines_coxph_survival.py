"""
pymargins._adapters.lifelines_coxph_survival

Concrete adapter for lifelines CoxPHFitter on the survival-probability scale.

Predicts survival probability at a fixed time using the fitted baseline
survival function:
    S(t|x) = S_0(t) ^ exp((X - X_mean) @ beta)

Uses WrappedFDAdapter because the baseline survival is a nonparametric
step function — no closed-form JAX expression. Finite-difference gradients
are hidden inside the JVP wrapper.

Delta-method SEs are INVALID for survival-probability estimands because they
ignore uncertainty in the estimated baseline hazard. Only bootstrap inference
is supported, matching marginaleffects' recommendation.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import WrappedFDAdapter, VariableInfo
from ._common import (
    extract_training_data,
    design_matrix_from_df,
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class LifelinesCoxPHSurvivalAdapter(WrappedFDAdapter):
    """Adapter for lifelines CoxPHFitter on the survival-probability scale.

    Predicts survival probability at a fixed time using the nonparametric
    baseline survival function:
        S(t|x) = S_0(t) ^ PH(x)

    Parameters
    ----------
    results : fitted lifelines CoxPHFitter

    training_data : pd.DataFrame, optional
        The data the model was fit on.

    prediction_time : float, optional
        The time at which to evaluate survival probabilities.
        Defaults to the median observed event time.
    """

    def __init__(
        self,
        results,
        training_data: Optional[pd.DataFrame] = None,
        prediction_time: Optional[float] = None,
    ):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.params_.index)
        self._x_mean = getattr(results, "_norm_mean", None)
        if self._x_mean is not None:
            self._x_mean = np.asarray(self._x_mean.values)
        self._duration_col = getattr(results, "duration_col", None)
        self._event_col = getattr(results, "event_col", None)
        self._formula = getattr(results, "formula", None)

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
        # Delta method ignores baseline hazard uncertainty → anti-conservative SEs.
        # Only bootstrap is valid for survival-probability estimands.
        return {"bootstrap"}

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="LifelinesCoxPHSurvivalAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params_.values)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.variance_matrix_)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesCoxPHSurvivalAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Compute survival probability at prediction_time.

        Receives NumPy beta and arbitrary X (may be JAX tracer during JVP,
        but the FD wrapper extracts the concrete value before calling this).
        """
        X_np = np.asarray(X)
        if self._x_mean is not None:
            X_np = X_np - self._x_mean
        # Partial hazard
        ph = np.exp(X_np @ beta_np)
        # Baseline survival at prediction_time
        S0_t = self._baseline_survival_at(self._prediction_time)
        # Survival probability
        return S0_t ** ph

    def _baseline_survival_at(self, t: float) -> float:
        """Interpolate baseline survival to time t."""
        S0_df = self.results.baseline_survival_
        if S0_df is None or S0_df.empty:
            raise ValueError("Baseline survival function not available on the fitted model.")
        col = S0_df.columns[0]
        times = S0_df.index.values
        surv = S0_df[col].values
        # np.interp requires x-coordinates to be increasing
        if times[0] > times[-1]:
            times = times[::-1]
            surv = surv[::-1]
        return float(np.interp(t, times, surv))

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
            self._exog_names, self.variable_metadata(), variable_name,
        )

    def variable_metadata(self) -> dict[str, VariableInfo]:
        if not hasattr(self, "_variable_metadata"):
            self._variable_metadata = build_variable_metadata(self.training_data)
        return self._variable_metadata

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "LifelinesCoxPHSurvivalAdapter":
        from lifelines import CoxPHFitter

        # Reset index to handle bootstrap resampling with replacement
        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }
        if self._formula is not None:
            kwargs["formula"] = self._formula

        new_cph = CoxPHFitter()
        new_cph.fit(df, **kwargs)
        return LifelinesCoxPHSurvivalAdapter(
            new_cph,
            training_data=df,
            prediction_time=self._prediction_time,
        )
