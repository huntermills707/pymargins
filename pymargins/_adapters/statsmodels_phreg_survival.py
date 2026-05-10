"""
pymargins._adapters.statsmodels_phreg_survival

Concrete adapter for statsmodels PHReg on the survival-probability scale.

Predicts survival probability at a fixed time using the Breslow baseline
cumulative hazard:
    S(t|x) = exp(-H_0(t) * exp(X @ beta))

Uses WrappedFDAdapter because the baseline cumulative hazard is a
nonparametric step function — no closed-form JAX expression.

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


class StatsmodelsPHRegSurvivalAdapter(WrappedFDAdapter):
    """Adapter for statsmodels PHReg on the survival-probability scale.

    Predicts survival probability at a fixed time:
        S(t|x) = exp(-H_0(t) * HR(x))

    Parameters
    ----------
    results : fitted statsmodels PHReg result object

    training_data : pd.DataFrame, optional
        The data the model was fit on. Required because PHReg array-fit
        does not store training data.

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
        self._exog_names = list(results.model.exog_names)

        if prediction_time is not None:
            self._prediction_time = float(prediction_time)
        else:
            self._prediction_time = self._compute_default_time(results)

        self._fd_step = 1e-6

    def _compute_default_time(self, results) -> float:
        """Use median observed event time as default."""
        status = np.asarray(results.model.status)
        endog = np.asarray(results.model.endog)
        observed = endog[status.astype(bool)]
        if len(observed) == 0:
            return float(np.median(endog))
        return float(np.median(observed))

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
        validate_vcov_spec(vcov, adapter_name="StatsmodelsPHRegSurvivalAdapter")
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

        raise ValueError(
            f"StatsmodelsPHRegSurvivalAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Compute survival probability at prediction_time.

        Uses statsmodels PHReg.predict(pred_type='surv') with endog set to
        the prediction time for all observations.
        """
        X_np = np.asarray(X)
        n_obs = X_np.shape[0]
        t = self._prediction_time
        pred = self.results.model.predict(
            beta_np,
            endog=np.full(n_obs, t),
            exog=X_np,
            pred_type="surv",
        )
        return np.asarray(pred.predicted_values)

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

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsPHRegSurvivalAdapter":
        from statsmodels.duration.hazard_regression import PHReg

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

        new_results = PHReg(endog, exog_df, status=status).fit()
        return StatsmodelsPHRegSurvivalAdapter(
            new_results,
            training_data=resampled_data,
            prediction_time=self._prediction_time,
        )

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
            duration_candidates = [c for c in non_exog_cols if c != status_name]
            if len(duration_candidates) == 1:
                return duration_candidates[0], status_name
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
