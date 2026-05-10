"""
pymargins._adapters.lifelines_generalized_gamma

Concrete adapter for lifelines GeneralizedGammaRegressionFitter.

Predicts survival probability at a fixed time using the fitted
GeneralizedGamma distribution. The survival function has no closed-form
JAX expression, so we use WrappedFDAdapter with lifelines' native
predict_survival_function.

Delta-method SEs are INVALID because the baseline is implicitly
nonparametric (the GeneralizedGamma shape is estimated from data).
Only bootstrap inference is supported.
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


class LifelinesGeneralizedGammaAdapter(WrappedFDAdapter):
    """Adapter for lifelines GeneralizedGammaRegressionFitter results.

    Predicts survival probability at a fixed time using the fitted
    GeneralizedGamma distribution.

    Parameters
    ----------
    results : fitted lifelines GeneralizedGammaRegressionFitter

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

        # Extract covariate names from all parameter groups
        # params_ has groups: sigma_, mu_, lambda_
        param_groups = ["sigma_", "mu_", "lambda_"]
        all_names = set()
        for grp in param_groups:
            if grp in results.params_.index.get_level_values(0):
                names = results.params_.loc[grp].index.tolist()
                all_names.update(names)
        self._exog_names = sorted(all_names)
        self._has_intercept = "Intercept" in self._exog_names

        if prediction_time is not None:
            self._prediction_time = float(prediction_time)
        else:
            self._prediction_time = self._compute_default_time(results)

        self._duration_col = getattr(results, "duration_col", None)
        self._event_col = getattr(results, "event_col", None)
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
        validate_vcov_spec(vcov, adapter_name="LifelinesGeneralizedGammaAdapter")
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
            f"LifelinesGeneralizedGammaAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Compute survival probability at prediction_time.

        Uses lifelines' predict_survival_function. We create a shallow copy
        of the fitter and override its parameters to evaluate at beta_np,
        avoiding mutation of the original fitter (which is not thread-safe).
        """
        X_np = np.asarray(X)
        n_obs = X_np.shape[0]
        t = self._prediction_time

        # lifelines predict_survival_function uses the stored params.
        # We need to evaluate at beta_np. Create a shallow copy to avoid
        # mutating the original fitter (critical for thread-safe bootstrap).
        from copy import copy
        fitter_copy = copy(self.results)
        fitter_copy.params_ = pd.Series(
            beta_np,
            index=self.results.params_.index,
        )
        # Build a DataFrame with the required columns
        df_pred = pd.DataFrame(X_np, columns=self._exog_names)
        S = fitter_copy.predict_survival_function(df_pred, times=[t])
        return S.values.flatten()

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
        if self._has_intercept and "Intercept" not in df.columns:
            aligned = aligned.copy()
            aligned["Intercept"] = 1.0
        aligned = aligned[self._exog_names]
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

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "LifelinesGeneralizedGammaAdapter":
        from lifelines import GeneralizedGammaRegressionFitter

        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }

        new_gg = GeneralizedGammaRegressionFitter()
        new_gg.fit(df, **kwargs)
        return LifelinesGeneralizedGammaAdapter(
            new_gg,
            training_data=df,
            prediction_time=self._prediction_time,
        )
