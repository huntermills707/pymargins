"""
pymargins._adapters.lifelines_crc_spline

Concrete adapter for lifelines CRCSplineFitter.

Predicts survival probability at a fixed time using the fitted spline
baseline hazard. Uses WrappedFDAdapter because the spline baseline has no
closed-form JAX expression.

Delta-method SEs are INVALID for survival-probability estimands because they
ignore uncertainty in the estimated spline baseline. Only bootstrap
inference is supported.
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


class LifelinesCRCSplineAdapter(WrappedFDAdapter):
    """Adapter for lifelines CRCSplineFitter.

    Predicts survival probability at a fixed time. Uses finite-difference
    JVP because the spline baseline hazard has no closed-form JAX
    expression.

    Parameters
    ----------
    results : fitted lifelines CRCSplineFitter

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

        # Extract parameter groups and covariate names
        params = results.params_
        self._param_groups = params.index.get_level_values(0).unique().tolist()

        # beta_ group contains covariate effects
        if "beta_" in self._param_groups:
            self._exog_names = params.loc["beta_"].index.tolist()
        else:
            self._exog_names = []
        self._has_intercept = "Intercept" in self._exog_names

        # Store constructor args needed for refit
        self._n_baseline_knots = getattr(results, "n_baseline_knots", None)
        self._regressors = getattr(results, "regressors", None)

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
        validate_vcov_spec(vcov, adapter_name="LifelinesCRCSplineAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params_.values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.variance_matrix_.values)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesCRCSplineAdapter only supports vcov=None or ndarray. "
            f"Got {vcov_spec!r}"
        )

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Compute survival probability at prediction_time."""
        df_eval = pd.DataFrame(np.asarray(X), columns=self._exog_names)
        fitter = copy.copy(self.results)
        pred = fitter.predict_survival_function(df_eval, times=[self._prediction_time])
        return pred.values.flatten()

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
    ) -> LifelinesCRCSplineAdapter:
        from lifelines import CRCSplineFitter

        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }

        ctor_kwargs = {}
        if self._n_baseline_knots is not None:
            ctor_kwargs["n_baseline_knots"] = self._n_baseline_knots
        if self._regressors is not None:
            # Convert CovariateParameterMappings back to dict of formula strings
            regressors_dict = {}
            for param, model_spec in self._regressors.mappings.items():
                regressors_dict[param] = str(model_spec.formula)
            kwargs["regressors"] = regressors_dict

        new_fitter = CRCSplineFitter(**ctor_kwargs)
        new_fitter.fit(df, **kwargs)
        return LifelinesCRCSplineAdapter(
            new_fitter,
            training_data=df,
            prediction_time=self._prediction_time,
        )
