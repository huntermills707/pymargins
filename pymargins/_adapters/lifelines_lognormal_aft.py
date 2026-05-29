"""
pymargins._adapters.lifelines_lognormal_aft

Concrete adapter for lifelines LogNormalAFTFitter.

Predicts survival probability at a fixed time:
    S(t|x) = 1 - Φ((log(t) - μ(x)) / σ)
    where μ(x) = X @ beta_mu
    and   σ    = exp(beta_sigma)

This is pure JAX — exact autodiff, delta-method SEs valid.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import ndtr

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    extract_training_data,
    validate_vcov_spec,
)


class LifelinesLogNormalAFTAdapter(ModelAdapter):
    """Adapter for lifelines LogNormalAFTFitter results.

    Predicts survival probability at a fixed time:
        S(t|x) = 1 - Φ((log(t) - μ(x)) / σ)

    Parameters
    ----------
    results : fitted lifelines LogNormalAFTFitter

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

        # Extract mu coefficients (location parameter)
        mu_params = results.params_.loc["mu_"]
        self._mu_names = list(mu_params.index)
        self._has_intercept = "Intercept" in self._mu_names

        # Extract sigma coefficient (scale parameter)
        sigma_params = results.params_.loc["sigma_"]
        self._sigma_names = list(sigma_params.index)

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
        validate_vcov_spec(vcov, adapter_name="LifelinesLogNormalAFTAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        # Flatten: [mu_coeffs..., sigma_coeffs...]
        mu_vals = self.results.params_.loc["mu_"].values
        sigma_vals = self.results.params_.loc["sigma_"].values
        return jnp.asarray(np.concatenate([mu_vals, sigma_vals]))

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            vm = self.results.variance_matrix_
            # Reorder to match flattened coefficients:
            # [mu_x1, mu_x2, ..., mu_intercept, sigma_intercept]
            mu_names = self._mu_names
            sigma_names = self._sigma_names
            idx_order = [("mu_", name) for name in mu_names] + [
                ("sigma_", name) for name in sigma_names
            ]
            cov = vm.loc[idx_order, idx_order].values
            return jnp.asarray(cov)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesLogNormalAFTAdapter only supports vcov=None or ndarray. "
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
        beta_mu = beta[:p]
        beta_sigma = beta[p]
        mu = jnp.asarray(X) @ beta_mu
        if offset is not None:
            mu = mu + jnp.asarray(offset)
        sigma = jnp.exp(beta_sigma)
        t = self._prediction_time
        z = (jnp.log(t) - mu) / sigma
        return 1.0 - ndtr(z)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        # Build design matrix from mu_ covariates
        missing_cols = [
            col
            for col in self._mu_names
            if col not in df.columns and col not in ("const", "Intercept")
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        aligned = df.reindex(columns=self._mu_names)
        if self._has_intercept and "Intercept" not in df.columns:
            aligned = aligned.copy()
            aligned["Intercept"] = 1.0
        # Reorder to match mu_names exactly
        aligned = aligned[self._mu_names]
        return jnp.asarray(aligned.values)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._mu_names,
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
    ) -> LifelinesLogNormalAFTAdapter:
        from lifelines import LogNormalAFTFitter

        # Reset index to handle bootstrap resampling with replacement
        df = resampled_data.reset_index(drop=True)
        kwargs = {
            "duration_col": self._duration_col,
            "event_col": self._event_col,
        }

        new_fitter = LogNormalAFTFitter()
        new_fitter.fit(df, **kwargs)
        return LifelinesLogNormalAFTAdapter(
            new_fitter,
            training_data=df,
            prediction_time=self._prediction_time,
        )
