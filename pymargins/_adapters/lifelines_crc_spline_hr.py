"""
pymargins._adapters.lifelines_crc_spline_hr

Concrete adapter for lifelines CRCSplineFitter on the hazard-ratio
(relative-risk) scale.

CRCSplineFitter is a proportional-hazards model with a spline baseline.
The ``beta_`` parameter group contains the log-relative-risk multipliers:

    HR = exp(X @ beta_beta)

where beta_beta are the coefficients from the ``beta_`` group (covariate
effects only). The ``gamma_`` parameters control the spline baseline and
are ignored on the hazard-ratio scale.

This is pure JAX — exact autodiff, delta-method SEs valid.

Not auto-detected because CRCSplineFitter shares its class with
LifelinesCRCSplineAdapter (survival-probability scale). Users must
construct this adapter explicitly and pass via adapter=.
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


class LifelinesCRCSplineHRAdapter(ModelAdapter):
    """Adapter for lifelines CRCSplineFitter on the hazard-ratio scale.

    Predicts relative risk: RR = exp(X @ beta_beta).

    Parameters
    ----------
    results : fitted lifelines CRCSplineFitter

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)

        # Extract beta_ coefficients (log-relative-risk multipliers)
        beta_params = results.params_.loc["beta_"]
        self._beta_names = list(beta_params.index)
        self._has_intercept = "Intercept" in self._beta_names

        self._n_baseline_knots = getattr(results, "n_baseline_knots", None)
        self._regressors = getattr(results, "regressors", None)

        self._duration_col = getattr(results, "duration_col", None)
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
        validate_vcov_spec(vcov, adapter_name="LifelinesCRCSplineHRAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params_.loc["beta_"].values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            vm = self.results.variance_matrix_
            # Extract the beta_ submatrix only
            beta_names = self._beta_names
            idx_order = [("beta_", name) for name in beta_names]
            cov = vm.loc[idx_order, idx_order].values
            return jnp.asarray(cov)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            f"LifelinesCRCSplineHRAdapter only supports vcov=None or ndarray. "
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
        eta = jnp.asarray(X) @ beta
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        return jnp.exp(eta)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        missing_cols = [
            col
            for col in self._beta_names
            if col not in df.columns and col not in ("const", "Intercept")
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        aligned = df.reindex(columns=self._beta_names)
        if self._has_intercept and "Intercept" not in df.columns:
            aligned = aligned.copy()
            aligned["Intercept"] = 1.0
        aligned = aligned[self._beta_names]
        return jnp.asarray(aligned.values)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._beta_names,
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
    ) -> LifelinesCRCSplineHRAdapter:
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
            regressors_dict = {}
            for param, model_spec in self._regressors.mappings.items():
                regressors_dict[param] = str(model_spec.formula)
            kwargs["regressors"] = regressors_dict

        new_fitter = CRCSplineFitter(**ctor_kwargs)
        new_fitter.fit(df, **kwargs)
        return LifelinesCRCSplineHRAdapter(new_fitter, training_data=df)
