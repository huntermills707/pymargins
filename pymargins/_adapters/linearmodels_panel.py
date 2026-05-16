"""
pymargins._adapters.linearmodels_panel

Concrete adapter for linearmodels panel-data models:
  PanelOLS, PooledOLS, RandomEffects, FirstDifferenceOLS, BetweenOLS.

Inherits predict() from LinearPredictionAdapter (simple Xβ).

Fixed effects (EntityEffects, TimeEffects) are absorbed by linearmodels
and do not appear in params.  For marginal effects and contrasts, the
absorbed terms cancel out, so predictions are computed on the non-
absorbed design matrix only.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, FirstDifferenceOLS, BetweenOLS

from .._adapter import LinearPredictionAdapter, VariableInfo
from ._common import (
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class LinearmodelsPanelAdapter(LinearPredictionAdapter):
    """Adapter for linearmodels panel-data result objects.

    Parameters
    ----------
    results : PanelResults, PanelEffectsResults, RandomEffectsResults, etc.
        Fitted linearmodels panel result object.
    training_data : pd.DataFrame, optional
        The data the model was fit on.  If not provided, the adapter
        attempts to reconstruct it from ``results.model.dependent`` and
        ``results.model.exog``.  Reconstruction works for formula-fit
        models but may fail for complex specifications; passing the
        original DataFrame explicitly is safer.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None, formula: Optional[str] = None):
        self.results = results
        self._training_data = self._resolve_training_data(results, training_data)
        self._exog_names = list(results.params.index)
        self._formula = getattr(results.model, "formula", None)
        self._formula_spec = None
        if formula is not None:
            from .._formula import FormulaSpec
            self._formula_spec = FormulaSpec(formula, self._training_data)
        self._model_cls = type(results.model)

    @property
    def training_data(self):
        return self._training_data

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_training_data(results, training_data: Optional[pd.DataFrame]):
        if training_data is not None:
            return training_data
        # Attempt reconstruction from model components
        try:
            dep = results.model.dependent.dataframe
            exog = results.model.exog.dataframe
            reconstructed = pd.concat([dep, exog], axis=1)
            # Drop duplicate columns (e.g. intercept already in exog)
            reconstructed = reconstructed.loc[:, ~reconstructed.columns.duplicated()]
            return reconstructed
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "training_data must be provided for this linearmodels fit; "
                "automatic reconstruction from model components failed: "
                f"{exc}"
            ) from exc

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="LinearmodelsPanelAdapter")
        super().attach(session)
        if self._formula_spec is not None:
            self._formula_spec.verify_against(self)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params.values)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.cov.values)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("unadjusted", "robust", "clustered", "kernel"):
                return self._refit_and_extract_cov(cov_type=spec_lower)
            raise ValueError(
                f"Unsupported vcov string for linearmodels: {vcov_spec!r}. "
                f"Supported: 'unadjusted', 'robust', 'clustered', 'kernel'."
            )

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                groups = vcov_spec.get("groups")
                if groups is None:
                    raise ValueError(
                        "cluster vcov requires 'groups' in the spec dict."
                    )
                return self._refit_and_extract_cov(
                    cov_type="clustered", cluster_entity=False,
                    cluster_groups=groups,
                )
            raise ValueError(
                f"Unsupported vcov dict type for linearmodels: {kind!r}"
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        if self._formula_spec is not None:
            return self._formula_spec.get_model_matrix(df)
        # Align columns to exog_names; auto-inject intercept if needed
        aligned = df.reindex(columns=self._exog_names)
        missing_cols = [
            col for col in self._exog_names
            if col not in df.columns and col not in ("const", "Intercept")
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns required by the model's exog_names: {missing_cols}. "
                f"Available columns: {list(df.columns)}."
            )
        if "const" in self._exog_names or "Intercept" in self._exog_names:
            intercept_name = "const" if "const" in self._exog_names else "Intercept"
            if intercept_name not in df.columns:
                aligned = aligned.copy()
                aligned[intercept_name] = 1.0
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

    def _refit_and_extract_cov(self, cov_type: str, cluster_entity=False, cluster_groups=None):
        """Refit the model with a specific cov_type and return its covariance."""
        if self._formula is not None:
            kwargs = {"cov_type": cov_type}
            if cluster_groups is not None:
                kwargs["clusters"] = cluster_groups
            new_results = self._model_cls.from_formula(
                self._formula, data=self._training_data,
            ).fit(**kwargs)
            return jnp.asarray(new_results.cov.values)

        raise NotImplementedError(
            "Array-fit refit with custom cov_type is not yet supported for linearmodels panel adapters."
        )

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "LinearmodelsPanelAdapter":
        """Refit the model on resampled data."""
        if self._formula is not None:
            new_results = self._model_cls.from_formula(
                self._formula, data=resampled_data,
            ).fit()
            return LinearmodelsPanelAdapter(new_results, training_data=resampled_data)

        raise NotImplementedError(
            "Array-fit refit is not yet supported for linearmodels panel adapters."
        )
