"""
pymargins._adapters.linearmodels_absorbing

Concrete adapter for linearmodels AbsorbingLS (high-dimensional fixed effects).

AbsorbingLS is the Python equivalent of reghdfe / lfe — it absorbs
high-dimensional fixed effects via the method of alternating projections
rather than including them in the design matrix.  The absorbed effects do
not appear in ``params``, so predictions and marginal effects are computed
on the non-absorbed design matrix only, exactly as with PanelOLS + FE.

This adapter supports only array-fit models (AbsorbingLS has no formula
interface).  ``training_data`` must contain the dependent, exogenous, and
absorb columns.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

from linearmodels.iv import AbsorbingLS

from .._adapter import LinearPredictionAdapter, VariableInfo
from ._common import (
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class LinearmodelsAbsorbingAdapter(LinearPredictionAdapter):
    """Adapter for linearmodels AbsorbingLS result objects.

    Parameters
    ----------
    results : AbsorbingLSResults
        Fitted AbsorbingLS result object.
    training_data : pd.DataFrame, optional
        DataFrame containing all columns used in the fit: dependent,
        exogenous, and absorb variables.  If not provided, the adapter
        attempts to reconstruct from ``results.model.dependent``,
        ``results.model.exog``, and ``results.model._absorb``.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = self._resolve_training_data(results, training_data)
        self._exog_names = list(results.params.index)
        self._dep = results.model.dependent
        self._exog = results.model.exog
        self._absorb = results.model._absorb if hasattr(results.model, "_absorb") else None

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
        try:
            dep = results.model.dependent
            dep_df = pd.DataFrame(dep.ndarray, columns=dep.cols)
            exog = results.model.exog
            exog_df = pd.DataFrame(exog.ndarray, columns=exog.cols)
            parts = [dep_df, exog_df]
            if hasattr(results.model, "_absorb") and results.model._absorb is not None:
                parts.append(results.model._absorb)
            reconstructed = pd.concat(parts, axis=1)
            reconstructed = reconstructed.loc[:, ~reconstructed.columns.duplicated()]
            return reconstructed
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "training_data must be provided for this AbsorbingLS fit; "
                "automatic reconstruction from model components failed: "
                f"{exc}"
            ) from exc

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="LinearmodelsAbsorbingAdapter")
        super().attach(session)

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
            if spec_lower in ("unadjusted", "robust", "clustered"):
                return self._refit_and_extract_cov(cov_type=spec_lower)
            raise ValueError(
                f"Unsupported vcov string for AbsorbingLS: {vcov_spec!r}. "
                f"Supported: 'unadjusted', 'robust', 'clustered'."
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
                    cov_type="clustered", clusters=groups,
                )
            raise ValueError(
                f"Unsupported vcov dict type for AbsorbingLS: {kind!r}"
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
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

    def _refit_and_extract_cov(self, cov_type: str, clusters=None):
        """Refit the model with a specific cov_type and return its covariance."""
        dep = self._dep.dataframe
        exog = self._exog.dataframe
        absorb = self._absorb
        kwargs = {}
        if clusters is not None:
            kwargs["clusters"] = clusters
        new_results = AbsorbingLS(dep, exog, absorb=absorb).fit(
            cov_type=cov_type, **kwargs
        )
        return jnp.asarray(new_results.cov.values)

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "LinearmodelsAbsorbingAdapter":
        """Refit the model on resampled data."""
        dep = resampled_data[self._dep.cols]
        exog = resampled_data[self._exog.cols]
        absorb = resampled_data[self._absorb.columns] if self._absorb is not None else None
        new_results = AbsorbingLS(dep, exog, absorb=absorb).fit()
        return LinearmodelsAbsorbingAdapter(new_results, training_data=resampled_data)
