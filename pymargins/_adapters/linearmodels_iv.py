"""
pymargins._adapters.linearmodels_iv

Concrete adapter for linearmodels instrumental-variables models:
  IV2SLS, IVGMM, IVLIML.

Inherits predict() from LinearPredictionAdapter (simple Xβ).

The endogenous variables are instrumented; the adapter works with the
second-stage (reduced-form) coefficients, which is the conventional
scale for marginal effects in IV settings.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import LinearPredictionAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    validate_vcov_spec,
)


class LinearmodelsIVAdapter(LinearPredictionAdapter):
    """Adapter for linearmodels IV result objects.

    Parameters
    ----------
    results : IVResults, IVGMMResults
        Fitted linearmodels IV result object.
    training_data : pd.DataFrame, optional
        The data the model was fit on.  linearmodels IV models do not
        store the original DataFrame, so passing this explicitly is
        strongly recommended.  If omitted, the adapter raises an error.
    formula : str, optional
        Formula string for array-fit models. When provided, a
        :class:`pymargins._formula.FormulaSpec` is built from ``training_data``
        so that ``design_matrix_from_df`` re-evaluates derived terms
        (interactions, polynomials, splines) correctly for ``dydx()``.
    """

    def __init__(
        self,
        results,
        training_data: pd.DataFrame | None = None,
        formula: str | None = None,
    ):
        self.results = results
        if training_data is None:
            training_data = self._try_reconstruct_training_data(results)
        if training_data is None:
            raise ValueError(
                "LinearmodelsIVAdapter requires training_data to be provided "
                "explicitly because linearmodels IV models do not store the "
                "original DataFrame."
            )
        self._training_data = training_data
        self._exog_names = list(results.params.index)
        self._formula = getattr(results.model, "formula", None)
        self._model_cls = type(results.model)
        self._formula_spec = None
        if formula is not None:
            from .._formula import FormulaSpec

            self._formula_spec = FormulaSpec(formula, self._training_data)

    @staticmethod
    def _try_reconstruct_training_data(results) -> pd.DataFrame | None:
        """Attempt to reconstruct training data from linearmodels IVData objects.

        This works for OLSResults (IV2SLS without endogenous variables) where
        the model stores dependent and exog as IVData with .ndarray and .cols.
        """
        model = results.model
        try:
            dep = model.dependent
            exog = model.exog
            dfs = []
            if dep is not None and hasattr(dep, "ndarray") and hasattr(dep, "cols"):
                dep_df = pd.DataFrame(dep.ndarray, columns=dep.cols)
                dfs.append(dep_df)
            if exog is not None and hasattr(exog, "ndarray") and hasattr(exog, "cols"):
                exog_df = pd.DataFrame(exog.ndarray, columns=exog.cols)
                # Drop intercept if present to avoid duplication
                exog_df = exog_df.loc[:, exog_df.columns != "Intercept"]
                if not exog_df.empty:
                    dfs.append(exog_df)
            if len(dfs) == 0:
                return None
            df = pd.concat(dfs, axis=1)
            # Add endog and instruments if present (for full IV models)
            if hasattr(model, "endog") and model.endog is not None:
                endog = model.endog
                if hasattr(endog, "ndarray") and hasattr(endog, "cols"):
                    endog_df = pd.DataFrame(endog.ndarray, columns=endog.cols)
                    df = pd.concat([df, endog_df], axis=1)
            if hasattr(model, "instruments") and model.instruments is not None:
                instr = model.instruments
                if hasattr(instr, "ndarray") and hasattr(instr, "cols"):
                    instr_df = pd.DataFrame(instr.ndarray, columns=instr.cols)
                    df = pd.concat([df, instr_df], axis=1)
            return df
        except (AttributeError, TypeError, ValueError):
            return None

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="LinearmodelsIVAdapter")
        super().attach(session)
        if self._formula_spec is not None:
            self._formula_spec.verify_against(self)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params.values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.cov.values)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("unadjusted", "robust", "clustered", "kernel"):
                return self._refit_and_extract_cov(cov_type=spec_lower)
            raise ValueError(
                f"Unsupported vcov string for linearmodels IV: {vcov_spec!r}. "
                f"Supported: 'unadjusted', 'robust', 'clustered', 'kernel'."
            )

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                groups = vcov_spec.get("groups")
                if groups is None:
                    raise ValueError("cluster vcov requires 'groups' in the spec dict.")
                return self._refit_and_extract_cov(
                    cov_type="clustered",
                    clusters=groups,
                )
            raise ValueError(
                f"Unsupported vcov dict type for linearmodels IV: {kind!r}"
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        if self._formula_spec is not None:
            return self._formula_spec.get_model_matrix(df)
        aligned = df.reindex(columns=self._exog_names)
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
        if "const" in self._exog_names or "Intercept" in self._exog_names:
            intercept_name = "const" if "const" in self._exog_names else "Intercept"
            if intercept_name not in df.columns:
                aligned = aligned.copy()
                aligned[intercept_name] = 1.0
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

    def _refit_and_extract_cov(self, cov_type: str, clusters=None):
        """Refit the model with a specific cov_type and return its covariance."""
        if self._formula is not None:
            kwargs = {"cov_type": cov_type}
            if clusters is not None:
                kwargs["clusters"] = clusters
            new_results = self._model_cls.from_formula(
                self._formula,
                data=self._training_data,
            ).fit(**kwargs)
            return jnp.asarray(new_results.cov.values)

        raise NotImplementedError(
            "Array-fit refit with custom cov_type is not yet supported for linearmodels IV adapters."
        )

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> LinearmodelsIVAdapter:
        """Refit the model on resampled data."""
        if self._formula is not None:
            new_results = self._model_cls.from_formula(
                self._formula,
                data=resampled_data,
            ).fit()
            return LinearmodelsIVAdapter(new_results, training_data=resampled_data)

        raise NotImplementedError(
            "Array-fit refit is not yet supported for linearmodels IV adapters."
        )
