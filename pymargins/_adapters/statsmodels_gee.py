"""
pymargins._adapters.statsmodels_gee

Concrete adapter for statsmodels GEE result objects.

Covers GEE (binary/count/continuous) with independent and various working
correlation structures.  Uses the same JAX-compatible predict as GLMAdapter
because the mean structure is identical.  The covariance is the robust
sandwich estimator from the GEE fit.

OrdinalGEE and NominalGEE are deferred to a follow-up because they require
multi-outcome predict machinery.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import GLMAdapter, VariableInfo
from .._gradients import make_glm_jvp_wrapper
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
)


class StatsmodelsGEEAdapter(GLMAdapter):
    """Adapter for statsmodels GEE results.

    Covers GEE (generalized estimating equations) with standard families
    (Binomial, Poisson, Gaussian, Gamma, etc.). Uses the same JAX-compatible
    predict as GLMAdapter because the mean structure is identical.
    The covariance is the robust sandwich estimator from the GEE fit.

    Parameters
    ----------
    results : GEEResults
        Fitted statsmodels GEE result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on. statsmodels exposes this via
        results.model.data.frame for formula-fit models, but not always
        for direct-array fits — provide explicitly in that case.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self.family = results.family
        self._predict_jax = make_glm_jvp_wrapper(self.family)
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)
        self._cov_struct = getattr(results.model, "cov_struct", None)
        self._groups = getattr(results.model, "groups", None)

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        """Validate session configuration at attach time."""
        vcov = getattr(session, "vcov_spec", None)
        if isinstance(vcov, str):
            if vcov.lower() not in (
                "naive",
                "robust",
                "robust_bc",
                "hc0",
                "hc1",
                "hc2",
                "hc3",
            ):
                raise ValueError(
                    f"StatsmodelsGEEAdapter does not support vcov={vcov!r}. "
                    f"Supported strings: 'naive', 'robust', 'robust_bc'."
                )
        if isinstance(vcov, dict):
            kind = vcov.get("type")
            if kind != "cluster":
                raise ValueError(
                    f"StatsmodelsGEEAdapter does not support vcov dict with type={kind!r}. "
                    f"Supported dict: {{'type': 'cluster', 'groups': ...}}."
                )
            groups = vcov.get("groups")
            if groups is None:
                raise ValueError(
                    "StatsmodelsGEEAdapter: cluster vcov requires 'groups' in the spec dict."
                )
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        """Return Σ̂, dispatching to the requested flavor.

        statsmodels GEE stores the sandwich covariance in ``cov_params()`` by
        default.  It also exposes:
          - ``cov_naive`` — model-based covariance
          - ``cov_robust`` — explicit robust sandwich
          - ``cov_robust_bc`` — bias-corrected robust sandwich
        """
        if vcov_spec is None:
            return jnp.asarray(self.results.cov_params())

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower == "naive":
                return jnp.asarray(self.results.cov_naive)
            if spec_lower == "robust":
                return jnp.asarray(self.results.cov_robust)
            if spec_lower == "robust_bc":
                cov = self.results.cov_robust_bc
                if cov is None:
                    raise ValueError("cov_robust_bc is not available on this fit.")
                return jnp.asarray(cov)
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                # GEE doesn't use HCx nomenclature; route to robust
                return jnp.asarray(self.results.cov_robust)
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                # GEE's robust covariance is inherently cluster-robust based on
                # the model's groups.  We return the default robust sandwich.
                groups = vcov_spec.get("groups")
                if groups is None:
                    raise ValueError("cluster vcov requires 'groups' in the spec dict.")
                return jnp.asarray(self.results.cov_robust)
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self._predict_jax(beta, X, offset)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        return design_matrix_from_df(self.results, self._exog_names, df)

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

    def _refit_and_extract_cov(self, cov_type: str) -> jnp.ndarray:
        """Refit the model with a specific cov_type and return its covariance."""
        from statsmodels.formula.api import gee as smf_gee

        model_kwargs = self._collect_original_model_kwargs()
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            new_results = smf_gee(
                formula,
                data=self._training_data,
                groups=self._groups,
                family=self.family,
                cov_struct=self._cov_struct,
                **model_kwargs,
            ).fit(cov_type=cov_type)
            return jnp.asarray(new_results.cov_params())

        # Array-fit refit
        endog = self.results.model.endog
        exog = self.results.model.exog
        new_results = sm.GEE(
            endog,
            exog,
            groups=self._groups,
            family=self.family,
            cov_struct=self._cov_struct,
            **model_kwargs,
        ).fit(cov_type=cov_type)
        return jnp.asarray(new_results.cov_params())

    def _collect_original_model_kwargs(self) -> dict:
        """Capture model-specific kwargs from the original model for refit."""
        kwargs = {}
        for attr in ("offset", "exposure"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        # GEE uses 'weights' not 'freq_weights'/'var_weights'
        weights = getattr(self.results.model, "weights", None)
        if weights is not None:
            kwargs["weights"] = weights
        return kwargs

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsGEEAdapter:
        """Refit the model on resampled data.

        Reconstructs the formula/family/cov_struct from the original results
        and fits a new GEE on the resampled data, returning a new adapter.
        """
        from statsmodels.formula.api import gee as smf_gee

        model_kwargs = self._collect_original_model_kwargs()
        if index is not None:
            for attr in ("offset", "exposure", "weights"):
                if attr in model_kwargs and hasattr(model_kwargs[attr], "__len__"):
                    model_kwargs[attr] = np.asarray(model_kwargs[attr])[index]

        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            # For formula fits, groups may be a column name or array.
            # If it is an array, resample it with index.
            groups = self._groups
            if groups is not None and not isinstance(groups, str) and index is not None:
                groups = np.asarray(groups)[index]
            new_results = smf_gee(
                formula,
                data=resampled_data,
                groups=groups,
                family=self.family,
                cov_struct=self._cov_struct,
                **model_kwargs,
            ).fit()
            return StatsmodelsGEEAdapter(new_results, training_data=resampled_data)

        # Array-fit refit: reconstruct exog and endog from resampled_data.
        endog_name = getattr(self.results.model, "endog_names", None)
        if endog_name is None:
            exog_cols = set(self._exog_names)
            exog_cols.discard("const")
            exog_cols.discard("Intercept")
            possible_endog = [c for c in resampled_data.columns if c not in exog_cols]
            if len(possible_endog) == 1:
                endog_name = possible_endog[0]
            else:
                raise NotImplementedError(
                    "Array-fit refit requires the response variable name. "
                    "Pass training_data with a clear response column, or use "
                    "formula-fit models."
                )
        exog_cols = [c for c in self._exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}. "
                "Pass training_data whose columns match the fitted exog_names."
            )
        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        if "const" in self._exog_names or "Intercept" in self._exog_names:
            intercept_name = "const" if "const" in self._exog_names else "Intercept"
            if intercept_name not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)

        groups = self._groups
        if groups is not None and index is not None:
            groups = np.asarray(groups)[index]

        new_results = sm.GEE(
            endog,
            exog_df,
            groups=groups,
            family=self.family,
            cov_struct=self._cov_struct,
            **model_kwargs,
        ).fit()
        return StatsmodelsGEEAdapter(new_results, training_data=resampled_data)
