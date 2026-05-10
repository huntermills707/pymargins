"""
pymargins._adapters.statsmodels_mixedlm

Concrete adapter for statsmodels MixedLM result objects.

Predicts on the population-average scale using only fixed effects:
    ŷ = X β̂

This is equivalent to the conditional mean for linear mixed models
(identity link). For future GLMM extensions, this would need to
integrate out random effects.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import LinearPredictionAdapter, VariableInfo
from ._common import (
    extract_training_data,
    design_matrix_from_df,
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class StatsmodelsMixedLMAdapter(LinearPredictionAdapter):
    """Adapter for statsmodels MixedLM results.

    Predicts on the population-average scale using only fixed effects:
        ŷ = X β̂

    This is equivalent to the conditional mean for linear mixed models
    (identity link). For future GLMM extensions, this would need to
    integrate out random effects.

    Parameters
    ----------
    results : MixedLMResults
        Fitted statsmodels MixedLM result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on. statsmodels exposes this via
        results.model.data.frame for formula-fit models, but not always
        for direct-array fits — provide explicitly in that case.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)
        self._exog_re = getattr(results.model, "exog_re", None)
        self._groups = getattr(results.model, "groups", None)

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        """Validate session configuration at attach time."""
        vcov = getattr(session, "vcov_spec", None)
        if isinstance(vcov, str):
            raise ValueError(
                f"StatsmodelsMixedLMAdapter does not support vcov={vcov!r}. "
                f"MixedLM only supports None or a user-supplied ndarray."
            )
        if isinstance(vcov, dict):
            kind = vcov.get("type")
            raise ValueError(
                f"StatsmodelsMixedLMAdapter does not support vcov dict with type={kind!r}. "
                f"MixedLM only supports None or a user-supplied ndarray."
            )
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        # fe_params excludes random-effects parameters
        return jnp.asarray(self.results.fe_params)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        """Return Σ̂ of the fixed effects.

        statsmodels MixedLM's ``cov_params()`` returns the covariance of the
        *full* parameter vector (fixed effects + random-effects covariance
        parameters).  We slice it to the fixed-effects block.
        """
        if vcov_spec is None:
            cov_full = self.results.cov_params()
            n_fe = len(self.results.fe_params)
            cov_fe = cov_full.iloc[:n_fe, :n_fe]
            return jnp.asarray(cov_fe.values)

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                raise ValueError(
                    f"StatsmodelsMixedLMAdapter does not support vcov={vcov_spec!r}. "
                    f"MixedLM does not expose heteroskedasticity-robust covariances."
                )
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            raise ValueError(
                f"StatsmodelsMixedLMAdapter does not support vcov dict with type={kind!r}. "
                f"MixedLM only supports None or a user-supplied ndarray."
            )

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

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

    # MixedLM does not support cov_type refitting; no _refit_and_extract_cov.

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsMixedLMAdapter":
        """Refit the model on resampled data.

        Reconstructs the formula or array inputs from the original results and
        fits a new MixedLM on the resampled data, returning a new adapter.
        """
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            groups = self._groups
            if groups is not None and not isinstance(groups, str) and index is not None:
                groups = np.asarray(groups)[index]
            new_results = smf_mixedlm(
                formula, data=resampled_data, groups=groups,
            ).fit()
            return StatsmodelsMixedLMAdapter(new_results, training_data=resampled_data)

        # Array-fit refit
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

        # Reconstruct exog_re if it was used in the original fit
        exog_re = self._exog_re
        if exog_re is not None and index is not None:
            exog_re = np.asarray(exog_re)[index]

        new_results = sm.MixedLM(
            endog, exog_df, groups=groups, exog_re=exog_re,
        ).fit()
        return StatsmodelsMixedLMAdapter(new_results, training_data=resampled_data)


# Local import to avoid polluting module namespace at load time
from statsmodels.formula.api import mixedlm as smf_mixedlm
