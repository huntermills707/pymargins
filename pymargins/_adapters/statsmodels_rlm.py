"""
pymargins._adapters.statsmodels_rlm

Concrete adapter for statsmodels RLM (robust linear model) result objects.
Inherits predict() from LinearPredictionAdapter (simple X @ beta).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import LinearPredictionAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
)


class StatsmodelsRLMAdapter(LinearPredictionAdapter):
    """Adapter for statsmodels RLM result objects.

    Parameters
    ----------
    results : RLMResults
        Fitted statsmodels RLM result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        if vcov is not None:
            # RLM does not support HC/cluster refit in the same way as OLS/GLM.
            # Only None and user-supplied ndarray are supported.
            if not isinstance(vcov, (np.ndarray, jnp.ndarray)):
                raise ValueError(
                    "StatsmodelsRLMAdapter only supports vcov=None or a "
                    "user-supplied ndarray. HC/cluster robust covariance is "
                    "not available for RLM."
                )
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        if vcov_spec is None:
            return jnp.asarray(self.results.cov_params())

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        raise ValueError(
            "StatsmodelsRLMAdapter only supports vcov=None or a user-supplied ndarray. "
            f"Got {vcov_spec!r}."
        )

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

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsRLMAdapter:
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            from statsmodels.formula.api import rlm as smf_rlm

            new_results = smf_rlm(formula, data=resampled_data).fit()
            return StatsmodelsRLMAdapter(new_results, training_data=resampled_data)

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
                    "Array-fit refit requires the response variable name."
                )
        exog_cols = [c for c in self._exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}."
            )
        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        if "const" in self._exog_names or "Intercept" in self._exog_names:
            intercept_name = "const" if "const" in self._exog_names else "Intercept"
            if intercept_name not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)
        new_results = sm.RLM(endog, exog_df).fit()
        return StatsmodelsRLMAdapter(new_results, training_data=resampled_data)
