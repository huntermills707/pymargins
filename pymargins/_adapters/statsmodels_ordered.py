"""
pymargins._adapters.statsmodels_ordered

Concrete adapter for statsmodels OrderedModel result objects.

OrderedModel predictions involve thresholds and a CDF (logit or probit),
so we use WrappedFDAdapter with the model's native predict method.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.miscmodels.ordinal_model as om

from .._adapter import VariableInfo, WrappedFDAdapter
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
    validate_vcov_spec,
)


class StatsmodelsOrderedAdapter(WrappedFDAdapter):
    """Adapter for statsmodels.miscmodels.ordinal_model.OrderedResults.

    Parameters
    ----------
    results : OrderedResults
        Fitted statsmodels OrderedModel result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        # OrderedModel's exog_names includes both covariates and thresholds.
        # The first n_exog names are the actual exog columns.
        self._exog_names = list(results.model.exog_names)
        self._n_exog = results.model.exog.shape[1]
        # Only the first n_exog names are real columns; the rest are threshold params
        self._design_exog_names = self._exog_names[: self._n_exog]
        self._n_outcomes = getattr(results.model, "k_levels", None)
        if self._n_outcomes is None:
            self._n_outcomes = int(results.model.k_extra) + 1
        endog = getattr(results.model, "endog", None)
        if endog is not None:
            try:
                uniq = np.unique(endog)
                if len(uniq) == self._n_outcomes:
                    self._outcome_labels = [str(u) for u in uniq]
                else:
                    self._outcome_labels = [str(i) for i in range(self._n_outcomes)]
            except (TypeError, ValueError):
                self._outcome_labels = [str(i) for i in range(self._n_outcomes)]
        else:
            self._outcome_labels = [str(i) for i in range(self._n_outcomes)]
        self._fd_step = 1e-6
        self._distr = getattr(results.model, "distr", "logit")

    @property
    def training_data(self):
        return self._training_data

    @property
    def n_outcomes(self) -> int:
        return self._n_outcomes

    @property
    def outcome_labels(self) -> list[str] | None:
        return self._outcome_labels

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="StatsmodelsOrderedAdapter")
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

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                if getattr(self.results, "cov_type", "").upper() == vcov_spec.upper():
                    return jnp.asarray(self.results.cov_params())
                return self._refit_and_extract_cov(cov_type=spec_lower)
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                groups = vcov_spec.get("groups")
                if groups is None:
                    raise ValueError("cluster vcov requires 'groups' in the spec dict.")
                return self._refit_and_extract_cov(
                    cov_type="cluster",
                    cov_kwds={"groups": groups},
                )
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction (via WrappedFDAdapter)
    # -----------------------------------------------------------------------

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Framework-native predict for OrderedModel.

        Receives NumPy beta and arbitrary X (may be JAX tracer during JVP,
        but the FD wrapper extracts the concrete value before calling this).
        """
        X_np = np.asarray(X)
        return np.asarray(self.results.model.predict(beta_np, exog=X_np))

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        # Use only the actual exog column names, not the threshold param names
        return design_matrix_from_df(self.results, self._design_exog_names, df)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._design_exog_names,
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

    def _refit_and_extract_cov(self, cov_type: str, cov_kwds=None) -> jnp.ndarray:
        # OrderedModel does not have a formula API in statsmodels, so we
        # always refit using the array API.
        endog = self.results.model.endog
        exog = self.results.model.exog
        new_results = om.OrderedModel(
            endog,
            exog,
            distr=self._distr,
        ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
        return jnp.asarray(new_results.cov_params())

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsOrderedAdapter:
        # OrderedModel does not have a formula API in statsmodels, so we
        # always refit using the array API.
        endog_name = getattr(self.results.model, "endog_names", None)
        if endog_name is None:
            exog_cols = set(self._design_exog_names)
            exog_cols.discard("const")
            exog_cols.discard("Intercept")
            possible_endog = [c for c in resampled_data.columns if c not in exog_cols]
            if len(possible_endog) == 1:
                endog_name = possible_endog[0]
            else:
                raise NotImplementedError(
                    "Array-fit refit requires the response variable name."
                )
        exog_cols = [c for c in self._design_exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._design_exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}."
            )
        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        new_results = om.OrderedModel(endog, exog_df, distr=self._distr).fit(disp=False)
        return StatsmodelsOrderedAdapter(new_results, training_data=resampled_data)
