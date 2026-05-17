"""
pymargins._adapters.statsmodels_mnlogit

Concrete adapter for statsmodels MNLogit result objects.

MNLogit stores parameters as a (p, K-1) DataFrame but cov_params() as a
flat (p*(K-1), p*(K-1)) matrix in Fortran (column-major) order. This
adapter handles the flattening/unflattening and implements the softmax
prediction in pure JAX so autodiff is exact.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    extract_training_data,
    design_matrix_from_df,
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


class StatsmodelsMNLogitAdapter(ModelAdapter):
    """Adapter for statsmodels.discrete.discrete_model.MNLogitResults.

    Parameters
    ----------
    results : MNLogitResults
        Fitted statsmodels MNLogit result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)
        self._n_outcomes = int(results.model.J)
        ynames = getattr(results.model, "_ynames_map", None)
        if ynames is not None:
            self._outcome_labels = [str(ynames.get(i, str(i))) for i in range(self._n_outcomes)]
        else:
            self._outcome_labels = [str(i) for i in range(self._n_outcomes)]

    @property
    def training_data(self):
        return self._training_data

    @property
    def n_outcomes(self) -> int:
        return self._n_outcomes

    @property
    def outcome_labels(self) -> Optional[list[str]]:
        return self._outcome_labels

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
        validate_vcov_spec(vcov, adapter_name="StatsmodelsMNLogitAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        # statsmodels stores params as (p, K-1) DataFrame and cov_params()
        # as a flat (p*(K-1), p*(K-1)) matrix in Fortran order.
        params_arr = np.asarray(self.results.params)
        return jnp.asarray(params_arr.ravel(order="F"))

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
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
                    cov_type="cluster", cov_kwds={"groups": groups},
                )
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    @property
    def predict(self):
        """Identity-stable predict callable for JAX compilation caching."""
        return _mnlogit_predict(self._n_outcomes)

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

    def _refit_and_extract_cov(self, cov_type: str, cov_kwds=None) -> jnp.ndarray:
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            if cov_kwds and "groups" in cov_kwds:
                groups = cov_kwds["groups"]
                if hasattr(groups, "__len__") and len(groups) != len(self._training_data):
                    raise ValueError(
                        f"groups length ({len(groups)}) must match training_data "
                        f"length ({len(self._training_data)})."
                    )
            from statsmodels.formula.api import mnlogit as smf_mnlogit
            new_results = smf_mnlogit(
                formula, data=self._training_data,
            ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            return jnp.asarray(new_results.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog
        new_results = sm.MNLogit(endog, exog).fit(
            cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False,
        )
        return jnp.asarray(new_results.cov_params())

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsMNLogitAdapter":
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            from statsmodels.formula.api import mnlogit as smf_mnlogit
            new_results = smf_mnlogit(formula, data=resampled_data).fit(disp=False)
            return StatsmodelsMNLogitAdapter(new_results, training_data=resampled_data)

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
        new_results = sm.MNLogit(endog, exog_df).fit(disp=False)
        return StatsmodelsMNLogitAdapter(new_results, training_data=resampled_data)


@lru_cache(maxsize=None)
def _mnlogit_predict(n_outcomes: int):
    """Cached predict factory so JAX sees the same callable across refits."""

    def predict(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        X_arr = jnp.asarray(X)
        n_obs, p = X_arr.shape
        K = n_outcomes
        # Unflatten beta into (p, K-1) matrix using Fortran order
        B = beta.reshape(p, K - 1, order="F")
        eta = X_arr @ B  # (n_obs, K-1)
        # Reference category has zero linear predictor
        eta_full = jnp.concatenate([jnp.zeros((n_obs, 1)), eta], axis=1)  # (n_obs, K)
        # Softmax
        eta_max = jnp.max(eta_full, axis=1, keepdims=True)
        exp_eta = jnp.exp(eta_full - eta_max)
        probs = exp_eta / jnp.sum(exp_eta, axis=1, keepdims=True)
        return probs

    return predict
