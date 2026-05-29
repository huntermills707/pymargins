"""
pymargins._adapters.statsmodels_discrete_binary

Concrete adapter for statsmodels discrete binary result objects:
  - Logit
  - Probit

These models predict the conditional probability via a link function
(expit for Logit, ndtr for Probit).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jax.scipy.special import expit, ndtr

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
    validate_vcov_spec,
)


class StatsmodelsDiscreteBinaryAdapter(ModelAdapter):
    """Adapter for statsmodels discrete binary model results (Logit, Probit).

    Parameters
    ----------
    results : fitted statsmodels discrete binary result object

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)
        self._model_cls_name = type(results.model).__name__

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
        validate_vcov_spec(vcov, adapter_name="StatsmodelsDiscreteBinaryAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def score_obs(self) -> np.ndarray:
        """Per-observation score ∂ℓ_i/∂β, shape (n_obs, p)."""
        return np.asarray(self.results.model.score_obs(self.results.params))

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
    # Prediction
    # -----------------------------------------------------------------------

    @property
    def predict(self):
        """Identity-stable predict callable for JAX compilation caching."""
        return _discrete_binary_predict(self._model_cls_name)

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

    def _refit_and_extract_cov(self, cov_type: str, cov_kwds=None) -> jnp.ndarray:
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            if cov_kwds and "groups" in cov_kwds:
                groups = cov_kwds["groups"]
                if hasattr(groups, "__len__") and len(groups) != len(
                    self._training_data
                ):
                    raise ValueError(
                        f"groups length ({len(groups)}) must match training_data "
                        f"length ({len(self._training_data)})."
                    )
            fit_kwargs = self._collect_original_fit_kwargs()
            if self._model_cls_name == "Logit":
                from statsmodels.formula.api import logit as smf_logit

                new_results = smf_logit(
                    formula,
                    data=self._training_data,
                ).fit(
                    cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False, **fit_kwargs
                )
            elif self._model_cls_name == "Probit":
                from statsmodels.formula.api import probit as smf_probit

                new_results = smf_probit(
                    formula,
                    data=self._training_data,
                ).fit(
                    cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False, **fit_kwargs
                )
            else:
                raise ValueError(f"Unknown model class: {self._model_cls_name}")
            return jnp.asarray(new_results.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog
        fit_kwargs = self._collect_original_fit_kwargs()
        if self._model_cls_name == "Logit":
            new_results = sm.Logit(endog, exog, **fit_kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        elif self._model_cls_name == "Probit":
            new_results = sm.Probit(endog, exog, **fit_kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        else:
            raise ValueError(f"Unknown model class: {self._model_cls_name}")
        return jnp.asarray(new_results.cov_params())

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsDiscreteBinaryAdapter:
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            fit_kwargs = self._collect_original_fit_kwargs()
            if index is not None:
                for attr in ("offset", "exposure", "freq_weights", "var_weights"):
                    if attr in fit_kwargs and hasattr(fit_kwargs[attr], "__len__"):
                        fit_kwargs[attr] = np.asarray(fit_kwargs[attr])[index]
            if self._model_cls_name == "Logit":
                from statsmodels.formula.api import logit as smf_logit

                new_results = smf_logit(
                    formula,
                    data=resampled_data,
                ).fit(disp=False, **fit_kwargs)
            elif self._model_cls_name == "Probit":
                from statsmodels.formula.api import probit as smf_probit

                new_results = smf_probit(
                    formula,
                    data=resampled_data,
                ).fit(disp=False, **fit_kwargs)
            else:
                raise ValueError(f"Unknown model class: {self._model_cls_name}")
            return StatsmodelsDiscreteBinaryAdapter(
                new_results, training_data=resampled_data
            )

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
        fit_kwargs = self._collect_original_fit_kwargs()
        if index is not None:
            for attr in ("offset", "exposure", "freq_weights", "var_weights"):
                if attr in fit_kwargs and hasattr(fit_kwargs[attr], "__len__"):
                    fit_kwargs[attr] = np.asarray(fit_kwargs[attr])[index]
        if self._model_cls_name == "Logit":
            new_results = sm.Logit(endog, exog_df, **fit_kwargs).fit(disp=False)
        elif self._model_cls_name == "Probit":
            new_results = sm.Probit(endog, exog_df, **fit_kwargs).fit(disp=False)
        else:
            raise ValueError(f"Unknown model class: {self._model_cls_name}")
        return StatsmodelsDiscreteBinaryAdapter(
            new_results, training_data=resampled_data
        )

    def _collect_original_fit_kwargs(self) -> dict:
        """Capture model-specific kwargs from the original fit for refit."""
        kwargs = {}
        for attr in ("offset", "exposure", "freq_weights", "var_weights"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        return kwargs


@cache
def _discrete_binary_predict(model_cls_name: str):
    """Cached predict factory so JAX sees the same callable across refits."""

    def predict(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        eta = jnp.asarray(X) @ beta
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        if model_cls_name == "Probit":
            return ndtr(eta)
        return expit(eta)

    return predict
