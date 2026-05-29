"""
pymargins._adapters.statsmodels_discrete_count

Concrete adapter for statsmodels discrete count result objects:
  - Poisson
  - NegativeBinomial
  - NegativeBinomialP
  - GeneralizedPoisson

These models all predict the conditional mean as exp(X β_mean), where
β_mean is the first p coefficients. Extra parameters (alpha, dispersion)
are present in the coefficient vector but not used in the mean prediction.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
    validate_vcov_spec,
)


class StatsmodelsDiscreteCountAdapter(ModelAdapter):
    """Adapter for statsmodels discrete count model results.

    Covers Poisson, NegativeBinomial, NegativeBinomialP, and
    GeneralizedPoisson. All predict the conditional mean via a log link.

    Parameters
    ----------
    results : fitted statsmodels discrete count result object

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        # Statsmodels appends extra param names (e.g. 'alpha') to exog_names.
        # Only the first n_exog names correspond to design-matrix columns.
        n_exog = results.model.exog.shape[1]
        self._exog_names = list(results.model.exog_names[:n_exog])

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
        validate_vcov_spec(vcov, adapter_name="StatsmodelsDiscreteCountAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def score_obs(self) -> np.ndarray:
        """Per-observation score ∂ℓ_i/∂β, shape (n_obs, p).

        For models with an extra dispersion parameter (e.g. NegativeBinomial),
        the trailing column corresponds to that parameter and aligns with the
        full ``params``/``cov_params`` dimension.
        """
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

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        X_arr = jnp.asarray(X)
        p = X_arr.shape[1]
        beta_mean = beta[:p]
        eta = X_arr @ beta_mean
        if offset is not None:
            eta = eta + jnp.asarray(offset)
        return jnp.exp(eta)

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
            model_cls_name = type(self.results.model).__name__
            if model_cls_name == "Poisson":
                from statsmodels.formula.api import poisson as smf_poisson

                new_results = smf_poisson(
                    formula,
                    data=self._training_data,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            elif model_cls_name == "NegativeBinomial":
                from statsmodels.formula.api import negativebinomial as smf_nb

                new_results = smf_nb(
                    formula,
                    data=self._training_data,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            elif model_cls_name == "NegativeBinomialP":
                from statsmodels.discrete.discrete_model import NegativeBinomialP

                new_results = NegativeBinomialP.from_formula(
                    formula,
                    data=self._training_data,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            elif model_cls_name == "GeneralizedPoisson":
                from statsmodels.discrete.discrete_model import GeneralizedPoisson

                new_results = GeneralizedPoisson.from_formula(
                    formula,
                    data=self._training_data,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            else:
                raise ValueError(f"Unknown model class: {model_cls_name}")
            return jnp.asarray(new_results.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog
        model_cls_name = type(self.results.model).__name__
        kwargs = self._collect_original_fit_kwargs()
        if model_cls_name == "Poisson":
            new_results = sm.Poisson(endog, exog, **kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        elif model_cls_name == "NegativeBinomial":
            new_results = sm.NegativeBinomial(endog, exog, **kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        elif model_cls_name == "NegativeBinomialP":
            new_results = sm.NegativeBinomialP(endog, exog, **kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        elif model_cls_name == "GeneralizedPoisson":
            new_results = sm.GeneralizedPoisson(endog, exog, **kwargs).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
                disp=False,
            )
        else:
            raise ValueError(f"Unknown model class: {model_cls_name}")
        return jnp.asarray(new_results.cov_params())

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsDiscreteCountAdapter:
        formula = getattr(self.results.model, "formula", None)
        fit_kwargs = self._collect_original_fit_kwargs()
        if index is not None:
            for attr in ("offset", "exposure"):
                if attr in fit_kwargs and hasattr(fit_kwargs[attr], "__len__"):
                    fit_kwargs[attr] = np.asarray(fit_kwargs[attr])[index]
        if formula is not None:
            model_cls_name = type(self.results.model).__name__
            if model_cls_name == "Poisson":
                from statsmodels.formula.api import poisson as smf_poisson

                new_results = smf_poisson(formula, data=resampled_data).fit(
                    disp=False, **fit_kwargs
                )
            elif model_cls_name == "NegativeBinomial":
                from statsmodels.formula.api import negativebinomial as smf_nb

                new_results = smf_nb(formula, data=resampled_data).fit(
                    disp=False, **fit_kwargs
                )
            elif model_cls_name == "NegativeBinomialP":
                from statsmodels.discrete.discrete_model import NegativeBinomialP

                new_results = NegativeBinomialP.from_formula(
                    formula,
                    data=resampled_data,
                ).fit(disp=False, **fit_kwargs)
            elif model_cls_name == "GeneralizedPoisson":
                from statsmodels.discrete.discrete_model import GeneralizedPoisson

                new_results = GeneralizedPoisson.from_formula(
                    formula,
                    data=resampled_data,
                ).fit(disp=False, **fit_kwargs)
            else:
                raise ValueError(f"Unknown model class: {model_cls_name}")
            return StatsmodelsDiscreteCountAdapter(
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
        model_cls_name = type(self.results.model).__name__
        if model_cls_name == "Poisson":
            new_results = sm.Poisson(endog, exog_df, **fit_kwargs).fit(disp=False)
        elif model_cls_name == "NegativeBinomial":
            new_results = sm.NegativeBinomial(endog, exog_df, **fit_kwargs).fit(
                disp=False
            )
        elif model_cls_name == "NegativeBinomialP":
            new_results = sm.NegativeBinomialP(endog, exog_df, **fit_kwargs).fit(
                disp=False
            )
        elif model_cls_name == "GeneralizedPoisson":
            new_results = sm.GeneralizedPoisson(endog, exog_df, **fit_kwargs).fit(
                disp=False
            )
        else:
            raise ValueError(f"Unknown model class: {model_cls_name}")
        return StatsmodelsDiscreteCountAdapter(
            new_results, training_data=resampled_data
        )

    def _collect_original_fit_kwargs(self) -> dict:
        """Capture model-specific kwargs from the original fit for refit."""
        kwargs = {}
        for attr in ("offset", "exposure"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        # Preserve model-specific parameters like p for GeneralizedPoisson / NegativeBinomialP
        if hasattr(self.results.model, "parameter"):
            kwargs["p"] = self.results.model.parameter
        elif hasattr(self.results.model, "p"):
            kwargs["p"] = self.results.model.p
        return kwargs
