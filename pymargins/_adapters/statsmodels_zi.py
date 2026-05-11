"""
pymargins._adapters.statsmodels_zi

Concrete adapter for statsmodels zero-inflated count model results:
  - ZeroInflatedPoisson
  - ZeroInflatedNegativeBinomialP
  - ZeroInflatedGeneralizedPoisson

Predictions are the conditional mean:
    E[y | x, z] = (1 - π(z)) * μ(x)
where π(z) = logistic(z' γ) and μ(x) = exp(x' β).

Extra parameters (alpha, p) are present in the coefficient vector but do
not enter the mean prediction.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import ModelAdapter, VariableInfo
from ._common import (
    extract_training_data,
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
)


_EXTRA_PARAM_NAMES = {"alpha", "p", "theta"}


class StatsmodelsZIAdapter(ModelAdapter):
    """Adapter for statsmodels zero-inflated count model results.

    Covers ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, and
    ZeroInflatedGeneralizedPoisson.

    Parameters
    ----------
    results : fitted statsmodels ZI result object

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)

        mod = results.model
        # Inflation design matrix info
        self._k_inflate = mod.model_infl.exog.shape[1]
        self._infl_names = list(mod.model_infl.exog_names)

        # Count design matrix info
        self._k_count = mod.exog.shape[1]

        # Full parameter names (inflation + count + extra)
        all_names = list(results.params.index)
        # Inflation params are first in the vector and prefixed with "inflate_"
        self._infl_param_names = [
            n for n in all_names if n.startswith("inflate_")
        ]
        # Count params are the non-inflate params that map to exog columns
        count_and_extra = [n for n in all_names if not n.startswith("inflate_")]
        self._count_param_names = count_and_extra[: self._k_count]
        self._extra_param_names = count_and_extra[self._k_count :]

        # exog_names for the full concatenated design matrix [infl | count]
        self._exog_names = self._infl_names + self._count_param_names

        # Store whether count model was formula-fit (has patsy design_info)
        self._has_design_info = (
            hasattr(mod, "data")
            and mod.data is not None
            and hasattr(mod.data, "design_info")
            and mod.data.design_info is not None
        )

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
        validate_vcov_spec(vcov, adapter_name="StatsmodelsZIAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

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

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        X_arr = jnp.asarray(X)
        k_infl = self._k_inflate
        k_count = self._k_count

        X_infl = X_arr[:, :k_infl]
        X_count = X_arr[:, k_infl : k_infl + k_count]

        beta_infl = beta[:k_infl]
        beta_count = beta[k_infl : k_infl + k_count]

        eta_infl = X_infl @ beta_infl
        pi = 1.0 / (1.0 + jnp.exp(-eta_infl))

        eta_count = X_count @ beta_count
        if offset is not None:
            eta_count = eta_count + jnp.asarray(offset)
        mu = jnp.exp(eta_count)

        return (1.0 - pi) * mu

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        # Build inflation design matrix
        X_infl = self._build_infl_matrix(df)
        # Build count design matrix
        X_count = self._build_count_matrix(df)
        return jnp.hstack([X_infl, X_count])

    def _build_infl_matrix(self, df: pd.DataFrame) -> jnp.ndarray:
        names = self._infl_names
        aligned = df.reindex(columns=names)
        missing = [n for n in names if n not in df.columns and n not in ("const", "Intercept")]
        if missing:
            raise ValueError(
                f"Missing inflation columns: {missing}. "
                f"Available: {list(df.columns)}."
            )
        if "const" in names or "Intercept" in names:
            intercept_name = "const" if "const" in names else "Intercept"
            if intercept_name not in df.columns:
                aligned = aligned.copy()
                aligned[intercept_name] = 1.0
        aligned = aligned[names]
        return jnp.asarray(aligned.values)

    def _build_count_matrix(self, df: pd.DataFrame) -> jnp.ndarray:
        if self._has_design_info:
            from patsy import dmatrix
            design_info = self.results.model.data.design_info
            X_np = np.asarray(dmatrix(design_info, df, return_type="matrix"))
            return jnp.asarray(X_np)

        names = self._count_param_names
        aligned = df.reindex(columns=names)
        missing = [n for n in names if n not in df.columns and n not in ("const", "Intercept")]
        if missing:
            raise ValueError(
                f"Missing count columns: {missing}. "
                f"Available: {list(df.columns)}."
            )
        if "const" in names or "Intercept" in names:
            intercept_name = "const" if "const" in names else "Intercept"
            if intercept_name not in df.columns:
                aligned = aligned.copy()
                aligned[intercept_name] = 1.0
        aligned = aligned[names]
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

    def _refit_and_extract_cov(self, cov_type: str, cov_kwds=None) -> jnp.ndarray:
        formula = getattr(self.results.model, "formula", None)
        model_cls_name = type(self.results.model).__name__
        fit_kwargs = self._collect_original_fit_kwargs()

        if formula is not None:
            if cov_kwds and "groups" in cov_kwds:
                groups = cov_kwds["groups"]
                if hasattr(groups, "__len__") and len(groups) != len(self._training_data):
                    raise ValueError(
                        f"groups length ({len(groups)}) must match training_data "
                        f"length ({len(self._training_data)})."
                    )

            from statsmodels.discrete.count_model import (
                ZeroInflatedPoisson,
                ZeroInflatedNegativeBinomialP,
                ZeroInflatedGeneralizedPoisson,
            )

            infl_cols = self._infl_names.copy()
            if "Intercept" in infl_cols:
                infl_cols.remove("Intercept")
            if "const" in infl_cols:
                infl_cols.remove("const")

            exog_infl = self._training_data[infl_cols] if infl_cols else None
            kwargs = {"exog_infl": exog_infl, **fit_kwargs}

            if model_cls_name == "ZeroInflatedPoisson":
                new_results = ZeroInflatedPoisson.from_formula(
                    formula, data=self._training_data, **kwargs,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            elif model_cls_name == "ZeroInflatedNegativeBinomialP":
                new_results = ZeroInflatedNegativeBinomialP.from_formula(
                    formula, data=self._training_data, **kwargs,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            elif model_cls_name == "ZeroInflatedGeneralizedPoisson":
                new_results = ZeroInflatedGeneralizedPoisson.from_formula(
                    formula, data=self._training_data, **kwargs,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
            else:
                raise ValueError(f"Unknown ZI model class: {model_cls_name}")
            return jnp.asarray(new_results.cov_params())

        # Array-fit refit
        endog = self.results.model.endog
        exog = self.results.model.exog
        exog_infl = self.results.model.exog_infl
        if model_cls_name == "ZeroInflatedPoisson":
            new_results = sm.ZeroInflatedPoisson(
                endog, exog, exog_infl=exog_infl, **fit_kwargs,
            ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
        elif model_cls_name == "ZeroInflatedNegativeBinomialP":
            new_results = sm.ZeroInflatedNegativeBinomialP(
                endog, exog, exog_infl=exog_infl, **fit_kwargs,
            ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
        elif model_cls_name == "ZeroInflatedGeneralizedPoisson":
            new_results = sm.ZeroInflatedGeneralizedPoisson(
                endog, exog, exog_infl=exog_infl, **fit_kwargs,
            ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, disp=False)
        else:
            raise ValueError(f"Unknown ZI model class: {model_cls_name}")
        return jnp.asarray(new_results.cov_params())

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsZIAdapter":
        formula = getattr(self.results.model, "formula", None)
        fit_kwargs = self._collect_original_fit_kwargs()
        if index is not None:
            for attr in ("offset", "exposure"):
                if attr in fit_kwargs and hasattr(fit_kwargs[attr], "__len__"):
                    fit_kwargs[attr] = np.asarray(fit_kwargs[attr])[index]

        model_cls_name = type(self.results.model).__name__

        if formula is not None:
            from statsmodels.discrete.count_model import (
                ZeroInflatedPoisson,
                ZeroInflatedNegativeBinomialP,
                ZeroInflatedGeneralizedPoisson,
            )

            infl_cols = self._infl_names.copy()
            if "Intercept" in infl_cols:
                infl_cols.remove("Intercept")
            if "const" in infl_cols:
                infl_cols.remove("const")

            exog_infl = resampled_data[infl_cols] if infl_cols else None
            kwargs = {"exog_infl": exog_infl, **fit_kwargs}

            if model_cls_name == "ZeroInflatedPoisson":
                new_results = ZeroInflatedPoisson.from_formula(
                    formula, data=resampled_data, **kwargs,
                ).fit(disp=False)
            elif model_cls_name == "ZeroInflatedNegativeBinomialP":
                new_results = ZeroInflatedNegativeBinomialP.from_formula(
                    formula, data=resampled_data, **kwargs,
                ).fit(disp=False)
            elif model_cls_name == "ZeroInflatedGeneralizedPoisson":
                new_results = ZeroInflatedGeneralizedPoisson.from_formula(
                    formula, data=resampled_data, **kwargs,
                ).fit(disp=False)
            else:
                raise ValueError(f"Unknown ZI model class: {model_cls_name}")
            return StatsmodelsZIAdapter(new_results, training_data=resampled_data)

        # Array-fit refit
        endog_name = getattr(self.results.model, "endog_names", None)
        if endog_name is None:
            exog_cols = set(self._count_param_names + self._infl_names)
            exog_cols.discard("const")
            exog_cols.discard("Intercept")
            possible_endog = [c for c in resampled_data.columns if c not in exog_cols]
            if len(possible_endog) == 1:
                endog_name = possible_endog[0]
            else:
                raise NotImplementedError(
                    "Array-fit refit requires the response variable name."
                )

        # Build count exog
        count_cols = [c for c in self._count_param_names if c in resampled_data.columns]
        if not count_cols:
            raise ValueError(
                f"None of the count param names {self._count_param_names} are present."
            )
        exog_df = resampled_data[count_cols]
        if "const" in self._count_param_names or "Intercept" in self._count_param_names:
            intercept_name = "const" if "const" in self._count_param_names else "Intercept"
            if intercept_name not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)

        # Build inflation exog
        infl_cols = [c for c in self._infl_names if c in resampled_data.columns]
        if not infl_cols:
            raise ValueError(
                f"None of the inflation names {self._infl_names} are present."
            )
        exog_infl_df = resampled_data[infl_cols]
        if "const" in self._infl_names or "Intercept" in self._infl_names:
            intercept_name = "const" if "const" in self._infl_names else "Intercept"
            if intercept_name not in exog_infl_df.columns:
                exog_infl_df = exog_infl_df.copy()
                exog_infl_df.insert(0, "const", 1.0)

        endog = resampled_data[endog_name].values
        if model_cls_name == "ZeroInflatedPoisson":
            new_results = sm.ZeroInflatedPoisson(
                endog, exog_df, exog_infl=exog_infl_df, **fit_kwargs,
            ).fit(disp=False)
        elif model_cls_name == "ZeroInflatedNegativeBinomialP":
            new_results = sm.ZeroInflatedNegativeBinomialP(
                endog, exog_df, exog_infl=exog_infl_df, **fit_kwargs,
            ).fit(disp=False)
        elif model_cls_name == "ZeroInflatedGeneralizedPoisson":
            new_results = sm.ZeroInflatedGeneralizedPoisson(
                endog, exog_df, exog_infl=exog_infl_df, **fit_kwargs,
            ).fit(disp=False)
        else:
            raise ValueError(f"Unknown ZI model class: {model_cls_name}")
        return StatsmodelsZIAdapter(new_results, training_data=resampled_data)

    def _collect_original_fit_kwargs(self) -> dict:
        """Capture model-specific kwargs from the original fit for refit."""
        kwargs = {}
        for attr in ("offset", "exposure"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        # Preserve dispersion / shape parameters
        for attr in ("p",):
            if hasattr(self.results.model, attr):
                kwargs[attr] = getattr(self.results.model, attr)
        if hasattr(self.results.model, "inflation"):
            kwargs["inflation"] = self.results.model.inflation
        return kwargs
