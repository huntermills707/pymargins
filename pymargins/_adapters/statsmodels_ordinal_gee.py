"""
pymargins._adapters.statsmodels_ordinal_gee

Concrete adapter for statsmodels OrdinalGEE result objects.

OrdinalGEE internally expands the design matrix to (n_obs * ncut, ncut + p_std)
but ``exog_orig`` preserves the original (n_obs, p_std) design matrix.  This
adapter reconstructs the expanded linear predictor from the standard design
matrix and computes cumulative-logit category probabilities in JAX.
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
    extract_training_data,
)
from ._common import (
    design_matrix_from_df as _common_design_matrix_from_df,
)


class StatsmodelsOrdinalGEEAdapter(ModelAdapter):
    """Adapter for statsmodels OrdinalGEE results.

    Parameters
    ----------
    results : OrdinalGEEResults
        Fitted statsmodels OrdinalGEE result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)

        # Outcome levels
        self._endog_values = np.asarray(results.model.endog_values)
        self._outcome_labels = [str(v) for v in self._endog_values]
        self._K = len(self._endog_values)
        self._ncut = self._K - 1

        param_names = list(results.params.index)
        n_params = len(param_names)
        if n_params < self._ncut:
            raise ValueError(
                f"Parameter count {n_params} is less than ncut={self._ncut}."
            )
        self._p_std = n_params - self._ncut

        # Standard names are the suffix of param names after the thresholds
        self._std_exog_names = param_names[self._ncut :]

        # Model attributes for refit
        self._cov_struct = getattr(results.model, "cov_struct", None)
        self._groups = getattr(results.model, "groups", None)
        self._family = getattr(results.model, "family", None)

    @property
    def training_data(self):
        return self._training_data

    @property
    def n_outcomes(self) -> int:
        return self._K

    @property
    def outcome_labels(self) -> list[str] | None:
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
                    f"StatsmodelsOrdinalGEEAdapter does not support vcov={vcov!r}. "
                    f"Supported strings: 'naive', 'robust', 'robust_bc'."
                )
        if isinstance(vcov, dict):
            kind = vcov.get("type")
            if kind != "cluster":
                raise ValueError(
                    f"StatsmodelsOrdinalGEEAdapter does not support vcov dict with type={kind!r}. "
                    f"Supported dict: {{'type': 'cluster', 'groups': ...}}."
                )
            groups = vcov.get("groups")
            if groups is None:
                raise ValueError(
                    "StatsmodelsOrdinalGEEAdapter: cluster vcov requires 'groups' in the spec dict."
                )
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params.values)

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
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
                return jnp.asarray(self.results.cov_robust)
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
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
        """Return class probabilities (n_obs, K) for a standard design matrix."""
        X_arr = jnp.asarray(X)
        n_obs = X_arr.shape[0]
        thresholds = beta[: self._ncut]  # (ncut,)
        reg_coefs = beta[self._ncut :]  # (p_std,)
        eta = X_arr @ reg_coefs  # (n_obs,)
        # cumulative logits: threshold_k + eta_i
        lpr = thresholds[None, :] + eta[:, None]  # (n_obs, ncut)
        cumprobs = 1.0 / (1.0 + jnp.exp(-lpr))  # (n_obs, ncut)
        # Category probabilities from cumulative probabilities
        probs = jnp.zeros((n_obs, self._K))
        probs = probs.at[:, 0].set(1.0 - cumprobs[:, 0])
        for k in range(1, self._ncut):
            probs = probs.at[:, k].set(cumprobs[:, k - 1] - cumprobs[:, k])
        probs = probs.at[:, self._ncut].set(cumprobs[:, self._ncut - 1])
        return probs

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            rhs = formula.split("~", 1)[1].strip()
            from patsy import dmatrix

            X = np.asarray(dmatrix(rhs, df, return_type="matrix"))
            return jnp.asarray(X)
        return _common_design_matrix_from_df(self.results, self._std_exog_names, df)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._std_exog_names,
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

    def _collect_original_model_kwargs(self) -> dict:
        kwargs = {}
        for attr in ("offset", "exposure"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        weights = getattr(self.results.model, "weights", None)
        if weights is not None:
            kwargs["weights"] = weights
        return kwargs

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsOrdinalGEEAdapter:
        from statsmodels.formula.api import ordinal_gee as smf_ordinal_gee

        model_kwargs = self._collect_original_model_kwargs()
        if index is not None:
            for attr in ("offset", "exposure", "weights"):
                if attr in model_kwargs and hasattr(model_kwargs[attr], "__len__"):
                    model_kwargs[attr] = np.asarray(model_kwargs[attr])[index]

        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            groups = self._groups
            if groups is not None and not isinstance(groups, str) and index is not None:
                groups = np.asarray(groups)[index]
            new_results = smf_ordinal_gee(
                formula,
                groups,
                data=resampled_data,
                family=self._family,
                cov_struct=self._cov_struct,
                **model_kwargs,
            ).fit()
            return StatsmodelsOrdinalGEEAdapter(
                new_results, training_data=resampled_data
            )

        # Array-fit refit
        endog_name = getattr(self.results.model, "endog_names", None)
        if endog_name is None:
            exog_cols = set(self._std_exog_names)
            exog_cols.discard("const")
            exog_cols.discard("Intercept")
            possible_endog = [c for c in resampled_data.columns if c not in exog_cols]
            if len(possible_endog) == 1:
                endog_name = possible_endog[0]
            else:
                raise NotImplementedError(
                    "Array-fit refit requires the response variable name."
                )
        exog_cols = [c for c in self._std_exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._std_exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}."
            )
        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        if "const" in self._std_exog_names or "Intercept" in self._std_exog_names:
            intercept_name = "const" if "const" in self._std_exog_names else "Intercept"
            if intercept_name not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)

        groups = self._groups
        if groups is not None and index is not None:
            groups = np.asarray(groups)[index]

        new_results = sm.OrdinalGEE(
            endog,
            exog_df,
            groups=groups,
            family=self._family,
            cov_struct=self._cov_struct,
            **model_kwargs,
        ).fit()
        return StatsmodelsOrdinalGEEAdapter(new_results, training_data=resampled_data)
