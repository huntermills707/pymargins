"""
pymargins._adapters.statsmodels_nominal_gee

Concrete adapter for statsmodels NominalGEE result objects.

NominalGEE internally expands the design matrix to (n_obs * ncut, p_std * ncut)
but ``exog_orig`` preserves the original (n_obs, p_std) design matrix.  This
adapter uses the original matrix for prediction and reconstructs the
multinomial linear predictors in JAX.
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
    design_matrix_from_df as _common_design_matrix_from_df,
    column_index_of_variable,
    build_variable_metadata,
)


class StatsmodelsNominalGEEAdapter(ModelAdapter):
    """Adapter for statsmodels NominalGEE results.

    Parameters
    ----------
    results : NominalGEEResults
        Fitted statsmodels NominalGEE result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self._training_data = extract_training_data(results, training_data)

        # Outcome levels
        self._endog_values = np.asarray(results.model.endog_values)
        self._outcome_labels = [str(v) for v in self._endog_values]
        self._K = len(self._endog_values)
        self._ncut = self._K - 1

        # Parse parameter names to extract standard exog names and category mapping
        param_names = list(results.params.index)
        n_params = len(param_names)
        if n_params % self._ncut != 0:
            raise ValueError(
                f"Parameter count {n_params} is not divisible by ncut={self._ncut}."
            )
        self._p_std = n_params // self._ncut

        # Standard names from the first block of params
        self._std_exog_names = []
        for name in param_names[:self._p_std]:
            base = name.rsplit("[", 1)[0]
            self._std_exog_names.append(base)

        # Categories with explicit parameters (one per column of beta)
        self._param_cats = []
        for i in range(self._ncut):
            idx = i * self._p_std
            name = param_names[idx]
            if "[" not in name or "]" not in name:
                raise ValueError(
                    f"Expected category suffix in param name {name!r}."
                )
            cat_str = name.split("[", 1)[1].rsplit("]", 1)[0]
            cat = float(cat_str) if "." in cat_str else int(cat_str)
            self._param_cats.append(cat)

        self._ref_cat = None
        for val in self._endog_values:
            if val not in self._param_cats:
                self._ref_cat = val
                break

        self._cat_to_col = {cat: i for i, cat in enumerate(self._param_cats)}

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
        if isinstance(vcov, str):
            if vcov.lower() not in ("naive", "robust", "robust_bc", "hc0", "hc1", "hc2", "hc3"):
                raise ValueError(
                    f"StatsmodelsNominalGEEAdapter does not support vcov={vcov!r}. "
                    f"Supported strings: 'naive', 'robust', 'robust_bc'."
                )
        if isinstance(vcov, dict):
            kind = vcov.get("type")
            if kind != "cluster":
                raise ValueError(
                    f"StatsmodelsNominalGEEAdapter does not support vcov dict with type={kind!r}. "
                    f"Supported dict: {{'type': 'cluster', 'groups': ...}}."
                )
            groups = vcov.get("groups")
            if groups is None:
                raise ValueError(
                    "StatsmodelsNominalGEEAdapter: cluster vcov requires 'groups' in the spec dict."
                )
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params.values)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
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
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return class probabilities (n_obs, K) for a standard design matrix."""
        X_arr = jnp.asarray(X)
        n_obs = X_arr.shape[0]
        # Unflatten beta into (p_std, ncut) in Fortran order
        B = beta.reshape(self._p_std, self._ncut, order="F")
        # Compute linear predictor for each non-reference category
        etas = []
        for cat in self._endog_values:
            if cat in self._cat_to_col:
                col = self._cat_to_col[cat]
                etas.append(X_arr @ B[:, col])
            else:
                etas.append(jnp.zeros(n_obs))
        eta_full = jnp.stack(etas, axis=1)  # (n_obs, K)
        # Stable softmax
        eta_max = jnp.max(eta_full, axis=1, keepdims=True)
        exp_eta = jnp.exp(eta_full - eta_max)
        probs = exp_eta / jnp.sum(exp_eta, axis=1, keepdims=True)
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
        return _common_design_matrix_from_df(
            self.results, self._std_exog_names, df
        )

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._std_exog_names, self.variable_metadata(), variable_name,
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

    def refit(self, resampled_data: pd.DataFrame, *, index=None) -> "StatsmodelsNominalGEEAdapter":
        from statsmodels.formula.api import nominal_gee as smf_nominal_gee

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
            new_results = smf_nominal_gee(
                formula,
                groups,
                data=resampled_data,
                cov_struct=self._cov_struct,
                **model_kwargs,
            ).fit()
            return StatsmodelsNominalGEEAdapter(new_results, training_data=resampled_data)

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

        new_results = sm.NominalGEE(
            endog, exog_df, groups=groups,
            cov_struct=self._cov_struct,
            **model_kwargs,
        ).fit()
        return StatsmodelsNominalGEEAdapter(new_results, training_data=resampled_data)
