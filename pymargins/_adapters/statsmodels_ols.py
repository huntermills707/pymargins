"""
pymargins._adapters.statsmodels_ols

Concrete adapter for statsmodels OLS, WLS, and GLS result objects.
Inherits predict() from LinearPredictionAdapter (simple Xβ).
"""

from __future__ import annotations

import warnings
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
    validate_vcov_spec,
)


class StatsmodelsOLSAdapter(LinearPredictionAdapter):
    """Adapter for statsmodels OLS/WLS/GLS result objects.

    Parameters
    ----------
    results : RegressionResults
        Fitted statsmodels OLS/WLS/GLS result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on. statsmodels exposes this via
        results.model.data.frame for formula-fit models.

    formula : str, optional
        Formula string for array-fit models. When provided, a
        :class:`pymargins._formula.FormulaSpec` is built from ``training_data``
        so that ``design_matrix_from_df`` re-evaluates derived terms
        (interactions, polynomials, splines) correctly for ``dydx()``.
    """

    def __init__(
        self,
        results,
        training_data: pd.DataFrame | None = None,
        formula: str | None = None,
    ):
        self.results = results
        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)
        self._formula_spec = None
        if formula is not None:
            from .._formula import FormulaSpec

            self._formula_spec = FormulaSpec(formula, self._training_data)

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        """Validate session configuration at attach time."""
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="StatsmodelsOLSAdapter")
        super().attach(session)
        if self._formula_spec is not None:
            self._formula_spec.verify_against(self)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def influence(self) -> jnp.ndarray | None:
        """Per-observation influence of β̂: ψ^β = score_obs @ cov_params.

        Reuses the same bread and scores as the survey linearization path.
        """
        scores = np.asarray(self.score_obs())
        cov = np.asarray(self.covariance())
        return jnp.asarray(scores @ cov)

    def score_obs(self) -> np.ndarray:
        """Per-observation score ∂ℓ_i/∂β = w_i x_i (y_i − x_iᵀβ̂) / σ̂², shape (n, p).

        OLS/WLS has no ``model.score_obs``; the Gaussian score is formed from
        the residuals directly. The ``results.scale`` factor cancels against
        the same factor in ``cov_params`` when forming the influence function.
        For WLS the observation weights are included in the score.
        """
        model_cls = type(self.results.model).__name__
        if model_cls not in ("OLS", "WLS"):
            raise NotImplementedError(
                f"score_obs() is only implemented for OLS/WLS, not {model_cls}."
            )
        exog = np.asarray(self.results.model.exog)
        resid = np.asarray(self.results.resid)
        score = exog * (resid / self.results.scale)[:, None]
        if model_cls == "WLS":
            w = getattr(self.results.model, "weights", None)
            if w is not None:
                score = score * np.asarray(w)[:, None]
        return score

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
        """Return Σ̂, dispatching to the requested flavor.

        OLS results expose cov_HC0 / cov_HC1 / cov_HC2 / cov_HC3 as
        attributes regardless of how the model was fit, so we can read
        them directly. Cluster-robust requires refit.
        """
        if vcov_spec is None:
            return jnp.asarray(self.results.cov_params())

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                attr = f"cov_{vcov_spec.upper()}"
                if hasattr(self.results, attr):
                    return jnp.asarray(getattr(self.results, attr))
                raise ValueError(f"{vcov_spec} not available on this fit.")
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
            if kind == "survey":
                return self._survey_covariance(vcov_spec["design"])
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    def _survey_covariance(self, design) -> jnp.ndarray:
        import numpy as np

        from .._inference._linearization import (
            linearization_cov,
            weights_proportional,
        )

        bread = np.asarray(self.results.cov_params())
        scores = self.score_obs()  # (n, p)

        w = np.asarray(design.weights)
        psu = None if design.psu is None else np.asarray(design.psu)
        strata = None if design.strata is None else np.asarray(design.strata)

        if len(w) != scores.shape[0]:
            raise ValueError(
                f"survey weights length {len(w)} != n_obs {scores.shape[0]}; "
                "fit the model on the same rows the design describes."
            )

        # Detect fitting weights (WLS). cov_params() and score_obs() already
        # incorporate them, so we use unit weights to avoid double-counting.
        fit_model = getattr(self.results, "model", None)
        if type(fit_model).__name__ == "WLS":
            fw = getattr(fit_model, "weights", None)
            if fw is not None and not np.allclose(np.asarray(fw), 1.0):
                # If the WLS fit weights are not proportional to the design
                # weights, the variance (fit weights) and the design-weighted
                # point estimate describe different weightings, so warn.
                if not weights_proportional(np.asarray(fw, dtype=float), w):
                    warnings.warn(
                        "The WLS fit weights are not proportional to "
                        "survey_design.weights, so the design-based variance "
                        "(from the fit weights) and the design-weighted point "
                        "estimate may be inconsistent. Fit the model unweighted "
                        "and let survey_design supply the weights, or pass "
                        "matching weights to both.",
                        UserWarning,
                        stacklevel=2,
                    )
                w = np.ones(scores.shape[0])

        fpc_fraction = self._survey_fpc_fraction(design, psu, strata)

        V = linearization_cov(
            bread, scores, w, psu, strata, fpc_fraction, nest=design.nest
        )
        return jnp.asarray(V)

    @staticmethod
    def _survey_fpc_fraction(design, psu, strata):
        import numpy as np

        if design.fpc is None:
            return None
        fpc = np.asarray(design.fpc, dtype=float)
        if design.fpc_is_fraction:
            return fpc
        n = len(fpc)
        if strata is None:
            strata = np.zeros(n, dtype=int)
        if psu is None:
            psu = np.arange(n)
        out = np.zeros(n)
        for h in np.unique(strata):
            in_h = strata == h
            n_h = len(np.unique(psu[in_h]))
            N_h = float(fpc[in_h][0])
            out[in_h] = n_h / N_h
        return out

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        return design_matrix_from_df(
            self.results, self._exog_names, df, formula_spec=self._formula_spec
        )

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
        """Refit the model with a specific cov_type and return its covariance."""
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
            if model_cls_name == "WLS":
                from statsmodels.formula.api import wls as smf_wls

                weights = getattr(self.results.model, "weights", None)
                new_results = smf_wls(
                    formula,
                    data=self._training_data,
                    weights=weights,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})
            elif model_cls_name == "GLS":
                from statsmodels.formula.api import gls as smf_gls

                sigma = getattr(self.results.model, "sigma", None)
                new_results = smf_gls(
                    formula,
                    data=self._training_data,
                    sigma=sigma,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})
            else:
                from statsmodels.formula.api import ols as smf_ols

                new_results = smf_ols(
                    formula,
                    data=self._training_data,
                ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})
            return jnp.asarray(new_results.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog
        model_cls_name = type(self.results.model).__name__
        if model_cls_name == "WLS":
            weights = getattr(self.results.model, "weights", None)
            new_results = sm.WLS(endog, exog, weights=weights).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
            )
        elif model_cls_name == "GLS":
            sigma = getattr(self.results.model, "sigma", None)
            new_results = sm.GLS(endog, exog, sigma=sigma).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
            )
        else:
            new_results = sm.OLS(endog, exog).fit(
                cov_type=cov_type,
                cov_kwds=cov_kwds or {},
            )
        return jnp.asarray(new_results.cov_params())

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsOLSAdapter:
        """Refit the model on resampled data."""
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            model_cls_name = type(self.results.model).__name__
            if model_cls_name == "WLS":
                from statsmodels.formula.api import wls as smf_wls

                weights = getattr(self.results.model, "weights", None)
                new_results = smf_wls(
                    formula, data=resampled_data, weights=weights
                ).fit()
            elif model_cls_name == "GLS":
                from statsmodels.formula.api import gls as smf_gls

                sigma = getattr(self.results.model, "sigma", None)
                new_results = smf_gls(formula, data=resampled_data, sigma=sigma).fit()
            else:
                from statsmodels.formula.api import ols as smf_ols

                new_results = smf_ols(formula, data=resampled_data).fit()
            return StatsmodelsOLSAdapter(new_results, training_data=resampled_data)

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
                    "Array-fit refit requires the response variable name."
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
            if "const" not in exog_df.columns and "Intercept" not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)
        model_cls_name = type(self.results.model).__name__
        if model_cls_name == "WLS":
            weights = getattr(self.results.model, "weights", None)
            if weights is not None and index is not None:
                weights = np.asarray(weights)[index]
            new_results = sm.WLS(endog, exog_df, weights=weights).fit()
        elif model_cls_name == "GLS":
            sigma = getattr(self.results.model, "sigma", None)
            if sigma is not None and index is not None:
                sigma = np.asarray(sigma)[np.ix_(index, index)]
            new_results = sm.GLS(endog, exog_df, sigma=sigma).fit()
        else:
            new_results = sm.OLS(endog, exog_df).fit()
        return StatsmodelsOLSAdapter(new_results, training_data=resampled_data)
